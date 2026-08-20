"""Offline master chain.

The chain is **declarative data**, not code: a list of stages with parameters,
living in ``master.yaml``. That matters because the same definition has more
than one consumer:

* today - rendered offline here with ffmpeg, so the phone preview is a
  *mastered* preview rather than a raw one;
* later - pushed onto the Cubase Stereo Out, either as a generated FX Chain
  preset (one click, no scripting) or by the MIDI Remote script (Tier 3).

Keeping it as data is what stops those from drifting apart. Write the chain
once; the destinations are just renderers of it.

**This is not the master.** It is a preview master, through generic ffmpeg DSP,
of audio that came from a General MIDI soundfont. It tells you whether an
arrangement holds up at competitive loudness. It tells you nothing about how
your actual plugins on your actual mix will behave.

Parameter units are a trap and are handled here once. ffmpeg's ``acompressor``
``makeup`` is a **linear factor 1-64** and ``alimiter`` ``limit`` is **linear
0.0625-1**, neither of which takes a dB suffix - so dB comes in from the spec
and is converted here. ``threshold`` does accept ``dB`` and is passed through.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class MasterError(Exception):
    pass


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


@dataclass
class Loudness:
    """EBU R128 measurement."""

    integrated_lufs: Optional[float] = None
    true_peak_db: Optional[float] = None
    range_lu: Optional[float] = None

    def __str__(self) -> str:
        def fmt(value, unit):
            return f"{value:.1f} {unit}" if value is not None else "?"
        return (
            f"{fmt(self.integrated_lufs, 'LUFS')}, "
            f"peak {fmt(self.true_peak_db, 'dBTP')}, "
            f"range {fmt(self.range_lu, 'LU')}"
        )


@dataclass
class MasterResult:
    source: Path
    output: Path
    filter_chain: str
    before: Loudness = field(default_factory=Loudness)
    after: Loudness = field(default_factory=Loudness)
    stages: List[str] = field(default_factory=list)
    two_pass: bool = False


# --------------------------------------------------------------- stage -> DSP

def _stage_highpass(p: Dict[str, Any]) -> str:
    return f"highpass=f={float(p.get('frequency', 30)):g}:p={int(p.get('poles', 2))}"


def _stage_lowpass(p: Dict[str, Any]) -> str:
    return f"lowpass=f={float(p.get('frequency', 18000)):g}:p={int(p.get('poles', 2))}"


def _stage_eq(p: Dict[str, Any]) -> str:
    return (
        f"equalizer=f={float(p.get('frequency', 1000)):g}"
        f":t=q:w={float(p.get('q', 1.0)):g}"
        f":g={float(p.get('gain_db', 0.0)):g}"
    )


def _stage_compressor(p: Dict[str, Any]) -> str:
    makeup = db_to_linear(float(p.get("makeup_db", 0.0)))
    # ffmpeg clamps makeup to [1, 64]; a negative makeup_db is not expressible.
    makeup = min(64.0, max(1.0, makeup))
    return (
        f"acompressor=threshold={float(p.get('threshold_db', -18)):g}dB"
        f":ratio={float(p.get('ratio', 3)):g}"
        f":attack={float(p.get('attack_ms', 20)):g}"
        f":release={float(p.get('release_ms', 200)):g}"
        f":makeup={makeup:g}"
        f":knee={float(p.get('knee_db', 2.8)):g}"
    )


def _stage_limiter(p: Dict[str, Any]) -> str:
    ceiling = db_to_linear(float(p.get("ceiling_db", -1.0)))
    ceiling = min(1.0, max(0.0625, ceiling))
    return (
        f"alimiter=limit={ceiling:g}"
        f":attack={float(p.get('attack_ms', 5)):g}"
        f":release={float(p.get('release_ms', 50)):g}"
    )


def _stage_loudness(p: Dict[str, Any], measured: Optional[Dict[str, Any]] = None,
                    print_json: bool = False) -> str:
    """EBU R128 normalisation.

    Single-pass loudnorm is a live estimator and routinely lands more than a
    LU off target - which makes twenty song previews unfair to A/B against each
    other. So :func:`master` runs it twice: once to measure what actually
    reaches this stage, then again with those measurements supplied, which hits
    the target exactly and in linear mode.
    """
    parts = [
        f"loudnorm=I={float(p.get('target_lufs', -14)):g}",
        f"TP={float(p.get('true_peak_db', -1)):g}",
        f"LRA={float(p.get('range_lu', 11)):g}",
    ]
    if measured:
        parts += [
            f"measured_I={measured['input_i']}",
            f"measured_TP={measured['input_tp']}",
            f"measured_LRA={measured['input_lra']}",
            f"measured_thresh={measured['input_thresh']}",
            f"offset={measured['target_offset']}",
            "linear=true",
        ]
    if print_json:
        parts.append("print_format=json")
    return ":".join(parts)


STAGES = {
    "highpass": _stage_highpass,
    "lowpass": _stage_lowpass,
    "eq": _stage_eq,
    "compressor": _stage_compressor,
    "limiter": _stage_limiter,
    "loudness": _stage_loudness,
}


def build_filter_chain(
    chain: Sequence[Dict[str, Any]],
    measured: Optional[Dict[str, Any]] = None,
    print_json: bool = False,
) -> str:
    """Declarative stage list -> an ffmpeg -af string."""
    parts: List[str] = []
    for entry in chain:
        if not isinstance(entry, dict):
            raise MasterError(f"master stage must be a mapping, got {entry!r}")
        if entry.get("enabled") is False:
            continue
        name = entry.get("stage")
        if name not in STAGES:
            raise MasterError(
                f"unknown master stage {name!r}. Known: {', '.join(sorted(STAGES))}"
            )
        params = {k: v for k, v in entry.items() if k not in ("stage", "enabled")}
        if name == "loudness":
            parts.append(_stage_loudness(params, measured=measured, print_json=print_json))
        else:
            parts.append(STAGES[name](params))
    return ",".join(parts)


def has_loudness_stage(chain: Sequence[Dict[str, Any]]) -> bool:
    return any(
        isinstance(e, dict) and e.get("stage") == "loudness" and e.get("enabled") is not False
        for e in chain
    )


def _measure_for_loudnorm(source: Path, chain: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pass one: what actually arrives at the loudness stage.

    Everything upstream of loudnorm changes the signal, so the measurement has
    to be taken through the chain, not off the raw file.
    """
    filter_chain = build_filter_chain(chain, print_json=True)
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(source),
         "-af", filter_chain, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    start = proc.stderr.rfind("{")
    end = proc.stderr.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(proc.stderr[start : end + 1])
    except json.JSONDecodeError:
        return None
    required = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    if not all(k in data for k in required):
        return None
    # A silent or near-silent input measures as -inf and cannot be normalised.
    if any(str(data[k]).lstrip("-").startswith("inf") for k in required):
        return None
    return data


def stage_names(chain: Sequence[Dict[str, Any]]) -> List[str]:
    return [
        str(e.get("stage"))
        for e in chain
        if isinstance(e, dict) and e.get("enabled") is not False
    ]


# ------------------------------------------------------------------ measuring

# ebur128 prints a running measurement for every frame AND a Summary block at
# the end. The per-frame lines carry the same field names, and the first ones
# report -70 LUFS because there is not yet enough audio to integrate. Parsing
# the whole stream therefore reads the startup value, not the result - so
# anchor on the Summary block and parse only what follows it.
_EBUR128_SUMMARY = re.compile(
    r"I:\s*(-?[\d.]+)\s*LUFS.*?LRA:\s*(-?[\d.]+)\s*LU.*?Peak:\s*(-?[\d.]+)\s*dBFS",
    re.DOTALL,
)


def measure(path: Path | str) -> Loudness:
    """Measure integrated loudness, range and true peak with ffmpeg's ebur128."""
    if not shutil.which("ffmpeg"):
        return Loudness()
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    _, separator, summary = proc.stderr.rpartition("Summary:")
    if not separator:
        return Loudness()
    match = _EBUR128_SUMMARY.search(summary)
    if not match:
        return Loudness()
    return Loudness(
        integrated_lufs=float(match.group(1)),
        range_lu=float(match.group(2)),
        true_peak_db=float(match.group(3)),
    )


# ------------------------------------------------------------------ rendering

def master(
    source: Path | str,
    output: Path | str,
    chain: Sequence[Dict[str, Any]],
    bitrate: str = "192k",
    dry_run: bool = False,
    two_pass: bool = True,
) -> MasterResult:
    """Apply the master chain to an audio file."""
    source, output = Path(source), Path(output)
    if not source.exists():
        raise MasterError(f"no such audio file: {source}")
    if not shutil.which("ffmpeg"):
        raise MasterError("ffmpeg not on PATH; needed for the master chain")

    measured = None
    if two_pass and has_loudness_stage(chain) and not dry_run:
        measured = _measure_for_loudnorm(source, chain)

    filter_chain = build_filter_chain(chain, measured=measured)
    result = MasterResult(
        source=source,
        output=output,
        filter_chain=filter_chain,
        stages=stage_names(chain),
        two_pass=measured is not None,
    )
    if dry_run:
        return result

    result.before = measure(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source)]
    if filter_chain:
        cmd += ["-af", filter_chain]
    cmd += ["-b:a", bitrate, str(output)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not output.exists():
        raise MasterError(f"master chain failed:\n{proc.stderr.strip()[:2000]}")

    result.after = measure(output)
    return result


# -------------------------------------------------------------------- loading

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHAIN_FILE = REPO_ROOT / "master.yaml"


def load_chain(path: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    """Read the master chain from master.yaml. Missing file means no chain."""
    import yaml

    p = Path(path) if path else DEFAULT_CHAIN_FILE
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text()) or {}
    return list(data.get("chain") or [])

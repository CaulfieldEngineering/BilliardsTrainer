"""``spec.yaml`` - the per-song source of truth.

Design rule: the spec is the master and the PM tool is a *view*. Status lives in
git next to the song, so a merge conflict is visible and a history is free.

The loader is deliberately non-destructive: it keeps the raw dict and writes it
back on save, so a key this code does not understand yet survives a round trip
instead of being silently deleted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Pipeline stages, in order. A song's stage index is how "done" it is.
STAGES: List[str] = [
    "idea",
    "arranged",
    "drums-programmed",
    "scratch-tracked",
    "guitars-tracked",
    "vocals",
    "mixed",
    "mastered",
    "done",
]

# Stages at which a human with a guitar in hand is the bottleneck. Used by the
# "what's ready for tracking" phone view.
TRACKING_STAGES = {"scratch-tracked", "guitars-tracked", "vocals"}

CURRENT_TEMPLATE_REVISION = 1


@dataclass(frozen=True)
class SpecIssue:
    level: str  # "error" | "warn"
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper():5}] {self.field}: {self.message}"


@dataclass
class Spec:
    """A loaded song spec. ``raw`` is authoritative; the properties are sugar."""

    raw: Dict[str, Any]
    path: Optional[Path] = None

    # ------------------------------------------------------------- accessors

    @property
    def slug(self) -> str:
        if self.raw.get("slug"):
            return str(self.raw["slug"])
        return self.path.parent.name if self.path else "unknown"

    @property
    def title(self) -> str:
        return str(self.raw.get("title") or self.slug)

    @property
    def dir(self) -> Path:
        return self.path.parent if self.path else Path(".")

    @property
    def stage(self) -> str:
        return str(self.raw.get("status", {}).get("stage", "idea"))

    @property
    def stage_index(self) -> int:
        try:
            return STAGES.index(self.stage)
        except ValueError:
            return -1

    @property
    def blocked_by(self) -> Optional[str]:
        value = self.raw.get("status", {}).get("blocked_by")
        return str(value) if value else None

    @property
    def is_blocked(self) -> bool:
        return self.blocked_by is not None

    @property
    def next_action(self) -> Optional[str]:
        value = self.raw.get("status", {}).get("next_action")
        return str(value) if value else None

    @property
    def bpm(self) -> Optional[float]:
        value = self.raw.get("musical", {}).get("bpm")
        return float(value) if value is not None else None

    @property
    def time_signature(self) -> List[int]:
        value = self.raw.get("musical", {}).get("time_signature") or [4, 4]
        return [int(value[0]), int(value[1])]

    @property
    def beats_per_bar(self) -> int:
        num, den = self.time_signature
        return max(1, int(round(num * 4 / den)))

    @property
    def key(self) -> Optional[str]:
        value = self.raw.get("musical", {}).get("key")
        return str(value) if value else None

    @property
    def template_revision(self) -> int:
        return int(self.raw.get("template", {}).get("revision", 0))

    @property
    def pending_migrations(self) -> List[str]:
        value = self.raw.get("template", {}).get("pending_migrations") or []
        return [str(v) for v in value]

    @property
    def seed(self) -> str:
        """Deterministic seed for this song's transforms.

        Defaults to the slug so two songs humanise differently but one song
        humanises the same way forever.
        """
        return str(self.raw.get("build", {}).get("seed") or self.slug)

    def sources(self) -> Dict[str, Path]:
        out: Dict[str, Path] = {}
        for name, rel in (self.raw.get("sources") or {}).items():
            if rel:
                out[str(name)] = self.dir / str(rel)
        return out

    def declared_sections(self) -> List[Any]:
        """Sections the operator wrote in spec.yaml, as Section objects.

        These are the arrangement's real names - "Verse 1", "Chorus" - and they
        beat anything detection infers. Detection only ever proposes; this is
        what actually becomes markers. Empty means "fall back to detection".
        """
        from .sections import Section

        out = []
        for entry in self.raw.get("sections") or []:
            if not isinstance(entry, dict) or not entry.get("label"):
                continue
            start = int(entry.get("start_bar", 1)) - 1  # spec.yaml is 1-indexed
            length = int(entry.get("length_bars", 4))
            out.append(
                Section(
                    label=str(entry["label"]),
                    start_bar=max(0, start),
                    end_bar=max(0, start) + max(1, length),
                    fingerprint="",
                    is_fill=bool(entry.get("fill", False)),
                )
            )
        return out

    def build_targets(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.raw.get("build", {}).get("targets") or {})

    @property
    def build_dir(self) -> Path:
        return self.dir / "build"

    # ------------------------------------------------------------------- io

    @classmethod
    def load(cls, path: Path | str) -> "Spec":
        p = Path(path)
        if p.is_dir():
            p = p / "spec.yaml"
        data = yaml.safe_load(p.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{p}: spec must be a YAML mapping, got {type(data).__name__}")
        return cls(raw=data, path=p)

    def save(self, path: Optional[Path] = None) -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no path to save to")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=True, width=100)
        )
        return target

    def set_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}")
        self.raw.setdefault("status", {})["stage"] = stage

    # ------------------------------------------------------------ validation

    def validate(self) -> List[SpecIssue]:
        issues: List[SpecIssue] = []

        if not self.raw.get("title"):
            issues.append(SpecIssue("warn", "title", "no title; falling back to the slug"))

        if self.stage not in STAGES:
            issues.append(
                SpecIssue("error", "status.stage", f"{self.stage!r} is not a known stage ({', '.join(STAGES)})")
            )

        if self.bpm is None:
            issues.append(SpecIssue("warn", "musical.bpm", "no tempo set; builds will use the source file's tempo"))
        elif not 20 <= self.bpm <= 400:
            issues.append(SpecIssue("error", "musical.bpm", f"{self.bpm} is outside a plausible range"))

        sig = self.time_signature
        if len(sig) != 2 or sig[1] not in (1, 2, 4, 8, 16, 32):
            issues.append(SpecIssue("error", "musical.time_signature", f"{sig!r} is not a valid time signature"))

        for name, path in self.sources().items():
            if not path.exists():
                issues.append(SpecIssue("error", f"sources.{name}", f"missing file: {path}"))

        for name, target in self.build_targets().items():
            source = target.get("source")
            if source and source not in (self.raw.get("sources") or {}):
                issues.append(
                    SpecIssue("error", f"build.targets.{name}.source", f"references undefined source {source!r}")
                )

        if self.template_revision < CURRENT_TEMPLATE_REVISION and not self.pending_migrations:
            issues.append(
                SpecIssue(
                    "warn",
                    "template.revision",
                    f"song is on template r{self.template_revision}, current is "
                    f"r{CURRENT_TEMPLATE_REVISION}, and no migrations are listed",
                )
            )

        return issues

    def errors(self) -> List[SpecIssue]:
        return [i for i in self.validate() if i.level == "error"]


def discover(songs_dir: Path | str) -> List[Spec]:
    """Load every song spec under ``songs/``, skipping ``_``-prefixed dirs."""
    root = Path(songs_dir)
    specs: List[Spec] = []
    if not root.exists():
        return specs
    for spec_path in sorted(root.glob("*/spec.yaml")):
        if spec_path.parent.name.startswith("_"):
            continue
        specs.append(Spec.load(spec_path))
    return specs

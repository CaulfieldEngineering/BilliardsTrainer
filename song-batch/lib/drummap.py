"""Load and query ``maps/ssd.json`` - the single source of truth for GM -> SSD.

Nothing else in the repo is allowed to hardcode an SSD note number. If you find
yourself typing a bare integer drum note anywhere outside maps/, that is a bug.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP = REPO_ROOT / "maps" / "ssd.json"
GM_REFERENCE = REPO_ROOT / "maps" / "gm.json"


class DrumMapError(Exception):
    """The map file is structurally unusable."""


@dataclass(frozen=True)
class Target:
    """Where an incoming GM note ends up."""

    note: int
    articulation: str
    confidence: str
    via: str  # "conditional" | "entry" | "passthrough"

    @property
    def is_verified(self) -> bool:
        return self.confidence == "verified"


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warn" | "info"
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper():5}] {self.code}: {self.message}"


class DrumMap:
    """A loaded GM -> SSD note map."""

    def __init__(self, data: dict, path: Optional[Path] = None):
        self.data = data
        self.path = path
        self.unmapped_policy: str = data.get("unmapped_policy", "keep")
        if self.unmapped_policy not in ("keep", "drop"):
            raise DrumMapError(
                f"unmapped_policy must be 'keep' or 'drop', got {self.unmapped_policy!r}"
            )

        self._entries: Dict[int, dict] = {}
        for entry in data.get("entries", []):
            gm = entry.get("gm")
            if not isinstance(gm, int) or not 0 <= gm <= 127:
                raise DrumMapError(f"entry has a bad 'gm' note number: {entry!r}")
            if gm in self._entries:
                raise DrumMapError(
                    f"GM note {gm} is mapped twice; a note may appear at most once "
                    f"in 'entries' (use 'conditional.rules' for velocity splits)"
                )
            self._entries[gm] = entry

        self._articulations: Dict[str, dict] = {
            k: v
            for k, v in data.get("articulations", {}).items()
            if not k.startswith("_") and isinstance(v, dict)
        }
        self._rules: List[dict] = [
            r for r in data.get("conditional", {}).get("rules", []) if isinstance(r, dict)
        ]

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: Path | str | None = None) -> "DrumMap":
        p = Path(path) if path else DEFAULT_MAP
        if not p.exists():
            raise DrumMapError(f"drum map not found: {p}")
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            raise DrumMapError(f"{p} is not valid JSON: {exc}") from exc
        return cls(data, p)

    @staticmethod
    def gm_names() -> Dict[int, str]:
        if not GM_REFERENCE.exists():
            return {}
        data = json.loads(GM_REFERENCE.read_text())
        return {int(k): v for k, v in data.get("names", {}).items()}

    # ---------------------------------------------------------------- queries

    def articulation_note(self, name: str) -> Optional[int]:
        """Resolve an articulation name to a note number, or None if unfilled.

        Looks in the named ``articulations`` block first, then falls back to any
        flat entry carrying that articulation name - so ``snare_center``
        resolves even though it only exists as a GM entry.
        """
        art = self._articulations.get(name)
        if art is not None and isinstance(art.get("to"), int):
            return art["to"]
        for entry in self._entries.values():
            if entry.get("articulation") == name and isinstance(entry.get("to"), int):
                return entry["to"]
        return None

    def lookup(self, note: int, velocity: int = 64) -> Optional[Target]:
        """Resolve one GM note+velocity to its SSD target.

        Returns None to mean "drop this note". Resolution order:
          1. enabled conditional rules whose velocity window matches
          2. the flat entry for that GM note
          3. ``unmapped_policy`` (keep = pass the note through untouched)

        A conditional rule pointing at an unfilled articulation is skipped, not
        an error - that is what lets the map ship half-verified.
        """
        for rule in self._rules:
            if not rule.get("enabled"):
                continue
            if rule.get("gm") != note:
                continue
            window = rule.get("when", {}).get("velocity")
            if window and not (window[0] <= velocity <= window[1]):
                continue
            target_note = None
            if isinstance(rule.get("to"), int):
                target_note = rule["to"]
            elif rule.get("to_articulation"):
                target_note = self.articulation_note(rule["to_articulation"])
            if target_note is None:
                continue  # unfilled articulation - fall through to the flat map
            return Target(
                note=target_note,
                articulation=rule.get("to_articulation", "conditional"),
                confidence=rule.get("confidence", "provisional"),
                via="conditional",
            )

        entry = self._entries.get(note)
        if entry is not None and isinstance(entry.get("to"), int):
            return Target(
                note=entry["to"],
                articulation=entry.get("articulation", "?"),
                confidence=entry.get("confidence", "provisional"),
                via="entry",
            )

        if self.unmapped_policy == "drop":
            return None
        return Target(note=note, articulation="passthrough", confidence="unmapped", via="passthrough")

    # ---------------------------------------------------------------- doctor

    def doctor(self) -> List[Issue]:
        """Report what is wrong, unverified, or still awaiting the operator.

        This is the function that keeps the map honest. `./sb map doctor` runs
        it; the build runs it too and refuses to emit a "verified" build while
        errors are outstanding.
        """
        issues: List[Issue] = []
        gm_names = self.gm_names()

        if not self.data.get("verified", False):
            issues.append(
                Issue(
                    "warn",
                    "map-unverified",
                    "ssd.json is marked verified:false - note numbers are educated "
                    "guesses at SSD's GM-compatible layout, not checked against the kit.",
                )
            )

        for gm, entry in sorted(self._entries.items()):
            to = entry.get("to")
            if to is None:
                issues.append(
                    Issue("warn", "entry-unfilled", f"GM {gm} ({gm_names.get(gm, '?')}) has no target note.")
                )
            elif not isinstance(to, int) or not 0 <= to <= 127:
                issues.append(
                    Issue("error", "entry-bad-note", f"GM {gm} maps to {to!r}, which is not a MIDI note 0-127.")
                )
            if entry.get("confidence") == "provisional":
                issues.append(
                    Issue("info", "entry-provisional", f"GM {gm} -> {to} ({entry.get('articulation')}) is unconfirmed.")
                )

        for name, art in sorted(self._articulations.items()):
            if art.get("to") is None:
                used_by = art.get("used_by")
                suffix = f" Needed by: {used_by}." if used_by else ""
                issues.append(
                    Issue("info", "articulation-awaiting", f"Articulation '{name}' has no note number yet.{suffix}")
                )

        for rule in self._rules:
            if not rule.get("enabled"):
                continue
            art = rule.get("to_articulation")
            if art and self.articulation_note(art) is None:
                issues.append(
                    Issue(
                        "warn",
                        "rule-dangling",
                        f"Conditional rule for GM {rule.get('gm')} is enabled but its target "
                        f"articulation '{art}' has no note number - the rule will be skipped.",
                    )
                )

        for group, members in self.data.get("choke_groups", {}).items():
            if group.startswith("_") or not isinstance(members, list):
                continue
            for member in members:
                if self.articulation_note(member) is None:
                    issues.append(
                        Issue("info", "choke-unresolved", f"Choke group '{group}' references unfilled '{member}'.")
                    )

        return issues

    def errors(self) -> List[Issue]:
        return [i for i in self.doctor() if i.level == "error"]

    def summary(self) -> str:
        filled = sum(1 for e in self._entries.values() if isinstance(e.get("to"), int))
        arts = len(self._articulations)
        arts_filled = sum(1 for a in self._articulations.values() if isinstance(a.get("to"), int))
        state = "verified" if self.data.get("verified") else "PROVISIONAL"
        return (
            f"{self.data.get('title', 'drum map')} [{state}] - "
            f"{filled}/{len(self._entries)} GM entries mapped, "
            f"{arts_filled}/{arts} extra articulations filled"
        )

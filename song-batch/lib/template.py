"""The Cubase template, as data.

Reads ``template.yaml``. Its job is to answer two questions: what should a
generated track be called, and in what order do tracks go.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = REPO_ROOT / "template.yaml"


@dataclass(frozen=True)
class TemplateTrack:
    name: str
    index: int
    role: Optional[str] = None
    channel: Optional[int] = None
    type: str = "instrument"


class Template:
    def __init__(self, data: dict, path: Optional[Path] = None):
        self.data = data
        self.path = path
        self.tracks: List[TemplateTrack] = [
            TemplateTrack(
                name=str(entry.get("name", f"Track {i + 1}")),
                index=i,
                role=entry.get("role"),
                channel=entry.get("channel"),
                type=str(entry.get("type", "instrument")),
            )
            for i, entry in enumerate(data.get("tracks", []))
        ]

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Template":
        p = Path(path) if path else DEFAULT_TEMPLATE
        if not p.exists():
            return cls({}, None)
        return cls(yaml.safe_load(p.read_text()) or {}, p)

    @property
    def revision(self) -> int:
        return int(self.data.get("revision", 0))

    @property
    def verified(self) -> bool:
        return bool(self.data.get("verified", False))

    def by_role(self, role: str) -> Optional[TemplateTrack]:
        for track in self.tracks:
            if track.role == role:
                return track
        return None

    def track_name_for(self, role: str) -> str:
        """The template's name for a build target, or the target name itself."""
        track = self.by_role(role)
        return track.name if track else role

    def order_key(self, role: str) -> int:
        """Position in MixConsole order. Unknown roles sort to the far right."""
        track = self.by_role(role)
        return track.index if track else len(self.tracks) + 1

    def summary(self) -> str:
        state = "verified" if self.verified else "PLACEHOLDER"
        return f"template r{self.revision} [{state}] - {len(self.tracks)} tracks"

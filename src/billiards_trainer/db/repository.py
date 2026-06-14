"""Repository: the only module that touches the ORM session.

Owns the SQLite engine, the session lifecycle, shot recording, the stat queries
the UI renders, and CSV/JSON export. Thread-safe enough for our use: the engine
uses ``check_same_thread=False`` and each call opens a short-lived Session.
"""

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from ..config import DB_PATH
from .models import Base, Session, Shot


class Repository:
    def __init__(self, db_path: Path | None = None):
        path = db_path or DB_PATH
        if path != Path(":memory:"):
            path.parent.mkdir(parents=True, exist_ok=True)
        url = "sqlite:///:memory:" if str(path) == ":memory:" else f"sqlite:///{path}"
        self.engine = create_engine(url, future=True,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(self.engine, expire_on_commit=False)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start_session(self, mode: str = "free_play", drill_key: str | None = None,
                      drill_target: int = 0, table_size: str = "9ft") -> int:
        with self._Session() as s:
            row = Session(mode=mode, drill_key=drill_key, drill_target=drill_target,
                          table_size=table_size)
            s.add(row)
            s.commit()
            return row.id

    def end_session(self, session_id: int) -> None:
        with self._Session() as s:
            row = s.get(Session, session_id)
            if row and row.ended_at is None:
                row.ended_at = datetime.now(timezone.utc)
                s.commit()

    def record_shot(self, session_id: int, outcome: str, num_pocketed: int = 0,
                    target_pocket: str | None = None, cue_scratch: bool = False,
                    duration_s: float = 0.0, shot_seconds: float = 0.0) -> int:
        with self._Session() as s:
            # streak index = running count of consecutive makes ending at this shot
            prev = s.execute(
                select(Shot).where(Shot.session_id == session_id).order_by(Shot.id.desc())
            ).scalars().first()
            streak = (prev.streak_index + 1) if (prev and prev.outcome == "make"
                                                 and outcome == "make") else \
                     (1 if outcome == "make" else 0)
            shot = Shot(session_id=session_id, outcome=outcome, num_pocketed=num_pocketed,
                        target_pocket=target_pocket, cue_scratch=cue_scratch,
                        duration_s=duration_s, shot_seconds=shot_seconds, streak_index=streak)
            s.add(shot)
            s.commit()
            return shot.id

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def session_summary(self, session_id: int) -> dict:
        with self._Session() as s:
            shots = s.execute(
                select(Shot).where(Shot.session_id == session_id).order_by(Shot.id)
            ).scalars().all()
            return _summarize(shots)

    def global_summary(self, recent: int = 10) -> dict:
        with self._Session() as s:
            n_sessions = s.execute(select(func.count(Session.id))).scalar_one()
            shots = s.execute(select(Shot).order_by(Shot.id)).scalars().all()
            base = _summarize(shots)
            base["sessions"] = int(n_sessions)

            # recent sessions table
            sess_rows = s.execute(
                select(Session).order_by(Session.id.desc()).limit(recent)
            ).scalars().all()
            recent_sessions = []
            for sess in sess_rows:
                ss = _summarize(sess.shots)
                date = sess.started_at.strftime("%Y-%m-%d %H:%M") if sess.started_at else "—"
                recent_sessions.append([date, sess.mode, ss["shots"], ss["makes"],
                                        f"{ss['make_pct']:.0f}%"])
            base["recent_sessions"] = recent_sessions
            return base

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def export_csv(self, dest: Path) -> Path:
        with self._Session() as s:
            rows = s.execute(
                select(Shot, Session).join(Session, Shot.session_id == Session.id)
                .order_by(Shot.id)
            ).all()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["shot_id", "session_id", "created_at", "mode", "outcome",
                        "num_pocketed", "target_pocket", "cue_scratch", "duration_s",
                        "shot_seconds", "streak_index"])
            for shot, sess in rows:
                w.writerow([shot.id, shot.session_id,
                            shot.created_at.isoformat() if shot.created_at else "",
                            sess.mode, shot.outcome, shot.num_pocketed, shot.target_pocket or "",
                            int(shot.cue_scratch), f"{shot.duration_s:.2f}",
                            f"{shot.shot_seconds:.2f}", shot.streak_index])
        return dest

    def export_json(self, dest: Path) -> Path:
        with self._Session() as s:
            sessions = s.execute(select(Session).order_by(Session.id)).scalars().all()
            payload = []
            for sess in sessions:
                payload.append({
                    "id": sess.id,
                    "started_at": sess.started_at.isoformat() if sess.started_at else None,
                    "ended_at": sess.ended_at.isoformat() if sess.ended_at else None,
                    "mode": sess.mode,
                    "drill_key": sess.drill_key,
                    "table_size": sess.table_size,
                    "shots": [{
                        "id": sh.id, "outcome": sh.outcome, "num_pocketed": sh.num_pocketed,
                        "target_pocket": sh.target_pocket, "cue_scratch": sh.cue_scratch,
                        "duration_s": sh.duration_s, "shot_seconds": sh.shot_seconds,
                        "streak_index": sh.streak_index,
                    } for sh in sess.shots],
                })
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return dest


def _summarize(shots: list[Shot]) -> dict:
    total = len(shots)
    makes = sum(1 for sh in shots if sh.outcome == "make")
    misses = sum(1 for sh in shots if sh.outcome == "miss")
    scratches = sum(1 for sh in shots if sh.outcome == "scratch")
    make_pct = (100.0 * makes / total) if total else 0.0

    best_streak = cur = 0
    for sh in shots:
        if sh.outcome == "make":
            cur += 1
            best_streak = max(best_streak, cur)
        else:
            cur = 0

    by_pocket_makes: dict[str, int] = defaultdict(int)
    by_pocket_att: dict[str, int] = defaultdict(int)
    for sh in shots:
        if sh.target_pocket:
            by_pocket_makes[sh.target_pocket] += 1
            by_pocket_att[sh.target_pocket] += 1
    by_pocket = []
    for pocket in sorted(by_pocket_att):
        m, a = by_pocket_makes[pocket], by_pocket_att[pocket]
        by_pocket.append([pocket, m, a, f"{(100.0 * m / a) if a else 0:.0f}%"])

    avg_shot_time = (sum(sh.shot_seconds for sh in shots) / total) if total else 0.0
    return {
        "shots": total, "makes": makes, "misses": misses, "scratches": scratches,
        "make_pct": make_pct, "current_streak": cur, "best_streak": best_streak,
        "by_pocket": by_pocket, "avg_shot_time": avg_shot_time,
    }

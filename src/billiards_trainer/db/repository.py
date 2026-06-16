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

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import sessionmaker

from ..config import DB_PATH
from .models import Base, Feedback, Session, Shot


class Repository:
    def __init__(self, db_path: Path | None = None):
        path = db_path or DB_PATH
        if path != Path(":memory:"):
            path.parent.mkdir(parents=True, exist_ok=True)
        url = "sqlite:///:memory:" if str(path) == ":memory:" else f"sqlite:///{path}"
        self.engine = create_engine(url, future=True,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self._migrate()
        self._Session = sessionmaker(self.engine, expire_on_commit=False)
        self._seed_devnote()

    def _migrate(self) -> None:
        """Lightweight additive migration: SQLAlchemy create_all never ALTERs, so
        add columns introduced after a user's DB was created."""
        wanted = {"sessions": [("synced", "BOOLEAN DEFAULT 0")],
                  "shots": [("synced", "BOOLEAN DEFAULT 0")]}
        with self.engine.begin() as conn:
            for table, cols in wanted.items():
                existing = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
                for name, ddl in cols:
                    if name not in existing:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))

    # Dev notes carried forward in the local feedback table (idempotent by id).
    _DEVNOTES = [
        (147, "Self-update button + in-app feedback form + Supabase sync skeleton",
         "v0.1.6: explicit Check-for-Updates button, local-first feedback form, "
         "and a config-gated Supabase sync skeleton."),
        (148, "#1 reliability blockers: false-positive make/miss counts + warped overhead",
         "v0.2.x live testing: the make/miss counter incremented on a still table "
         "(classical-CV noise), and the overhead view showed the warped camera. "
         "Fixed by motion-energy-gated shot detection (cue + sustained motion + "
         "ball travel + pocket approach/vanish, warm-up/cool-down, confidence "
         "floor) plus a clean rendered schematic overhead. Classical detection "
         "remains demo-grade; YOLO weights in models/ are the accuracy upgrade."),
        (149, "Classical CV ball detection unusable on real-world camera — "
         "pivot to YOLO with manual mode as graceful default",
         "v0.2.13 live test on Joe's real table: classical detection rendered "
         "balls in wrong colours/sizes, a tiny phantom cue, jittering IDs, and "
         "blobs floating in the pockets. No amount of HSV/Hough tuning fixes this "
         "class of problem on a real overhead camera. v0.2.14 holds the line on "
         "reliability: the app opens straight into a live camera PREVIEW with "
         "auto-detection OFF (clean empty overhead + manual +Make/-Miss), a strict "
         "ball-size band + full pocket-region masking + a 0.85 render floor when "
         "detection is on, and a 'Capture 60s for analysis' button that zips raw "
         "frames for training. v0.2.15: fine-tune YOLO on captured frames "
         "(scripts/label_session.py) and make it the default backend when weights "
         "are present; classical becomes the fallback."),
        (150, "Off-the-shelf detection: COCO doesn't work, fixed yellow/black "
         "balls, faster dev loop",
         "v0.2.15. (1) Off-the-shelf weights: searched + TESTED. Generic COCO "
         "YOLO 'sports ball' detects ZERO top-down pool balls; Roboflow pool "
         "models are API-key-gated; community .pt is unlicensed. No free model "
         "works out of the box. Shipped OnnxYoloDetector (ONNX Runtime, no "
         "torch) so a pool-specific .onnx dropped in models/ just works; "
         "onnxruntime is an optional [onnx] extra, not bundled. (2) Balls were "
         "yellow/black because the schematic used a fixed per-CLASS palette "
         "(SOLID=yellow, EIGHT=black) and ignored the measured tr.bgr; now draws "
         "the real measured colour + grey '?' when uncertain (ui."
         "measured_ball_colors). (3) Faster iteration: run_dev.ps1, "
         "eval_detection --video, feature flags (allow_without_model), README "
         "dev-loop docs, stop local PyInstaller pre-builds (CI builds anyway)."),
        (151, "Roboflow Pool V2: API-key-gated; turnkey local-train path shipped",
         "v0.2.16. Joe sent universe.roboflow.com/pool-table/pool-v2 (751 imgs, "
         "4 classes, mAP@50 66.8%, CC BY 4.0). Verified: Roboflow's public API "
         "and the roboflow SDK both refuse anonymous access ('A valid API key "
         "must be provided'), so neither dataset nor weights download without a "
         "free key; the hosted inference API would stream frames off-device, "
         "breaking the offline rule. Constraint-compliant path: download the "
         "CC-BY dataset (free key) -> train YOLOv8n locally -> export "
         "models/pool_balls.onnx. Shipped tools/train_pool_model.py to do this in "
         "one command. Drop-in verified: a pool_balls.onnx auto-selects over "
         "classical via the v0.2.15 OnnxYoloDetector, fully offline. Blocked only "
         "on a free Roboflow key (Joe's to create) + one-time training compute."),
    ]

    def _seed_devnote(self) -> None:
        with self._Session() as s:
            changed = False
            for nid, title, desc in self._DEVNOTES:
                if s.get(Feedback, nid) is None:
                    s.add(Feedback(id=nid, kind="devnote", title=title,
                                   description=desc, app_version=_app_version(),
                                   synced=False))
                    changed = True
            if changed:
                s.commit()

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


    # ------------------------------------------------------------------ #
    # Feedback
    # ------------------------------------------------------------------ #
    def add_feedback(self, kind: str, title: str, description: str = "",
                     severity: str = "", app_version: str = "", system_info: str = "",
                     attachments: list[str] | None = None) -> int:
        with self._Session() as s:
            row = Feedback(
                kind=kind, title=title[:120], description=description,
                severity=severity, app_version=app_version, system_info=system_info,
                attachments=json.dumps(attachments or []),
            )
            s.add(row)
            s.commit()
            return row.id

    def attach_to_feedback(self, feedback_id: int, path: str) -> None:
        with self._Session() as s:
            row = s.get(Feedback, feedback_id)
            if row is None:
                return
            paths = json.loads(row.attachments or "[]")
            paths.append(path)
            row.attachments = json.dumps(paths)
            row.synced = False  # changed -> needs re-sync
            s.commit()

    def recent_feedback(self, limit: int = 50) -> list[dict]:
        with self._Session() as s:
            rows = s.execute(
                select(Feedback).order_by(Feedback.id.desc()).limit(limit)
            ).scalars().all()
            return [self._feedback_dict(r) for r in rows]

    def unsynced_feedback(self) -> list[dict]:
        with self._Session() as s:
            rows = s.execute(
                select(Feedback).where(Feedback.synced.is_(False)).order_by(Feedback.id)
            ).scalars().all()
            return [self._feedback_dict(r) for r in rows]

    def mark_feedback_synced(self, ids: list[int]) -> None:
        if not ids:
            return
        with self._Session() as s:
            for fid in ids:
                row = s.get(Feedback, fid)
                if row:
                    row.synced = True
            s.commit()

    @staticmethod
    def _feedback_dict(r: Feedback) -> dict:
        return {
            "id": r.id, "created_at": r.created_at.isoformat() if r.created_at else None,
            "kind": r.kind, "title": r.title, "description": r.description,
            "severity": r.severity, "app_version": r.app_version,
            "system_info": r.system_info,
            "attachments": json.loads(r.attachments or "[]"), "synced": r.synced,
        }

    # ------------------------------------------------------------------ #
    # Sync helpers (used by the Supabase sync manager)
    # ------------------------------------------------------------------ #
    def unsynced_sessions(self, limit: int = 500) -> list[dict]:
        with self._Session() as s:
            rows = s.execute(
                select(Session).where(Session.synced.is_(False)).order_by(Session.id).limit(limit)
            ).scalars().all()
            return [{
                "id": r.id,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "ended_at": r.ended_at.isoformat() if r.ended_at else None,
                "mode": r.mode, "drill_key": r.drill_key, "table_size": r.table_size,
            } for r in rows]

    def unsynced_shots(self, limit: int = 2000) -> list[dict]:
        with self._Session() as s:
            rows = s.execute(
                select(Shot).where(Shot.synced.is_(False)).order_by(Shot.id).limit(limit)
            ).scalars().all()
            return [{
                "id": r.id, "session_id": r.session_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "outcome": r.outcome, "num_pocketed": r.num_pocketed,
                "target_pocket": r.target_pocket, "cue_scratch": r.cue_scratch,
                "duration_s": r.duration_s, "shot_seconds": r.shot_seconds,
                "streak_index": r.streak_index,
            } for r in rows]

    def mark_synced(self, table: str, ids: list[int]) -> None:
        if not ids:
            return
        model = {"sessions": Session, "shots": Shot, "feedback": Feedback}.get(table)
        if model is None:
            return
        with self._Session() as s:
            for rid in ids:
                row = s.get(model, rid)
                if row:
                    row.synced = True
            s.commit()


def _app_version() -> str:
    from ..version import __version__
    return __version__


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

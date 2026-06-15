"""ORM models: practice sessions, the shots within them, and user feedback.

Drills are code-defined templates; a drill run is just a session with
``drill_key`` set. The ``synced`` columns drive the optional Supabase backup
(rows start unsynced and are flipped once pushed)."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    mode: Mapped[str] = mapped_column(String(32), default="free_play")
    drill_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    drill_target: Mapped[int] = mapped_column(Integer, default=0)
    table_size: Mapped[str] = mapped_column(String(16), default="9ft")
    notes: Mapped[str] = mapped_column(String(512), default="")
    synced: Mapped[bool] = mapped_column(Boolean, default=False)

    shots: Mapped[list["Shot"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Shot.id"
    )


class Shot(Base):
    __tablename__ = "shots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    outcome: Mapped[str] = mapped_column(String(16))           # make | miss | scratch
    num_pocketed: Mapped[int] = mapped_column(Integer, default=0)
    target_pocket: Mapped[str | None] = mapped_column(String(24), nullable=True)
    cue_scratch: Mapped[bool] = mapped_column(default=False)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    shot_seconds: Mapped[float] = mapped_column(Float, default=0.0)  # time on the clock
    streak_index: Mapped[int] = mapped_column(Integer, default=0)
    synced: Mapped[bool] = mapped_column(Boolean, default=False)

    session: Mapped["Session"] = relationship(back_populates="shots")


class Feedback(Base):
    """A bug report / feature request / dev note submitted from inside the app.

    Stored locally first (source of truth for display) and optionally pushed to
    Supabase by the sync manager once credentials are configured."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    kind: Mapped[str] = mapped_column(String(16), default="bug")  # bug | feature | devnote
    title: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[str] = mapped_column(String(16), default="")  # low | medium | high
    app_version: Mapped[str] = mapped_column(String(32), default="")
    system_info: Mapped[str] = mapped_column(String(512), default="")
    attachments: Mapped[str] = mapped_column(Text, default="")     # JSON list of local paths
    synced: Mapped[bool] = mapped_column(Boolean, default=False)

"""ORM models. Two tables are enough for the MVP: practice sessions and the
shots within them. Drills are code-defined templates; a drill run is just a
session with ``drill_key`` set and a target."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
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

    session: Mapped["Session"] = relationship(back_populates="shots")

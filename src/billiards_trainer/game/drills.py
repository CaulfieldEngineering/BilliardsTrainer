"""Drill templates for Practice/Drill mode.

A drill is a lightweight target structure: a name, a description of the setup,
and a completion goal (number of makes). The event layer counts makes against
the active drill; nothing here touches CV directly.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Drill:
    key: str
    name: str
    description: str
    difficulty: str          # beginner | intermediate | advanced
    target_makes: int
    balls: int               # how many object balls the setup uses
    tags: tuple[str, ...] = ()


DRILL_TEMPLATES: list[Drill] = [
    Drill(
        key="straight_in",
        name="Straight-In Repeater",
        description="Place the object ball and cue ball in a straight line to a "
                    "corner pocket. Pot it 10 times in a row to build a pure stroke.",
        difficulty="beginner",
        target_makes=10,
        balls=1,
        tags=("fundamentals", "potting"),
    ),
    Drill(
        key="stop_shot",
        name="Stop-Shot Control",
        description="Pot the object ball and stop the cue dead. Trains centre-ball "
                    "contact and speed control.",
        difficulty="beginner",
        target_makes=8,
        balls=1,
        tags=("cue control",),
    ),
    Drill(
        key="long_straight",
        name="Long Straight Pot",
        description="Full-table straight pot. Object ball on the foot spot, cue on "
                    "the head spot. Unforgiving alignment test.",
        difficulty="intermediate",
        target_makes=7,
        balls=1,
        tags=("potting", "alignment"),
    ),
    Drill(
        key="cut_ladder",
        name="Cut Angle Ladder",
        description="Progressively wider cut angles to the same pocket. Pot each "
                    "angle to climb the ladder.",
        difficulty="intermediate",
        target_makes=6,
        balls=1,
        tags=("potting", "angles"),
    ),
    Drill(
        key="l_drill",
        name="The L Drill",
        description="Balls along two rails forming an L. Run them in order with "
                    "position play between shots.",
        difficulty="advanced",
        target_makes=6,
        balls=6,
        tags=("position", "run-out"),
    ),
    Drill(
        key="ghost_ball_5",
        name="Playing the Ghost (5)",
        description="Break and run 5 balls against the 'ghost'. Any miss restarts "
                    "the rack. Classic solo benchmark.",
        difficulty="advanced",
        target_makes=5,
        balls=5,
        tags=("run-out", "pressure"),
    ),
]

DRILLS_BY_KEY = {d.key: d for d in DRILL_TEMPLATES}


def get_drill(key: str) -> Drill | None:
    return DRILLS_BY_KEY.get(key)

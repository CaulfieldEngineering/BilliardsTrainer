"""In-app active-learning for ball identity.

The colour heuristic can't reliably tell adjacent-coloured / same-hue balls apart
on a given table, and a model trained on other tables doesn't transfer. So the
robust path is a model fine-tuned on THIS table — and because the camera/angle/
lighting can change, that has to be a thing the user can redo. This package is the
loop: the model guesses ball numbers, the user corrects them, the corrections
accumulate in a persistent store (the "foundation"), and a fine-tune turns them
into an updated model the app uses.
"""

from .store import TrainingStore

__all__ = ["TrainingStore"]

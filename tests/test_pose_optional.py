"""The optional pose module must import cleanly and degrade gracefully when
MediaPipe isn't installed (the default base install)."""

from billiards_trainer.pose import PoseAnalyzer, is_available


def test_is_available_returns_bool():
    assert isinstance(is_available(), bool)


def test_analyzer_raises_when_unavailable():
    if is_available():
        import pytest
        pytest.skip("mediapipe installed; unavailability path not exercised")
    import pytest
    with pytest.raises(RuntimeError):
        PoseAnalyzer()

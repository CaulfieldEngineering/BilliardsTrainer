"""Shared pytest fixtures: the labelled table capture and its ground truth."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
CORNER_ORDER = ("TL", "TR", "BR", "BL")


@pytest.fixture(scope="session")
def table_frame() -> np.ndarray:
    img = cv2.imread(str(FIXTURES / "table_live.png"))
    assert img is not None, "table_live.png fixture missing"
    return img


@pytest.fixture(scope="session")
def table_truth() -> dict:
    data = json.loads((FIXTURES / "table_fixture.json").read_text())
    cp = data["felt"]["corners_px"]
    corners = np.array([[cp[k]["x"], cp[k]["y"]] for k in CORNER_ORDER], dtype=np.float32)
    return {
        "corners": corners,
        "H": np.array(data["rectification"]["H"], dtype=np.float64),
        "dst_size": (data["rectification"]["dstSize"]["w"], data["rectification"]["dstSize"]["h"]),
    }

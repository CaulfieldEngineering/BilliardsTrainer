"""Vision-language-model client for auto-labelling ball crop sheets.

Given a crop sheet (a grid of enlarged ball crops captioned #0, #1, …) the model
returns each cell's ball number. Backend is OpenRouter (OpenAI-compatible chat
completions with image content), config-gated via ``AutoLabelSettings``. Uses
``requests`` (already a dependency); no SDK. Degrades gracefully — a missing key
or backend 'off' returns an empty mapping so the caller falls back to manual.
"""

from __future__ import annotations

import base64
import json
import logging
import re

import cv2
import numpy as np
import requests

from ..config import AutoLabelSettings

log = logging.getLogger("train.vlm")

_PROMPT = (
    "This is a grid of cropped pool balls from an overhead camera, one ball per "
    "cell. Each cell is captioned with an index like #0, #1. Identify each ball "
    "by its number using colour + pattern:\n"
    "SOLIDS 1-8: 1=yellow, 2=blue, 3=red, 4=purple, 5=orange, 6=green, "
    "7=maroon/burgundy, 8=black. STRIPES 9-15: white ball with a coloured stripe "
    "of the same colour order (9=yellow,10=blue,11=red,12=purple,13=orange,"
    "14=green,15=maroon). The plain WHITE ball is the cue = 0. If a cell is NOT a "
    "ball (a pocket, a hand, felt, blur), use -1.\n"
    "Reply with ONLY a JSON object mapping each index to its number, e.g. "
    '{"0": 3, "1": 0, "2": -1}. No prose.'
)


def available(s: AutoLabelSettings) -> bool:
    return (s.backend or "off").lower() == "openrouter" and bool(s.api_key.strip())


def label_crop_sheet(sheet_bgr: np.ndarray, s: AutoLabelSettings,
                     timeout: float = 90.0) -> dict[int, int]:
    """Return {detection_index: ball_number} for a crop sheet, or {} on failure."""
    if not available(s):
        return {}
    ok, buf = cv2.imencode(".png", sheet_bgr)
    if not ok:
        return {}
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    payload = {
        "model": s.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {s.api_key.strip()}",
               "Content-Type": "application/json"}
    try:
        r = requests.post(s.endpoint, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        log.warning("VLM request failed: %s", exc)
        return {}
    return _parse(text)


def _parse(text: str) -> dict[int, int]:
    """Pull the {index: number} JSON out of the model's reply, tolerating fences."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        raw = json.loads(m.group(0))
    except ValueError:
        return {}
    out: dict[int, int] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out

"""Per-shot thumbnails: each shot-list row shows its table at strike time.

Dossier polish (G6): scanning a session by eye beats reading timestamps.
Extraction is a pure function over cv2 (testable, no Qt); the panel turns
frames into icons with the outcome painted as a left-edge bar so the
thumbnail and the outcome colour share the row's single icon slot.

Threading contract: ``extract_thumbs`` is called from a worker thread and
returns plain BGR arrays; ONLY the UI thread converts them to pixmaps.
"""

import cv2
import numpy as np

#: Icon geometry. 16:9-ish, small enough that 60 rows stay light (~25 KB).
THUMB_W, THUMB_H = 96, 54


def extract_thumbs(video_path: str, times: list[float],
                   size: tuple[int, int] = (THUMB_W, THUMB_H)) -> dict[float, np.ndarray]:
    """{start_time: small BGR frame} for each requested shot start.

    One sequential pass in time order (seeks on long GOP video are cheap
    forward, expensive backward). A frame that fails to decode is simply
    absent — the row keeps its dot icon."""
    out: dict[float, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return out
    for t in sorted(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out[t] = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    cap.release()
    return out

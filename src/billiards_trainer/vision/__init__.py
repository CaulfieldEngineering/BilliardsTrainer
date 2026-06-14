"""Computer-vision pipeline: felt/table detection, rectification, ball detection,
tracking, and the per-frame orchestration that produces structured table state.

All heavy imports (cv2, numpy) live in the submodules so importing this package
is cheap. Optional ML backends (YOLO) are lazy-imported only when selected.
"""

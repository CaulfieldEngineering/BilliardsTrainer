"""The measurement core (docs/MEASUREMENT_CORE.md).

M1: an offline engine that re-processes recorded sessions into DENSE
sidecars — ball positions every frame, a motion-model tracker that
coasts through blur, output byte-compatible with SidecarReader so the
entire downstream (trails, exports, phone) inherits density unchanged.
"""

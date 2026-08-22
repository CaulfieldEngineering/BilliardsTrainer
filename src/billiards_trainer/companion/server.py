"""Companion server: session review on Joe's phone.

Joe's brief, verbatim scope: "a way to be able to load a session and then
(if that session has shot detection markers), be able to navigate the clip
by shot (along with basic transport controls)." Review only, not live play.

Design for that scope and nothing more:
- Zero new dependencies: stdlib ThreadingHTTPServer.
- Serves the PWA (one static index.html — the exact file a Vercel deploy
  would host later, pointed at this same API).
- /api/sessions            -> recordings with duration + shot counts
                              (reuses the sidebar's summary cache)
- /api/session/<stem>/shots -> shot markers from the analysis sidecar
                              (light line scan; never loads track frames)
- /media/<file>            -> the mp4 with HTTP Range support — iOS Safari
                              refuses to seek (or play) without 206s.
- CORS wide open: the frontend may live on Vercel later while media stays
  on this box; same-origin today, cross-origin tomorrow, same code.

LAN only, no auth — the phone and the mini PC share Joe's network. The
first launch may need a Windows Firewall allow for python.
"""

from __future__ import annotations

import json
import re
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import Settings  # noqa: E402

_STATIC = Path(__file__).parent / "static"
_VIDEO_EXT = {".mp4", ".mov", ".m4v"}


def _recordings_dir() -> Path:
    return Path(Settings.load().recording.resolved_dir())


def read_shots(video: Path) -> list[dict]:
    """Shot markers via a light line scan — the sidecar's track frames can
    be tens of MB and are irrelevant here. Corrections still win: later
    correction lines override the shot's outcome, same rule as the app."""
    sc = Path(str(video) + ".analysis.jsonl")
    if not sc.is_file():
        return []
    shots: list[dict] = []
    try:
        with open(sc, encoding="utf-8") as f:
            for line in f:
                if line.startswith('{"type":"shot"') or line.startswith('{"type": "shot"'):
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    shots.append({"start": d.get("start", 0.0),
                                  "end": d.get("end", 0.0),
                                  "outcome": d.get("outcome", "miss"),
                                  "pocketed": d.get("pocketed", 0)})
                elif line.startswith('{"type":"correction"') \
                        or line.startswith('{"type": "correction"'):
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    for s in shots:
                        if abs(float(s["start"]) - float(d.get("start", -1))) < 0.2:
                            s["outcome"] = d.get("outcome", s["outcome"])
                            s["corrected"] = True
                            break
                elif line.startswith('{"type":"action"') \
                        or line.startswith('{"type": "action"'):
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    for s in shots:
                        if abs(float(s["start"]) - float(d.get("start", -1))) < 0.2:
                            s["action"] = d.get("action", "stroke")
                            break
    except OSError:
        return []
    return shots


def list_sessions() -> list[dict]:
    from billiards_trainer.vision import session_summaries as ss
    d = _recordings_dir()
    cache = ss.load_cache()
    out = []
    dirty_before = len(cache)
    try:
        vids = sorted((p for ext in _VIDEO_EXT for p in d.glob(f"*{ext}")
                       if not p.name.startswith(".")),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    for p in vids[:80]:
        try:
            s = ss.summarize(p, cache)
        except Exception:  # noqa: BLE001 - one unreadable file, keep listing
            s = {"dur_s": 0.0, "shots": None}
        out.append({
            "name": p.name,
            "mtime": int(p.stat().st_mtime),
            "size_mb": round(p.stat().st_size / 1e6, 1),
            "duration_s": s.get("dur_s", 0.0),
            "shots": s.get("shots"),   # null = no sidecar (still playable)
        })
    if len(cache) != dirty_before:
        ss.save_cache(cache)
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "BTCompanion/1"

    # -- plumbing ------------------------------------------------------- #
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        self.send_header("Access-Control-Expose-Headers",
                         "Content-Range, Accept-Ranges, Content-Length")

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quiet: only errors reach the console
        pass

    # -- routes --------------------------------------------------------- #
    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = unquote(self.path.split("?", 1)[0])
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html", "text/html; charset=utf-8")
            if path == "/api/sessions":
                return self._json(list_sessions())
            m = re.fullmatch(r"/api/session/([^/]+)/shots", path)
            if m:
                video = self._safe_media(m.group(1))
                if video is None:
                    return self._json({"error": "not found"}, 404)
                return self._json(read_shots(video))
            m = re.fullmatch(r"/media/([^/]+)", path)
            if m:
                return self._media(m.group(1))
            self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # iOS aborts range requests constantly while scrubbing; fine

    def _static(self, name: str, ctype: str) -> None:
        f = _STATIC / name
        body = f.read_bytes()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _safe_media(self, name: str) -> Path | None:
        """Filename-only lookup inside the recordings dir — no traversal."""
        if "/" in name or "\\" in name or name.startswith("."):
            return None
        p = _recordings_dir() / name
        return p if (p.is_file() and p.suffix.lower() in _VIDEO_EXT) else None

    def _media(self, name: str) -> None:
        p = self._safe_media(name)
        if p is None:
            return self._json({"error": "not found"}, 404)
        size = p.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:                       # suffix form: bytes=-N
                    start = max(0, size - int(m.group(2)))
            if start >= size:
                self.send_response(416)
                self._cors()
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        end = min(end, size - 1)
        length = end - start + 1
        self.send_response(206 if rng else 200)
        self._cors()
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def lan_addresses() -> list[str]:
    out = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       family=socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in out:
                out.append(ip)
    except OSError:
        pass
    return out


def serve(port: int = 8765) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    return httpd


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    httpd = serve(args.port)
    # phone review verdicts arrive via Dropbox sync; apply them here
    from .corrections_watcher import start_watcher
    start_watcher(_recordings_dir())
    print(f"Companion server on port {args.port} — open on your phone:")
    for ip in lan_addresses() or ["<this-pc-ip>"]:
        print(f"  http://{ip}:{args.port}/")
    print("(Safari -> Share -> Add to Home Screen for the app feel)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

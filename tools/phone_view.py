"""Screenshot the REAL phone player, so Claude can review its own work at
the level Joe reviews it.

WHY THIS EXISTS (Joe, 2026-08-30, after sending two screenshots of wrong
trails): "I need for you to find a way you can review yourself via your
VLM to this degree of detail." Every vision check in this campaign so far
has been on overlays THIS repo renders for itself - cv2 composites of
detections and tracks. None of them show what the phone actually draws,
which is why wild trails on the pinned session's first clip reached Joe
before they reached me.

This loads the real deployed app in QtWebEngine, authenticates it the way
the phone does, navigates to a session and shot, and screenshots the
player. The overlay code that runs is the SHIPPED code, and the trail
data is the SHIPPED data.

ONE HONEST LIMITATION, stated because a harness that lies is worse than
no harness: Qt's Chromium has no H.264
(canPlayType('video/mp4; codecs="avc1.42E01E"') == ""), so the <video>
never decodes. The frame is therefore extracted from the local mp4 with
OpenCV and placed under the overlay with the same object-fit: contain
letterbox the video element uses, and videoWidth/videoHeight are stubbed
to the file's real dimensions. Trail points are normalised 0-1 against
that box (see videoBox in app.js), so the OVERLAY GEOMETRY IS EXACT.
What is NOT exercised is decoding, buffering and the rVFC clock - so
this tool can prove a trail is drawn wrong, and cannot prove playback is
smooth.

    python tools/phone_view.py --session session-20260824-220247.mp4 \
        --shot 1 --out shot1.png
    python tools/phone_view.py --session ... --shot 5 --frac 0.5
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--disable-gpu --autoplay-policy=no-user-gesture-required")

ROOT = Path(__file__).resolve().parents[1]
REC = Path(r"C:/Users/Joe/Dropbox/Billiards/BilliardsTrainer")
SECRET = Path(r"C:/Users/Joe/.billiards-secrets/page_key.txt")
URL = "https://billiards-review.vercel.app"


def frame_data_uri(video: Path, t: float):
    """The real frame at absolute time t, plus the file's dimensions."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, fr = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ok:
        return None, w, h
    ok2, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok2:
        return None, w, h
    return ("data:image/jpeg;base64,"
            + base64.b64encode(buf.tobytes()).decode("ascii")), w, h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="session-*.mp4")
    ap.add_argument("--shot", type=int, default=1, help="1-based shot number")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="where in the shot window to sample (0=start, 1=end)")
    ap.add_argument("--t", type=float, default=None,
                    help="absolute video seconds; overrides --frac")
    ap.add_argument("--out", default="phone_view.png")
    ap.add_argument("--trails", default="1", help="1/0 - force trails on")
    a = ap.parse_args()

    video = REC / a.session
    if not video.is_file():
        print(f"no such recording: {video}")
        return 2
    key = SECRET.read_text(encoding="utf-8").strip()

    from PySide6.QtCore import QTimer, QUrl
    from PySide6.QtWidgets import QApplication
    from PySide6.QtWebEngineWidgets import QWebEngineView

    app = QApplication(sys.argv)
    view = QWebEngineView()
    view.resize(390, 844)
    view.show()
    st = {"phase": 0, "rc": 1}

    def js(code, cb=None):
        view.page().runJavaScript(code, cb or (lambda _r: None))

    def js_async(body, cb, tries=60):
        """runJavaScript cannot await a Promise - it marshals one as
        undefined, which cost the first run of this tool. Kick the async
        work off, park the result on window.__pv, and poll for it."""
        view.page().runJavaScript(
            "window.__pv = undefined; (async function(){ try {"
            + body + "} catch(e) { window.__pv = 'ERR ' + e; } })(); 1")

        def poll(n=0):
            def got(r):
                # An explicit sentinel, NOT null: this PySide6 build
                # marshals JS null to '' rather than None, so a "still
                # pending" poll looked like an empty answer and the tool
                # gave up on its own first call.
                if r and r != "__PENDING__":
                    cb(r)
                elif n < tries:
                    QTimer.singleShot(250, lambda: poll(n + 1))
                else:
                    fail("timed out waiting for the page")
            view.page().runJavaScript("(window.__pv === undefined) "
                                      "? '__PENDING__' : String(window.__pv)",
                                      got)
        QTimer.singleShot(250, poll)

    def fail(msg):
        print(msg)
        app.quit()

    def on_load(ok):
        if not ok:
            return fail("LOAD FAILED")
        st["phase"] += 1
        if st["phase"] == 1:          # authenticate the way the phone does
            js(f"localStorage.setItem('key', {key!r}); 1",
               lambda _r: view.load(QUrl(URL)))
        else:
            QTimer.singleShot(3500, open_session)

    def open_session():
        js_async(f"""
            const s = (sessCache || []).find(x => x.name === {a.session!r}
                                              || x.id === {a.session!r});
            if (!s) {{ window.__pv = 'NO SESSION; known: '
                + JSON.stringify((sessCache||[]).slice(0,3).map(z => z.name || z.id));
                return; }}
            await openSession(s);
            for (let i = 0; i < 60 && !shots.length; i++)
                await new Promise(r => setTimeout(r, 100));
            window.__pv = 'shots=' + shots.length;
          """, got_session)

    def got_session(res):
        print("session:", repr(res))
        if not str(res).startswith("shots="):
            return fail("could not open the session")
        n = int(str(res).split("=")[1])
        if not (1 <= a.shot <= n):
            return fail(f"shot {a.shot} out of range (1..{n})")
        js_async(f"""
            await gotoShot({a.shot - 1}, false);
            for (let i = 0; i < 40 && swapInFlight; i++)
                await new Promise(r => setTimeout(r, 50));
            const s = shots[cur] || {{}};
            window.__pv = JSON.stringify({{
              outcome: s.outcome, start: s.start, end: s.end,
              trails: (s.trails || []).map(t => ({{n: t.n, pts: (t.p||[]).length}})),
            }});
          """, got_shot)

    def got_shot(res):
        try:
            info = json.loads(res)
        except Exception:
            return fail(f"bad shot info: {res}")
        print("shot:", json.dumps(info))
        t0 = float(info.get("start") or 0.0)
        t1 = float(info.get("end") or (t0 + 1.0))
        t = a.t if a.t is not None else t0 + a.frac * max(0.0, t1 - t0)
        st["t"] = t
        uri, w, h = frame_data_uri(video, t)
        if uri is None:
            return fail(f"could not read a frame at {t:.2f}s")
        # Stub the decoder Qt does not have, place the REAL frame with the
        # same letterbox the <video> uses, and draw the shipped overlay at
        # the chosen time.
        js(f"""(function() {{
            const v = document.getElementById('video');
            // Stub the playback state Qt cannot provide. This is what lets
            // the app's OWN render loop draw - drawing once by hand was
            // wiped ~16ms later, because that loop clears the overlay
            // whenever readyState < 2 (the "no overlay while the picture
            // can't match it" rule). Stubbing is faithful here: the loop,
            // videoBox and drawTrails all run exactly as shipped.
            const defs = {{
              videoWidth: {w}, videoHeight: {h}, readyState: 4,
              seeking: false, paused: true, currentTime: {t!r},
            }};
            for (const k in defs) Object.defineProperty(v, k, {{
              get: () => defs[k], configurable: true }});
            window.scrubActive = false;
            let img = document.getElementById('__frame');
            if (!img) {{
              img = document.createElement('img');
              img.id = '__frame';
              img.style.cssText = 'position:absolute;inset:0;width:100%;'
                + 'height:100%;object-fit:contain;z-index:0;';
              v.parentElement.insertBefore(img, v);
            }}
            img.src = {uri!r};
            v.style.opacity = '0';
            if ({a.trails!r} === '1') {{
              window.TRAILS_ON = true;
              if (window.OV) OV.paths = true;
            }}
            return 'stubbed at t={t:.2f}, trails=' + !!window.TRAILS_ON;
          }})()""", drew)

    def drew(res):
        print("overlay:", res)
        QTimer.singleShot(900, shoot)

    def shoot():
        out = Path(a.out)
        if not out.is_absolute():
            out = ROOT / "_train" / "bench_fix" / "asjoesees" / out
        out.parent.mkdir(parents=True, exist_ok=True)
        view.grab().save(str(out))
        print("wrote", out)
        st["rc"] = 0
        app.quit()

    view.loadFinished.connect(on_load)
    view.load(QUrl(URL))
    QTimer.singleShot(60000, app.quit)
    app.exec()
    return st["rc"]


if __name__ == "__main__":
    raise SystemExit(main())

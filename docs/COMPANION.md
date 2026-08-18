# Phone companion — session review on iOS

Review-only mobile app (Joe's brief): open a session, and when it has
shot markers, navigate the clip shot-by-shot with transport controls.

## Use it now (LAN)
The mini PC serves everything — no App Store, no build step:

1. `python -m billiards_trainer.companion` (the loop keeps it running)
2. On the iPhone (same Wi-Fi): open `http://<pc-ip>:8765/`
   (the server prints the exact URL at startup)
3. Safari -> Share -> **Add to Home Screen** -> it opens full-screen
   like a native app.

Sessions list shows duration + shot counts (same cache as the desktop
sidebar). Tapping a session loads the video (native H.264 streaming with
range requests, so scrubbing works) plus its shot chips: colour = the
desktop timeline's outcome colours, tap = jump to that shot's pre-roll.
Prev / Replay / Next walk the shots; corrections made on the desktop show
as ✓ (the server re-applies sidecar correction records).

## Vercel later
`static/index.html` IS the deployable frontend — one file, no build.
When Joe sets up Vercel credentials: deploy that file as-is and set the
`API` constant at the top of the script to the PC/tunnel URL. The server
already sends permissive CORS + exposes range headers, so the hosted
frontend streams from the PC unchanged. (Media itself never goes to
Vercel; recordings stay on the rig.)

## Notes
- Windows Firewall: setup added an inbound allow rule for TCP 8765 on
  private networks (`BilliardsTrainer Companion`). Remove with
  `netsh advfirewall firewall delete rule name="BilliardsTrainer Companion"`.
- No auth by design while LAN-only. Revisit before any tunnel/Vercel use.

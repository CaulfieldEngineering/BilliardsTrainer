# Billiards Review — cloud companion

Phone review app at https://billiards-review.vercel.app — streams
sessions straight from Dropbox (works with the mini PC off).

- `api/` — Vercel serverless proxy; holds the READ-ONLY Dropbox refresh
  token in env vars, gates every call on `PAGE_KEY` (x-key header; the
  page learns it once from the install link's `?k=` and keeps it in
  localStorage).
- `public/` — single-file PWA. `__BUILD_ID__` is stamped at deploy;
  the page polls `/version.json` and offers one-tap reload when a newer
  build ships (Joe's self-update ask).
- `deploy.py` — stamp + `vercel deploy --prod` using the token in
  `C:/Users/Joe/.billiards-secrets/`. Data source: `<video>.shots.json`
  summaries written next to each recording by the session-close pass.

Secrets live in `C:/Users/Joe/.billiards-secrets/` (never in the repo):
vercel_token.txt, dropbox_app.txt, dropbox_refresh_token.txt,
page_key.txt.

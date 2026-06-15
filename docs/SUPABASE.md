# Cloud backup & feedback sync (Supabase) — setup

The app works fully **without** this. Feedback and stats are always saved locally
in SQLite first. Setting up Supabase just adds an off-machine backup of your
feedback, sessions, and shots (read-only/backup for now — no cross-device edits).

Until you complete the steps below, cloud sync is a no-op and the Settings →
Feedback card shows **"Cloud sync: not configured"**.

## What you need to do (one-time, ~5 min)

1. **Create a free Supabase project** at <https://supabase.com> (New project →
   pick a name like `billiards-trainer` → choose a region → create).
2. **Run the schema.** In the project, open **SQL Editor → New query**, paste the
   contents of [`docs/supabase_schema.sql`](supabase_schema.sql), and Run.
3. **Get your credentials.** Project **Settings → API**:
   - `Project URL` (e.g. `https://abcd1234.supabase.co`)
   - `service_role` key (under *Project API keys* — **secret**, keep it private).
4. **Give them to the app.** Create a file named `supabase.json` in the app data
   folder and paste:
   ```json
   {
     "url": "https://YOUR-PROJECT.supabase.co",
     "service_role_key": "YOUR-SERVICE-ROLE-KEY"
   }
   ```
   The app data folder is:
   `%LOCALAPPDATA%\BilliardsTrainer\` (Windows) — i.e.
   `C:\Users\<you>\AppData\Local\BilliardsTrainer\supabase.json`.

   *(Alternatively set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` as
   environment variables — those win over the file.)*
5. **Restart the app.** Settings → Feedback now shows **"Cloud sync: configured"**.

## What syncs, and when

- **What:** the `feedback`, `sessions`, and `shots` tables (rows are pushed by
  primary-key upsert, so re-runs are safe and idempotent).
- **When:** ~8 s after launch, every 5 minutes if there are new rows, after you
  submit feedback, and on app close. Anything queued while offline/unconfigured
  syncs once credentials are present.

## Security note

The `service_role` key bypasses Row-Level Security, so the schema enables RLS
with no policies — nothing but your key can touch these tables. Keep the key out
of screenshots and public repos. (A production multi-user design would use a
narrowly-scoped key or an Edge Function instead; this is fine for a single-user
backup.)

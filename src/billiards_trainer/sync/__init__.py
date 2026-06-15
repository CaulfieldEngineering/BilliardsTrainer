"""Optional Supabase backup/sync. Local SQLite stays the source of truth; this
pushes feedback + session/shot rows to Supabase when credentials are configured.
Entirely a no-op until ``supabase.json`` (or env vars) provide creds."""

from .supabase import SupabaseClient, SyncManager, sync_status

__all__ = ["SupabaseClient", "SyncManager", "sync_status"]

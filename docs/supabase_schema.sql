-- Billiards Trainer — Supabase schema
-- Run this once in your Supabase project's SQL editor.
-- The app syncs with the service-role key, which bypasses RLS, so RLS can stay
-- enabled with no policies (nothing else can read/write these tables).

create table if not exists public.feedback (
    id               bigint primary key,
    created_at       timestamptz,
    kind             text,            -- bug | feature | devnote
    title            text,
    description      text,
    severity         text,            -- low | medium | high | ''
    app_version      text,
    system_info      text,
    attachment_count int default 0,
    received_at      timestamptz default now()
);

create table if not exists public.sessions (
    id          bigint primary key,
    started_at  timestamptz,
    ended_at    timestamptz,
    mode        text,
    drill_key   text,
    table_size  text,
    received_at timestamptz default now()
);

create table if not exists public.shots (
    id            bigint primary key,
    session_id    bigint,
    created_at    timestamptz,
    outcome       text,               -- make | miss | scratch
    num_pocketed  int,
    target_pocket text,
    cue_scratch   boolean,
    duration_s    double precision,
    shot_seconds  double precision,
    streak_index  int,
    received_at   timestamptz default now()
);

alter table public.feedback enable row level security;
alter table public.sessions enable row level security;
alter table public.shots    enable row level security;

-- Handy views
create or replace view public.bug_reports as
    select * from public.feedback where kind = 'bug' order by created_at desc;
create or replace view public.feature_requests as
    select * from public.feedback where kind = 'feature' order by created_at desc;

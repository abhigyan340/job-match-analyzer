-- Run this in the Supabase SQL editor to create the analyses table.
-- Optional: only needed once you wire up history/auth.

create table if not exists analyses (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id),
    resume_filename text,
    job_description text,
    match_score numeric,
    tfidf_score numeric,
    embedding_score numeric,
    matched_skills text[],
    missing_skills text[],
    feedback text,
    created_at timestamptz default now()
);

-- Row-level security: users can only see their own analyses
alter table analyses enable row level security;

create policy "Users can view their own analyses"
    on analyses for select
    using (auth.uid() = user_id);

create policy "Users can insert their own analyses"
    on analyses for insert
    with check (auth.uid() = user_id);

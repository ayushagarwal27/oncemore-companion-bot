-- Companion core: memory schema. Idempotent, safe to re-run.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- Identity and conversation log
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    persona_id  TEXT
);

ALTER TABLE users ADD COLUMN IF NOT EXISTS persona_id TEXT;

CREATE TABLE IF NOT EXISTS sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    thread_id   TEXT NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at    TIMESTAMPTZ,
    turn_count  INTEGER NOT NULL DEFAULT 0,
    summary     TEXT
);

CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions (user_id, started_at DESC);

-- Raw turn log. Every memory points back here for provenance.
CREATE TABLE IF NOT EXISTS messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL,
    turn_index      INTEGER NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('user', 'companion')),
    content         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    trace_id        TEXT,
    feedback        SMALLINT CHECK (feedback IN (-1, 0, 1)),
    feedback_reason TEXT
);

CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (session_id, turn_index);
CREATE INDEX IF NOT EXISTS messages_feedback_idx ON messages (user_id, feedback)
    WHERE feedback IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Semantic memory, shape 1: profile
--
-- The ~15 attributes most likely to be contradicted later. Keeping them here
-- means "I broke up with my ex" costs one UPDATE instead of an embedding
-- search plus an adjudicator call.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS user_profile (
    user_id             TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    preferred_name      TEXT,
    pronouns            TEXT,
    age_range           TEXT,
    location            TEXT,
    occupation          TEXT,
    employer            TEXT,
    relationship_status TEXT,
    partner_name        TEXT,
    living_situation    TEXT,
    key_people          JSONB NOT NULL DEFAULT '[]'::jsonb,
    current_focus       TEXT,
    ongoing_stressor    TEXT,
    communication_style TEXT,
    topics_to_avoid     JSONB NOT NULL DEFAULT '[]'::jsonb,
    version             INTEGER NOT NULL DEFAULT 1,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Field-level audit trail: what makes profile supersession demonstrable.
CREATE TABLE IF NOT EXISTS user_profile_history (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           TEXT NOT NULL,
    field             TEXT NOT NULL,
    old_value         TEXT,
    new_value         TEXT,
    changed_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
    reason            TEXT
);

CREATE INDEX IF NOT EXISTS profile_history_user_idx
    ON user_profile_history (user_id, changed_at DESC);

-- ---------------------------------------------------------------------------
-- Semantic memory, shape 2: bi-temporal fact ledger
--
--   valid_from / valid_to    when the fact was true in the world
--   created_at / expired_at  when the system believed it
--
-- Nothing is deleted. Contradiction closes valid_to and sets superseded_by.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory_facts (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    subject           TEXT NOT NULL DEFAULT 'user',
    predicate         TEXT NOT NULL,
    object            TEXT NOT NULL,
    text              TEXT NOT NULL,
    fact_type         TEXT NOT NULL CHECK (fact_type IN (
                          'identity', 'relationship', 'preference',
                          'event', 'plan', 'opinion', 'mood')),

    embedding         vector(1536),

    status            TEXT NOT NULL DEFAULT 'active' CHECK (status IN (
                          'active', 'superseded', 'retracted', 'duplicate')),

    valid_from        TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to          TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    expired_at        TIMESTAMPTZ,
    superseded_by     UUID REFERENCES memory_facts(id) ON DELETE SET NULL,

    confidence        REAL NOT NULL DEFAULT 0.8,
    importance        REAL NOT NULL DEFAULT 0.5,
    access_count      INTEGER NOT NULL DEFAULT 0,
    last_accessed_at  TIMESTAMPTZ,

    source_message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
    source_session_id UUID REFERENCES sessions(id) ON DELETE SET NULL
);

-- The (subject, predicate) pair is the "slot" the adjudicator reasons over.
CREATE INDEX IF NOT EXISTS facts_slot_idx
    ON memory_facts (user_id, subject, predicate)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS facts_active_idx
    ON memory_facts (user_id, fact_type)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS facts_embedding_idx
    ON memory_facts USING hnsw (embedding vector_cosine_ops)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS facts_text_trgm_idx
    ON memory_facts USING gin (text gin_trgm_ops);

-- ---------------------------------------------------------------------------
-- Persona self-memory
--
-- LangMem's taxonomy covers the user (semantic, episodic) and the agent's
-- skills (procedural), but not what the companion has asserted about itself.
-- Same bi-temporal shape, global namespace rather than user-scoped.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS persona_commitments (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    persona_id        TEXT NOT NULL DEFAULT 'default',
    topic             TEXT NOT NULL,
    text              TEXT NOT NULL,
    embedding         vector(1536),
    status            TEXT NOT NULL DEFAULT 'active' CHECK (status IN (
                          'active', 'superseded', 'retracted')),
    valid_from        TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to          TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    superseded_by     UUID REFERENCES persona_commitments(id) ON DELETE SET NULL,
    confidence        REAL NOT NULL DEFAULT 0.9,
    source_message_id UUID REFERENCES messages(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS commitments_topic_idx
    ON persona_commitments (persona_id, topic)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS commitments_embedding_idx
    ON persona_commitments USING hnsw (embedding vector_cosine_ops)
    WHERE status = 'active';

-- ---------------------------------------------------------------------------
-- Episodic memory
--   kind='relational' -> "the night she talked about her dad"
--   kind='voice'      -> pinned few-shot anchors that resist tone flattening
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS episodes (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind              TEXT NOT NULL CHECK (kind IN ('relational', 'voice')),

    title             TEXT NOT NULL,
    observation       TEXT NOT NULL,
    companion_action  TEXT,
    outcome           TEXT,
    text              TEXT NOT NULL,
    embedding         vector(1536),

    salience          REAL NOT NULL DEFAULT 0.5,
    pinned            BOOLEAN NOT NULL DEFAULT FALSE,
    access_count      INTEGER NOT NULL DEFAULT 0,

    occurred_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    session_id        UUID REFERENCES sessions(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS episodes_user_idx ON episodes (user_id, kind, salience DESC);
CREATE INDEX IF NOT EXISTS episodes_embedding_idx
    ON episodes USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- Procedural memory: versioned prompt zones
--
-- 'frozen' is never auto-written; it holds the persona canon.
-- 'adaptive' holds optimizer output, which must pass the eval gate before
-- promoted flips to true. An optimizer that rewrites persona traits from
-- thumbs-down signals is a drift generator, so the zones stay separate.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS prompt_versions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    persona_id  TEXT NOT NULL DEFAULT 'default',
    zone        TEXT NOT NULL CHECK (zone IN ('frozen', 'adaptive')),
    content     TEXT NOT NULL,
    parent_id   UUID REFERENCES prompt_versions(id) ON DELETE SET NULL,
    promoted    BOOLEAN NOT NULL DEFAULT FALSE,
    eval_scores JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS prompt_active_idx
    ON prompt_versions (persona_id, zone, created_at DESC)
    WHERE promoted;

-- ---------------------------------------------------------------------------
-- Eval harness storage
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS eval_runs (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario   TEXT NOT NULL,
    config     JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at   TIMESTAMPTZ,
    summary    JSONB
);

CREATE TABLE IF NOT EXISTS eval_results (
    id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id   UUID NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    probe_id TEXT NOT NULL,
    metric   TEXT NOT NULL,
    passed   BOOLEAN,
    score    REAL,
    detail   JSONB
);

CREATE INDEX IF NOT EXISTS eval_results_run_idx ON eval_results (run_id, metric);

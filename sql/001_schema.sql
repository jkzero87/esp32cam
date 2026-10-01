-- Phase 4: people the camera may recognize, and short durable facts about them.
-- Face embeddings are not stored here: they stay in data/gallery/<name>.npy,
-- linked by people.name.
CREATE SCHEMA IF NOT EXISTS cam;

CREATE TABLE IF NOT EXISTS cam.people (
    id         bigserial PRIMARY KEY,
    name       text NOT NULL UNIQUE CHECK (name ~ '^[a-z0-9_-]+$'),  -- = gallery file name
    created_at timestamptz NOT NULL DEFAULT now(),
    consent_at timestamptz            -- when they agreed to be remembered; NULL = enrolled by hand (enroll.py)
);

CREATE TABLE IF NOT EXISTS cam.facts (
    id                     bigserial PRIMARY KEY,
    person_id              bigint NOT NULL REFERENCES cam.people(id) ON DELETE CASCADE,
    fact                   text NOT NULL CHECK (length(fact) BETWEEN 1 AND 200),
    created_at             timestamptz NOT NULL DEFAULT now(),
    source_conversation_at timestamptz NOT NULL  -- start of the conversation it came from
);

CREATE INDEX IF NOT EXISTS facts_person_created ON cam.facts (person_id, created_at DESC);

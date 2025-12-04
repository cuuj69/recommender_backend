-- Enable pgvector and create core tables.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name TEXT,
    kyc_preferences JSONB,
    kyc_embedding vector(768),
    cf_vector vector(128),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS books (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT,
    description TEXT,
    genres TEXT[],
    content_embedding vector(768),
    cf_embedding vector(128),
    gnn_vector vector(256),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS interactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    book_id INTEGER REFERENCES books(id),
    interaction_type TEXT NOT NULL,
    rating NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_kyc_embedding ON users USING ivfflat (kyc_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_books_content_embedding ON books USING ivfflat (content_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_books_cf_embedding ON books USING ivfflat (cf_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_books_gnn_vector ON books USING ivfflat (gnn_vector vector_cosine_ops);

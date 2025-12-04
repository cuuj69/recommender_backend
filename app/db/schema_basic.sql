-- Schema without pgvector - using JSONB arrays for vectors
-- Vector operations will be done in Python

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    first_name TEXT,
    last_name TEXT,
    is_admin BOOLEAN DEFAULT FALSE,
    kyc_preferences JSONB,
    kyc_embedding JSONB,  -- Array of floats [0.1, 0.2, ...] for 768 dimensions
    cf_vector JSONB,      -- Array of floats for 128 dimensions
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS books (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT,
    description TEXT,
    genres TEXT[],
    content_embedding JSONB,  -- Array of floats for 768 dimensions
    cf_embedding JSONB,        -- Array of floats for 128 dimensions
    gnn_vector JSONB,          -- Array of floats for 256 dimensions
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS interactions (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    book_id INTEGER REFERENCES books(id),
    interaction_type TEXT NOT NULL,
    rating NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes (will be updated after pgvector is enabled)
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_books_title ON books(title);
CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_book_id ON interactions(book_id);


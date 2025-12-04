-- Migration script to convert TEXT columns to vector types
-- Run this AFTER enabling pgvector extension in Azure Portal and restarting the server

-- First, enable the extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Convert users table
ALTER TABLE users 
  ALTER COLUMN kyc_embedding TYPE vector(768) USING kyc_embedding::vector,
  ALTER COLUMN cf_vector TYPE vector(128) USING cf_vector::vector;

-- Convert books table
ALTER TABLE books 
  ALTER COLUMN content_embedding TYPE vector(768) USING content_embedding::vector,
  ALTER COLUMN cf_embedding TYPE vector(128) USING cf_embedding::vector,
  ALTER COLUMN gnn_vector TYPE vector(256) USING gnn_vector::vector;

-- Create vector indexes
CREATE INDEX IF NOT EXISTS idx_users_kyc_embedding ON users USING ivfflat (kyc_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_books_content_embedding ON books USING ivfflat (content_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_books_cf_embedding ON books USING ivfflat (cf_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_books_gnn_vector ON books USING ivfflat (gnn_vector vector_cosine_ops);


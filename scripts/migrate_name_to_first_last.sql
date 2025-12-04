-- Migration: Convert 'name' column to 'first_name' and 'last_name'
-- Run this if you have existing data with the 'name' column

-- Add new columns
ALTER TABLE users 
  ADD COLUMN IF NOT EXISTS first_name TEXT,
  ADD COLUMN IF NOT EXISTS last_name TEXT;

-- Migrate existing name data (split on first space)
-- This assumes names are in "First Last" format
UPDATE users 
SET 
  first_name = SPLIT_PART(name, ' ', 1),
  last_name = CASE 
    WHEN POSITION(' ' IN name) > 0 THEN SPLIT_PART(name, ' ', 2)
    ELSE NULL
  END
WHERE name IS NOT NULL AND first_name IS NULL;

-- Drop old name column (uncomment when ready)
-- ALTER TABLE users DROP COLUMN IF EXISTS name;


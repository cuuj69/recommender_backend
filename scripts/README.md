# Database Migration Scripts

This directory contains SQL migration scripts for database schema changes.

## Migration Scripts

### `migrate_name_to_first_last.sql`
**Status:**  Applied (if you migrated from old schema)  
**Purpose:** Converts the `name` column to `first_name` and `last_name` columns in the `users` table.

**When to run:**
- Only if you have existing data with the old `name` column
- For fresh installations, use `schema_basic.sql` which already has `first_name`/`last_name`

**Usage:**
```bash
psql -h <host> -U <user> -d <database> -f scripts/migrate_name_to_first_last.sql
```

### `migrate_to_pgvector.sql`
**Status:**  Not applied (requires pgvector extension)  
**Purpose:** Converts JSONB vector columns to native `vector` types for better performance with pgvector.

**Prerequisites:**
1. Enable `pgvector` extension in Azure PostgreSQL (or your PostgreSQL instance)
2. Restart the PostgreSQL server
3. Then run this migration

**When to run:**
- After enabling pgvector extension
- When you want to use native vector operations instead of JSONB arrays

**Usage:**
```bash
psql -h <host> -U <user> -d <database> -f scripts/migrate_to_pgvector.sql
```

**Note:** Currently using `schema_basic.sql` which stores vectors as JSONB. This migration converts them to native vector types.

## Setup Scripts

### `enable_pgvector.sh`
Script to enable pgvector extension on a standard PostgreSQL instance.

### `enable_pgvector_azure.sh`
Script to enable pgvector extension on Azure PostgreSQL (requires Azure CLI).

## Current Schema

The current production schema uses:
- `schema_basic.sql` - JSONB arrays for vectors (no pgvector required)
- UUID for user IDs
- `first_name` and `last_name` for users

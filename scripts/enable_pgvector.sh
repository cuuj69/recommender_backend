#!/bin/bash
# Enable pgvector extension in Azure PostgreSQL via Azure CLI

# Replace these with your values
RESOURCE_GROUP="your-resource-group"
SERVER_NAME="books"  # Your server name (from books.postgres.database.azure.com)

echo "Enabling pgvector extension in Azure PostgreSQL..."

# Enable the extension
az postgres flexible-server parameter set \
  --resource-group "$RESOURCE_GROUP" \
  --server-name "$SERVER_NAME" \
  --name "shared_preload_libraries" \
  --value "vector" \
  --output table

echo "✓ pgvector extension enabled!"
echo "Note: You may need to restart the server for changes to take effect."


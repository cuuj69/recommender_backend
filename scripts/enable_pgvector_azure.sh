#!/bin/bash
# Enable pgvector in Azure PostgreSQL via Azure CLI

# You need to replace these with your actual values
RESOURCE_GROUP="your-resource-group-name"
SERVER_NAME="books"  # Your server name (without .postgres.database.azure.com)

echo "Enabling pgvector extension..."

# Get current value
CURRENT_VALUE=$(az postgres server configuration show \
  --resource-group "$RESOURCE_GROUP" \
  --server-name "$SERVER_NAME" \
  --name "shared_preload_libraries" \
  --query "value" -o tsv)

echo "Current value: $CURRENT_VALUE"

# Add vector if not already present
if [[ "$CURRENT_VALUE" == *"vector"* ]]; then
    echo "✓ vector is already in shared_preload_libraries"
else
    NEW_VALUE="${CURRENT_VALUE},vector"
    echo "New value: $NEW_VALUE"
    
    az postgres server configuration set \
      --resource-group "$RESOURCE_GROUP" \
      --server-name "$SERVER_NAME" \
      --name "shared_preload_libraries" \
      --value "$NEW_VALUE"
    
    echo "✓ Updated! Now restart your server in Azure Portal"
fi


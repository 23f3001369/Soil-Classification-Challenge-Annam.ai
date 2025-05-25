#!/bin/bash

# Google Drive folder ID
FOLDER_ID="130g3C17xyvlLhjTd5lSX4bDkL5f-3nDR"
TARGET_DIR="./data"

echo "📁 Downloading folder from Google Drive (ID: $FOLDER_ID)"
mkdir -p "$TARGET_DIR"

# Check if gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "❌ 'gdown' not found. Please install it using: pip install gdown"
    exit 1
fi

# Download the folder
gdown --folder "$FOLDER_ID" -O "$TARGET_DIR"

echo "✅ Download complete. Files saved to: $TARGET_DIR"

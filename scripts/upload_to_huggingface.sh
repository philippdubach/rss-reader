#!/bin/bash
# Upload HN Success Predictor model to HuggingFace Hub
# Usage: ./upload_to_huggingface.sh

set -e

MODEL_DIR="rss_reader/models/hn_model_v7"
REPO_NAME="philippdubach/hn-success-predictor"

echo "=== HuggingFace Model Upload ==="
echo "Model: $MODEL_DIR"
echo "Repo: $REPO_NAME"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Login check
echo "Checking HuggingFace login..."
huggingface-cli whoami || {
    echo ""
    echo "Please login to HuggingFace:"
    huggingface-cli login
}

# Create repo if it doesn't exist
echo ""
echo "Creating/checking repository..."
huggingface-cli repo create "$REPO_NAME" --type model 2>/dev/null || echo "Repo already exists"

# Upload files
echo ""
echo "Uploading model files..."
huggingface-cli upload "$REPO_NAME" "$MODEL_DIR" . \
    --repo-type model \
    --commit-message "Upload HN Success Predictor V7"

echo ""
echo "=== Upload Complete ==="
echo "Model available at: https://huggingface.co/$REPO_NAME"
echo ""
echo "To download in your project:"
echo "  from huggingface_hub import snapshot_download"
echo "  snapshot_download('$REPO_NAME', local_dir='rss_reader/models/hn_model_v7')"

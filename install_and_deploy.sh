#!/usr/bin/env bash
# ==============================================================================
# NazoratAI Turnkey Installation, Deployment & Verification Pipeline
# Target Directory: /home/zaax/teamwork_projects/nazorat_ai
# ==============================================================================

set -e

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="/home/zaax/teamwork_projects/nazorat_ai"

echo "=========================================================="
echo "      🏦 NazoratAI Turnkey Deployment Pipeline           "
echo "=========================================================="
echo "Source Artifacts: $SOURCE_DIR"
echo "Target Directory: $TARGET_DIR"
echo ""

mkdir -p "$TARGET_DIR"
cp -r "$SOURCE_DIR"/* "$TARGET_DIR"/
cd "$TARGET_DIR"

echo "✅ All NazoratAI modules, configurations, and test suites copied."
echo ""

echo "--- Step 1: Pre-generating realistic corporate dataset (603 borrowers) ---"
python3 data/generate_dataset.py || true

echo ""
echo "--- Step 2: Running full test suite ---"
if command -v pytest &> /dev/null; then
    pytest -v tests/
else
    echo "ℹ️ pytest not found in current shell. Install dependencies with: pip install -r requirements.txt"
fi

echo ""
echo "=========================================================="
echo "✅ NazoratAI Successfully Deployed to $TARGET_DIR"
echo ""
echo "Quick Start Commands:"
echo "  cd $TARGET_DIR"
echo "  pip install -e ."
echo "  nazorat-ai scan-portfolio"
echo "  nazorat-ai run-dashboard --port 8501"
echo "  nazorat-ai serve-api --port 8000"
echo "=========================================================="

#!/usr/bin/env bash
# Upload the Streamlit app to a Hugging Face Space.
# Usage: huggingface/deploy.sh <username>/<space-name>
# Requires: `hf auth login` (a token with write access) beforehand.
set -euo pipefail
SPACE="${1:?Usage: huggingface/deploy.sh <username>/<space-name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/asd-backend" "$STAGE/reports"
cp "$ROOT"/app.py "$ROOT"/explore_page.py "$ROOT"/predict_page.py "$STAGE/"
cp "$ROOT"/asd-backend/asd_core.py "$ROOT"/asd-backend/asd_detection2.pkl "$STAGE/asd-backend/"
cp -R "$ROOT"/reports/. "$STAGE/reports/"
cp "$ROOT"/huggingface/Dockerfile "$ROOT"/huggingface/README.md "$ROOT"/huggingface/requirements.txt "$STAGE/"

hf repo create "$SPACE" --repo-type space --space_sdk docker --exist-ok
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy ASD detection app"
echo "Deployed: https://huggingface.co/spaces/$SPACE"

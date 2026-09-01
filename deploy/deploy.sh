#!/bin/sh
# Deploy the showcase studio to Cloud Run.
#
#   ./deploy/deploy.sh <gcp-project> [region]
#
# Assumes deploy/ctx was staged: the locaish package, the approved rooms
# from twins/showcase, and the ClickHouse dumps. GOOGLE_API_KEY is read
# from .env and set on the service; nothing is printed.
set -e
PROJECT="${1:?usage: deploy.sh <gcp-project> [region]}"
REGION="${2:-us-central1}"
KEY=$(grep '^GOOGLE_API_KEY=' .env | cut -d= -f2-)
[ -n "$KEY" ] || { echo "GOOGLE_API_KEY missing from .env"; exit 1; }
gcloud run deploy locaish \
  --source deploy/ctx \
  --project "$PROJECT" --region "$REGION" \
  --allow-unauthenticated \
  --memory 8Gi --cpu 4 --timeout 3600 --concurrency 20 --max-instances 1 \
  --set-env-vars "GOOGLE_API_KEY=$KEY,LOCAISH_SHOWCASE=1"

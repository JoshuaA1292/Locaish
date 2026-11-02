#!/bin/sh
# Deploy the showcase studio to Cloud Run.
#
#   ./deploy/deploy.sh <gcp-project> [region]
#
# Assumes deploy/ctx was staged: the locaish package, the approved rooms
# (deploy/ctx/rooms/<job>), and the ClickHouse dumps. The Meta Model API key
# and the SAM 3D endpoint are read from .env at deploy time and set on the
# service as environment variables, so judges can rehearse, direct, finish
# and lift in the hosted dorm; remove the three lines to make it read-only.
set -e
PROJECT="${1:?usage: deploy.sh <gcp-project> [region]}"
REGION="${2:-us-central1}"
ENVFILE="$(dirname "$0")/../.env"
val() { grep -E "^$1=" "$ENVFILE" | head -1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//'; }
META_API_KEY="$(val META_API_KEY)"
SAM3D_URL="$(val LOCAISH_SAM3D_URL)"
SAM3D_TOKEN="$(val LOCAISH_SAM3D_TOKEN)"
[ -n "$META_API_KEY" ] || { echo "META_API_KEY missing from .env"; exit 1; }
printf 'META_API_KEY=%s\nLOCAISH_SAM3D_URL=%s\nLOCAISH_SAM3D_TOKEN=%s\nLOCAISH_SHOWCASE=1\nGOOGLE_GENAI_USE_VERTEXAI=TRUE\nGOOGLE_CLOUD_PROJECT=%s\nGOOGLE_CLOUD_LOCATION=global\n' \
  "$META_API_KEY" "$SAM3D_URL" "$SAM3D_TOKEN" "$PROJECT" > "$(dirname "$0")/.env.cloudrun.yaml.tmp"
# gcloud wants YAML for --env-vars-file
sed -e 's/^\([A-Z_0-9]*\)=\(.*\)$/\1: "\2"/' "$(dirname "$0")/.env.cloudrun.yaml.tmp" > "$(dirname "$0")/.env.cloudrun.yaml"
rm -f "$(dirname "$0")/.env.cloudrun.yaml.tmp"
trap 'rm -f "$(dirname "$0")/.env.cloudrun.yaml"' EXIT
gcloud run deploy locaish \
  --source deploy/ctx \
  --project "$PROJECT" --region "$REGION" \
  --allow-unauthenticated \
  --memory 8Gi --cpu 4 --no-cpu-throttling --timeout 3600 --concurrency 20 --max-instances 1 \
  --env-vars-file "$(dirname "$0")/.env.cloudrun.yaml"

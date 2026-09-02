#!/bin/sh
# Deploy the showcase studio to Cloud Run.
#
#   ./deploy/deploy.sh <gcp-project> [region]
#
# Assumes deploy/ctx was staged: the locaish package, the approved rooms
# from twins/showcase, and the ClickHouse dumps. Gemini runs through Vertex
# AI using the service's own identity; grant the Cloud Run service account
# roles/aiplatform.user once and no API key ever ships.
set -e
PROJECT="${1:?usage: deploy.sh <gcp-project> [region]}"
REGION="${2:-us-central1}"
gcloud run deploy locaish \
  --source deploy/ctx \
  --project "$PROJECT" --region "$REGION" \
  --allow-unauthenticated \
  --memory 8Gi --cpu 4 --no-cpu-throttling --timeout 3600 --concurrency 20 --max-instances 1 \
  --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=$PROJECT,GOOGLE_CLOUD_LOCATION=global,LOCAISH_SHOWCASE=1"

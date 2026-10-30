#!/bin/sh
# Deploy the live demo to Google Cloud Run. Run it in Cloud Shell, in a project with billing:
#   git clone https://github.com/moziyadh110/creditshield-copilot && sh creditshield-copilot/demo/deploy.sh
# Safe to re-run: it rebuilds from GitHub main and keeps the stored key.
set -e
cd "$(dirname "$0")"
REGION=us-central1  # a US region, so the always-free Cloud Run allowance applies
PROJECT=$(gcloud config get-value project 2> /dev/null)

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com secretmanager.googleapis.com

if ! gcloud secrets describe deepseek-api-key > /dev/null 2>&1; then
    printf "Paste your DeepSeek API key (it won't show), then press Enter: "
    stty -echo; read -r KEY; stty echo; echo
    printf %s "$KEY" | gcloud secrets create deepseek-api-key --data-file=-
fi
NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding deepseek-api-key --quiet > /dev/null \
    --member="serviceAccount:$NUMBER-compute@developer.gserviceaccount.com" \
    --role=roles/secretmanager.secretAccessor

gcloud config set builds/timeout 1800  # the image is ~3.5 GB (CPU PyTorch); allow 30 minutes

# One instance at most (the rate limiter and the database live in it), none while idle.
gcloud run deploy creditshield-demo --source . --region "$REGION" --quiet \
    --allow-unauthenticated --port 7860 --memory 2Gi --cpu 1 --cpu-boost \
    --min-instances 0 --max-instances 1 --timeout 120 \
    --set-secrets DEEPSEEK_API_KEY=deepseek-api-key:latest

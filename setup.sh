#!/usr/bin/env bash
# ==============================================================================
# Setup Google Cloud Infrastructure for Video Trimmer
#
# 100% Native gcloud Provisioning (Zero Terraform Dependency, Cloud Shell Ready)
#
# Usage:
#   ./setup.sh [OPTIONS]
#
# Options:
#   -p, --project PROJECT_ID     Google Cloud Project ID (overrides .env)
#   -r, --region REGION          Google Cloud Region (default: us-central1)
#   -b, --bucket BUCKET_NAME     Custom GCS bucket name (default: video-preprocessing-${PROJECT_ID})
#   -s, --service-account SA     Custom service account email (default: video-trimmer-sa@...)
#   -n, --dry-run                Preview provisioning commands without executing
#   -h, --help                   Show this help message and exit
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# 1. Load Environment Configuration
# ------------------------------------------------------------------------------
if [ -f "$REPO_ROOT/.env" ]; then
    echo "[*] Loading environment variables from: $REPO_ROOT/.env"
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

PROJECT_ID="${GCP_PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
PROJECT_NUMBER="${GCP_PROJECT_NUMBER:-${PROJECT_NUMBER:-}}"
REGION="${GCP_REGION:-us-central1}"
BUCKET_NAME="${VIDEO_TRIMMER_BUCKET:-}"
SERVICE_ACCOUNT="${GCP_SERVICE_ACCOUNT:-${SERVICE_ACCOUNT:-}}"
DRY_RUN=false

usage() {
    cat <<EOF
Usage: ./setup.sh [OPTIONS]

Setup Google Cloud Infrastructure for Video Trimmer.
(100% native gcloud provisioning - zero Terraform dependency, Cloud Shell ready)

Environment Variables (.env or shell):
  GOOGLE_CLOUD_PROJECT / GCP_PROJECT  Target GCP Project ID
  GCP_REGION                          Target GCP Region (default: us-central1)
  VIDEO_TRIMMER_BUCKET                GCS bucket name (default: video-preprocessing-\${PROJECT_ID})
  GCP_SERVICE_ACCOUNT                 Custom Service Account for the tool (default: video-trimmer-sa@...)

Options:
  -p, --project PROJECT_ID     Google Cloud Project ID (overrides .env)
  -r, --region REGION          Google Cloud Region (default: us-central1)
  -b, --bucket BUCKET_NAME     Custom GCS bucket name (default: video-preprocessing-\${PROJECT_ID})
  -s, --service-account SA     Custom service account email (default: video-trimmer-sa@...)
  -n, --dry-run                Preview provisioning commands without executing
  -h, --help                   Show this help message and exit

Examples:
  ./setup.sh                                          # Automatic provisioning based on current gcloud project
  ./setup.sh --project my-project-id                 # Specify GCP project ID explicitly
  ./setup.sh --bucket video-preprocessing-my-proj    # Use a custom storage bucket name
  ./setup.sh --dry-run                               # Preview actions without modifying cloud resources
EOF
    exit 0
}

# ------------------------------------------------------------------------------
# 2. Parse Command-Line Flags
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--project)
            PROJECT_ID="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -b|--bucket)
            BUCKET_NAME="$2"
            shift 2
            ;;
        -s|--service-account)
            SERVICE_ACCOUNT="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "[!] Error: Unknown argument \"$1\""
            usage
            ;;
    esac
done

echo "=================================================================="
echo "🚀 Video Trimmer: Native Google Cloud Resource Setup"
echo "   100% Native gcloud (Zero Terraform Dependency)"
echo "=================================================================="

# ------------------------------------------------------------------------------
# 3. Prerequisites & Context Resolution
# ------------------------------------------------------------------------------
if ! command -v gcloud &> /dev/null; then
    echo "[!] Error: gcloud CLI is not found in PATH."
    echo "    Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
fi

if [ -z "$PROJECT_ID" ]; then
    read -rp "Enter your Google Cloud Project ID: " PROJECT_ID
fi

if [ -z "$PROJECT_ID" ]; then
    echo "[!] Error: Project ID is required."
    exit 1
fi

if [ -z "$PROJECT_NUMBER" ]; then
    PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)" 2>/dev/null || true)"
fi

# Strip gs:// prefix if user specified gs://bucket
if [ -n "$BUCKET_NAME" ]; then
    BUCKET_NAME="${BUCKET_NAME#gs://}"
fi

# Default bucket name uses project name as suffix: video-preprocessing-${PROJECT_ID}
if [ -z "$BUCKET_NAME" ]; then
    BUCKET_NAME="video-preprocessing-${PROJECT_ID}"
fi

if [ -z "$SERVICE_ACCOUNT" ]; then
    SERVICE_ACCOUNT="video-trimmer-sa@${PROJECT_ID}.iam.gserviceaccount.com"
fi

echo "[✓] Target GCP Project: $PROJECT_ID"
if [ -n "$PROJECT_NUMBER" ]; then
    echo "[✓] GCP Project Number: $PROJECT_NUMBER"
fi
echo "[✓] Target Region:      $REGION"
echo "[✓] Storage Bucket:     gs://$BUCKET_NAME"
echo "[✓] Service Account:    $SERVICE_ACCOUNT"

# ------------------------------------------------------------------------------
# 3.5. Enable Required Google Cloud APIs & Verify ADC (with Google Drive Scope)
# ------------------------------------------------------------------------------
echo ""
echo "[*] Enabling required Google Cloud APIs (Vertex AI, GCS, Google Drive)..."
if [ "$DRY_RUN" = false ]; then
    gcloud services enable \
        aiplatform.googleapis.com \
        storage.googleapis.com \
        drive.googleapis.com iam.googleapis.com iamcredentials.googleapis.com iamcredentials.googleapis.com \
        --project="$PROJECT_ID" --quiet 2>/dev/null || true
    echo "    [✓] Enabled aiplatform.googleapis.com, storage.googleapis.com, drive.googleapis.com iam.googleapis.com iamcredentials.googleapis.com iamcredentials.googleapis.com."

    ADC_SCOPES="https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/drive.readonly"
    ADC_TOKEN="$(gcloud auth application-default print-access-token 2>/dev/null || true)"
    if [ -z "$ADC_TOKEN" ]; then
        echo "    [!] ADC not found. Launching login with Cloud Platform & Google Drive Read-Only scopes..."
        gcloud auth application-default login 
    else
        TOKEN_INFO="$(curl -s "https://oauth2.googleapis.com/tokeninfo?access_token=${ADC_TOKEN}" || true)"
        if ! echo "$TOKEN_INFO" | grep -q "drive"; then
            echo "    [!] Tip: To enable direct Google Drive link/folder ingestion, run:"
            echo "        gcloud auth application-default login "
        else
            echo "    [✓] ADC verified with Google Drive Read-Only scope."
        fi
    fi
    gcloud auth application-default set-quota-project "$PROJECT_ID" --quiet 2>/dev/null || true
else
    echo "    [Dry-Run] Would enable aiplatform.googleapis.com, storage.googleapis.com, drive.googleapis.com iam.googleapis.com iamcredentials.googleapis.com iamcredentials.googleapis.com and verify ADC scopes."
fi

# ------------------------------------------------------------------------------
# 4. Step 1: Storage Bucket Provisioning (Uniform Access, CORS, Lifecycle)
# ------------------------------------------------------------------------------
echo ""
echo "[*] Step 1: Provisioning / Verifying GCS Storage Bucket: gs://$BUCKET_NAME..."

if [ "$DRY_RUN" = false ]; then
    if ! gcloud storage buckets describe "gs://$BUCKET_NAME" --project="$PROJECT_ID" &>/dev/null; then
        echo "    [*] Creating GCS bucket: gs://$BUCKET_NAME..."
        gcloud storage buckets create "gs://$BUCKET_NAME" \
            --project="$PROJECT_ID" \
            --location="$REGION" \
            --uniform-bucket-level-access \
            --public-access-prevention \
            --quiet
        echo "    [✓] Storage bucket gs://$BUCKET_NAME created successfully."
    else
        echo "    [✓] Storage bucket gs://$BUCKET_NAME already exists."
    fi

    # Configure CORS for signed URL streaming and web playback
    CORS_FILE="$(mktemp 2>/dev/null || echo "/tmp/cors_$$.json")"
    cat << 'EOF' > "$CORS_FILE"
[
  {
    "origin": ["*"],
    "responseHeader": ["*"],
    "method": ["GET", "HEAD"],
    "maxAgeSeconds": 86400
  }
]
EOF
    gcloud storage buckets update "gs://$BUCKET_NAME" --cors-file="$CORS_FILE" --quiet 2>/dev/null || true
    rm -f "$CORS_FILE"
    echo "    [✓] Applied CORS configuration (24h cache, GET/HEAD)."

    # Configure Lifecycle Rules:
    # - raw/: Delete after 2 days (ephemeral video staging auto-cleanup)
    LIFECYCLE_FILE="$(mktemp 2>/dev/null || echo "/tmp/lifecycle_$$.json")"
    cat << 'EOF' > "$LIFECYCLE_FILE"
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {
        "age": 2,
        "matchesPrefix": ["raw/"]
      }
    },
    {
      "action": {"type": "Delete"},
      "condition": {
        "age": 15,
        "matchesPrefix": ["output/", "deliverables/", "trimmed/"]
      }
    }
  ]
}
EOF
    gcloud storage buckets update "gs://$BUCKET_NAME" --lifecycle-file="$LIFECYCLE_FILE" --quiet 2>/dev/null || true
    rm -f "$LIFECYCLE_FILE"
    echo "    [✓] Applied two-tier lifecycle rules (raw/: 2 days, output/deliverables/trimmed/: 15 days)."
else
    echo "    [Dry-Run] Would ensure GCS bucket gs://$BUCKET_NAME exists with CORS & 2-day ephemeral auto-delete on raw/."
fi

# ------------------------------------------------------------------------------
# 5. Step 2: Dedicated Service Account & Least-Privilege IAM Provisioning
# ------------------------------------------------------------------------------
echo ""
echo "[*] Step 2: Provisioning Dedicated Service Account & Least-Privilege IAM..."

SA_NAME="${SERVICE_ACCOUNT%%@*}"
if [ "$DRY_RUN" = false ]; then
    if ! gcloud iam service-accounts describe "$SERVICE_ACCOUNT" --project="$PROJECT_ID" &>/dev/null; then
        echo "    [*] Creating dedicated service account: $SERVICE_ACCOUNT..."
        gcloud iam service-accounts create "$SA_NAME" \
            --display-name="Video Trimmer Service Account" \
            --project="$PROJECT_ID" \
            --quiet 2>/dev/null || true
        echo "    [✓] Service account created: $SERVICE_ACCOUNT"
    else
        echo "    [✓] Dedicated service account $SERVICE_ACCOUNT already exists."
    fi

    # Project-level IAM bindings
    echo "    [*] Verifying project IAM bindings for $SERVICE_ACCOUNT..."
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SERVICE_ACCOUNT" \
        --role="roles/aiplatform.user" \
        --condition=None --quiet 2>/dev/null || true
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SERVICE_ACCOUNT" \
        --role="roles/logging.logWriter" \
        --condition=None --quiet 2>/dev/null || true
    echo "    [✓] Granted roles/aiplatform.user and roles/logging.logWriter."

    # Bucket-level least-privilege IAM bindings
    echo "    [*] Verifying storage IAM bindings on gs://$BUCKET_NAME..."
    gcloud storage buckets add-iam-policy-binding "gs://$BUCKET_NAME" \
        --member="serviceAccount:$SERVICE_ACCOUNT" \
        --role="roles/storage.objectUser" --quiet 2>/dev/null || true

    # Grant storage.objectUser to Vertex AI Service Agents
    if [ -n "$PROJECT_NUMBER" ]; then
        VERTEX_AGENTS=(
            "service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com"
            "service-${PROJECT_NUMBER}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
        )
        for sa in "${VERTEX_AGENTS[@]}"; do
            gcloud storage buckets add-iam-policy-binding "gs://$BUCKET_NAME" \
                --member="serviceAccount:$sa" \
                --role="roles/storage.objectUser" --quiet 2>/dev/null || true
        done
    fi
    echo "    [✓] Granted roles/storage.objectUser on gs://$BUCKET_NAME."
else
    echo "    [Dry-Run] Would ensure service account $SERVICE_ACCOUNT exists with roles/aiplatform.user and roles/logging.logWriter."
    echo "    [Dry-Run] Would grant roles/storage.objectUser on gs://$BUCKET_NAME to $SERVICE_ACCOUNT and Vertex AI service agents."
fi

# ------------------------------------------------------------------------------
# 6. Step 3: Local Environment File Configuration (.env)
# ------------------------------------------------------------------------------
echo ""
echo "[*] Step 3: Updating local environment configuration (.env)..."

ENV_FILE="$REPO_ROOT/.env"
if [ "$DRY_RUN" = false ]; then
    if [ ! -f "$ENV_FILE" ]; then
        cat << EOF > "$ENV_FILE"
# Google Cloud Vertex AI & Application Default Credentials (ADC)
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GOOGLE_CLOUD_LOCATION=global
GCP_REGION=$REGION

# GCS Staging Bucket (project name suffix)
VIDEO_TRIMMER_BUCKET=$BUCKET_NAME

# Default Multimodal Model
MODEL_NAME=gemini-3.8-flash
EOF
        echo "    [✓] Created new .env file with configured values."
    else
        # Update existing keys or append
        python3 - << PYEOF
import re

env_path = "$ENV_FILE"
with open(env_path, "r", encoding="utf-8") as f:
    content = f.read()

updates = {
    "GOOGLE_CLOUD_PROJECT": "$PROJECT_ID",
    "GOOGLE_CLOUD_LOCATION": "global",
    "GCP_REGION": "$REGION",
    "VIDEO_TRIMMER_BUCKET": "$BUCKET_NAME",
}

for k, v in updates.items():
    pattern = rf"^{k}=.*$"
    replacement = f"{k}={v}"
    if re.search(pattern, content, flags=re.MULTILINE):
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    else:
        content += f"\n{replacement}"

if "MODEL_NAME=" not in content:
    content += "\nMODEL_NAME=gemini-3.8-flash"

with open(env_path, "w", encoding="utf-8") as f:
    f.write(content.strip() + "\n")
PYEOF
        echo "    [✓] Updated existing .env file with active settings."
    fi
else
    echo "    [Dry-Run] Would write/update .env with GOOGLE_CLOUD_PROJECT=$PROJECT_ID and VIDEO_TRIMMER_BUCKET=$BUCKET_NAME."
fi

echo ""
echo "=================================================================="
echo "✅ Google Cloud Infrastructure Setup Complete!"
echo "=================================================================="
echo "GCP Project:    $PROJECT_ID"
echo "Storage Bucket: gs://$BUCKET_NAME"
echo "Service Account:$SERVICE_ACCOUNT"
echo ""
echo "Next steps:"
echo "1. Authenticate locally with Application Default Credentials (ADC):"
echo "   gcloud auth application-default login --scopes=\"https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/drive.readonly\""
echo ""
echo "2. Run Video Trimmer on your video:"
echo "   python video_trimmer.py -i \"/path/to/raw_footage.mp4\""
echo "=================================================================="

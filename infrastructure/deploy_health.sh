#!/usr/bin/env bash
# infrastructure/deploy_health.sh — Build and deploy the daily health check Lambda.
#
# Lightweight container: numpy/pandas/scipy/boto3 only (~120MB).
# Runs daily predictor health monitoring (Phases 2a, 2b, 5).
#
# Prerequisites:
#   - Docker installed and running
#   - AWS CLI configured
#   - ECR repo 'alpha-engine-health-check' exists in your account
#   - Lambda function 'alpha-engine-predictor-health-check' already created
#
# Usage:
#   ./infrastructure/deploy_health.sh                # full deploy
#   ./infrastructure/deploy_health.sh --dry-run      # build image only, skip AWS push
#
# Environment variables (auto-detected if not set):
#   AWS_ACCOUNT_ID   — 12-digit AWS account ID (auto-detected via aws sts)
#   AWS_REGION       — defaults to us-east-1

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
ECR_REPO="alpha-engine-health-check"
LAMBDA_FUNCTION="alpha-engine-predictor-health-check"
IMAGE_TAG="latest"
DRY_RUN=false

# Parse flags
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Resolve AWS identity ─────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
if [ -z "${AWS_ACCOUNT_ID:-}" ] && [ "$DRY_RUN" = false ]; then
  AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION" 2>/dev/null) || { echo "ERROR: Could not auto-detect AWS_ACCOUNT_ID. Set it manually or configure AWS CLI."; exit 1; }
  echo "Auto-detected AWS_ACCOUNT_ID: $AWS_ACCOUNT_ID"
fi

# Move to repo root (script may be called from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "Working directory: $REPO_ROOT"

# ── Stage alpha-engine-lib into vendor/ ──────────────────────────────────────
# alpha-engine-lib is a private repo that provides BasePreflight +
# setup_logging. The Dockerfile pip-installs it via a local path because
# git+https inside Docker needs a PAT secret (more ceremony than this
# staging approach). Mirrors alpha-engine-predictor/infrastructure/deploy.sh.
LIB_REPO_DIR="${LIB_REPO_DIR:-$(dirname "$REPO_ROOT")/alpha-engine-lib}"
LIB_STAGED_FROM_REPO=0

if [ -d "vendor/alpha-engine-lib" ]; then
  echo "Using existing vendor/alpha-engine-lib (local dev workflow)"
else
  if [ -d "$LIB_REPO_DIR" ]; then
    echo "Staging vendor/alpha-engine-lib from $LIB_REPO_DIR"
    mkdir -p vendor
    cp -R "$LIB_REPO_DIR" vendor/alpha-engine-lib
    LIB_STAGED_FROM_REPO=1
  else
    echo "ERROR: alpha-engine-lib not found — tried:"
    echo "  vendor/alpha-engine-lib (local dev)"
    echo "  $LIB_REPO_DIR (sibling checkout)"
    echo "Hint: clone cipher813/alpha-engine-lib as a sibling directory,"
    echo "      or set LIB_REPO_DIR=/path/to/alpha-engine-lib"
    exit 1
  fi
fi

# Cleanup staged artifacts on exit so a failed deploy doesn't leave
# vendor/ in a dev laptop checkout.
cleanup_staged_artifacts() {
  if [ "$LIB_STAGED_FROM_REPO" = "1" ] && [ -d vendor/alpha-engine-lib ]; then
    rm -rf vendor/alpha-engine-lib
    rmdir vendor 2>/dev/null || true
  fi
}
trap cleanup_staged_artifacts EXIT

# ── Step 1: Build Docker image ────────────────────────────────────────────────
echo ""
echo "==> Building Docker image..."
docker build --platform linux/amd64 --provenance=false --tag "${ECR_REPO}:${IMAGE_TAG}" --file lambda_health/Dockerfile .

echo "  Image built: ${ECR_REPO}:${IMAGE_TAG}"

if [ "$DRY_RUN" = true ]; then
  echo ""
  echo "==> DRY RUN: Skipping ECR push and Lambda update."
  echo "    Image built successfully as ${ECR_REPO}:${IMAGE_TAG}"
  exit 0
fi

# ── Step 2: Authenticate to ECR ───────────────────────────────────────────────
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

echo ""
echo "==> Authenticating to ECR (${ECR_REGISTRY})..."
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# ── Step 3: Tag and push image ────────────────────────────────────────────────
echo ""
echo "==> Tagging image: ${ECR_IMAGE}"
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_IMAGE}"

echo "==> Pushing to ECR..."
docker push "${ECR_IMAGE}"
echo "  Pushed: ${ECR_IMAGE}"

# ── Step 4: Update Lambda function code ──────────────────────────────────────
echo ""
echo "==> Updating Lambda function: ${LAMBDA_FUNCTION}"
aws lambda update-function-code --function-name "${LAMBDA_FUNCTION}" --image-uri "${ECR_IMAGE}" --region "${AWS_REGION}" --output json | python3 -c "import sys,json; d=json.load(sys.stdin); print('  FunctionArn:', d.get('FunctionArn','?')); print('  LastModified:', d.get('LastModified','?'))"

# ── Step 4b: Sync env vars from master .env ─────────────────────────────────
LAMBDA_ENV_FILE="$(dirname "$REPO_ROOT")/alpha-engine-data/.env"
if [ ! -f "$LAMBDA_ENV_FILE" ]; then
  LAMBDA_ENV_FILE="$REPO_ROOT/.env"
fi
if [ -f "$LAMBDA_ENV_FILE" ]; then
  LAMBDA_ENV_JSON=$(python3 -c "
import json
env = {}
with open('$LAMBDA_ENV_FILE') as f:
    for line in f:
        line = line.strip()
        if line == '# LAMBDA_SKIP':
            break
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('\"', \"'\"):
            val = val[1:-1]
        if key and val:
            env[key] = val
if env:
    print(json.dumps({'Variables': env}))
else:
    print('')
")
  if [ -n "$LAMBDA_ENV_JSON" ]; then
    echo ""
    echo "==> Syncing env vars from $LAMBDA_ENV_FILE"
    echo "  Keys: $(echo "$LAMBDA_ENV_JSON" | python3 -c "import sys,json; print(', '.join(json.load(sys.stdin).get('Variables',{}).keys()))")"
    aws lambda wait function-updated --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}" 2>/dev/null || sleep 5
    aws lambda update-function-configuration --function-name "${LAMBDA_FUNCTION}" --environment "$LAMBDA_ENV_JSON" --region "${AWS_REGION}" > /dev/null
  fi
else
  echo "  WARNING: No .env file found — Lambda env vars not updated"
fi

# ── Step 5: Wait for update to complete ──────────────────────────────────────
echo ""
echo "==> Waiting for Lambda update to complete..."
aws lambda wait function-updated --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}"

# ── Step 6: Publish version and update 'live' alias ──────────────────────────
echo ""
echo "==> Publishing Lambda version..."
VERSION=$(aws lambda publish-version --function-name "${LAMBDA_FUNCTION}" --query "Version" --output text --region "${AWS_REGION}")
echo "  Published version: ${VERSION}"

echo "==> Updating 'live' alias → version ${VERSION}"
aws lambda update-alias --function-name "${LAMBDA_FUNCTION}" --name live --function-version "${VERSION}" --region "${AWS_REGION}" 2>/dev/null || aws lambda create-alias --function-name "${LAMBDA_FUNCTION}" --name live --function-version "${VERSION}" --region "${AWS_REGION}"

# ── Step 7: Canary invocation ───────────────────────────────────────────────
echo ""
echo "==> Running canary invocation (dry_run=true)..."
CANARY_OUT=$(mktemp)
aws lambda invoke --function-name "${LAMBDA_FUNCTION}:live" --payload '{"dry_run": true}' --cli-binary-format raw-in-base64-out --cli-read-timeout 300 --region "${AWS_REGION}" "$CANARY_OUT" > /dev/null

CANARY_STATUS=$(python3 -c "import json; d=json.load(open('$CANARY_OUT')); print(d.get('statusCode', 0))" 2>/dev/null || echo "0")
rm -f "$CANARY_OUT"

if [ "$CANARY_STATUS" != "200" ]; then
  echo ""
  echo "ERROR: Canary returned status $CANARY_STATUS"
  echo "  Check CloudWatch Logs: /aws/lambda/${LAMBDA_FUNCTION}"
  exit 1
fi
echo "  Canary passed (status=$CANARY_STATUS)"

echo ""
echo "==> Deploy complete!"
echo "    Function : ${LAMBDA_FUNCTION}"
echo "    Version  : ${VERSION}"
echo "    Alias    : live → ${VERSION}"
echo "    Image    : ${ECR_IMAGE}"

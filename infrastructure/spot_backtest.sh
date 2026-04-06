#!/usr/bin/env bash
# infrastructure/spot_backtest.sh — Run weekly backtest on a spot EC2 instance.
#
# Launches a c5.large spot instance (~$0.03/hr), clones the backtester +
# predictor + executor repos, runs the full backtest pipeline with 10y of
# price data, uploads results to S3, and self-terminates.
#
# Usage:
#   ./infrastructure/spot_backtest.sh                   # full run (--mode all)
#   ./infrastructure/spot_backtest.sh --smoke-only      # quick validation, then terminate
#   ./infrastructure/spot_backtest.sh --mode simulate   # override backtest mode
#   ./infrastructure/spot_backtest.sh --instance-type c5.xlarge  # override instance type
#
# Prerequisites:
#   - AWS CLI configured (uses alpha-engine-executor-profile for S3/SES access)
#   - SSH key at ~/.ssh/alpha-engine-key.pem
#   - Code committed and pushed to origin (instance clones from GitHub)
#   - .env file with EMAIL_SENDER, EMAIL_RECIPIENTS, GMAIL_APP_PASSWORD
#   - config.yaml (gitignored — SCP'd to EC2 by this script)
#
# For scheduled weekly runs, call this script from the always-on EC2 cron
# or from an EventBridge → Lambda trigger:
#
#   0 8 * * 1  cd ~/alpha-engine-backtester && bash infrastructure/spot_backtest.sh >> /var/log/backtester-spot.log 2>&1

set -euo pipefail

# ── Ensure HOME is set (SSM RunCommand does not set it) ──────────────────────
export HOME="${HOME:-/home/ec2-user}"

# ── Load .env ────────────────────────────────────────────────────────────────
# Master .env lives in alpha-engine-data; fall back to ~/.alpha-engine.env
# (Step Functions SSM), then local .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_FILE="$(dirname "$REPO_ROOT")/alpha-engine-data/.env"
if [ ! -f "$ENV_FILE" ]; then
    ENV_FILE="$HOME/.alpha-engine.env"
fi
if [ ! -f "$ENV_FILE" ]; then
    ENV_FILE="$REPO_ROOT/.env"
fi
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo "Loaded .env from $ENV_FILE"
else
    echo "WARNING: No .env file found"
fi

# ── Configuration ──────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-alpha-engine-research}"
BRANCH="${BRANCH:-main}"
INSTANCE_TYPE="c5.large"            # 2 vCPU, 4GB RAM — sufficient for 10y backtest
AMI_ID="ami-0c421724a94bba6d6"      # Amazon Linux 2023 x86_64
KEY_NAME="alpha-engine-key"
KEY_FILE="$HOME/.ssh/alpha-engine-key.pem"
SECURITY_GROUP="sg-03cd3c4bd91e610b0"
SUBNET_ID="subnet-e07166ec"
IAM_PROFILE="alpha-engine-executor-profile"
BACKTEST_MODE="all"

# ── Parse flags ──────────────────────────────────────────────────────────────
RUN_MODE="full"  # full | smoke-only
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-only) RUN_MODE="smoke-only"; shift ;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
        --mode) BACKTEST_MODE="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

echo "═══════════════════════════════════════════════════════════════"
echo "  Backtester Spot Run — $(date +%Y-%m-%d)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Instance type : $INSTANCE_TYPE"
echo "  AMI           : $AMI_ID"
echo "  Region        : $AWS_REGION"
echo "  Branch        : $BRANCH"
echo "  Backtest mode : $BACKTEST_MODE"
echo "  Run mode      : $RUN_MODE"
echo "  S3 bucket     : $S3_BUCKET"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────
if [ ! -f "$KEY_FILE" ]; then
    echo "ERROR: SSH key not found at $KEY_FILE"
    exit 1
fi

if [ ! -f "$REPO_ROOT/config.yaml" ]; then
    echo "ERROR: config.yaml not found — copy from config.yaml.example"
    exit 1
fi

# ── Launch spot instance ──────────────────────────────────────────────────────
echo "==> Requesting spot instance ($INSTANCE_TYPE)..."

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --subnet-id "$SUBNET_ID" \
    --iam-instance-profile Name="$IAM_PROFILE" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=alpha-engine-backtest-$(date +%Y%m%d)}]" \
    --region "$AWS_REGION" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "  Instance ID: $INSTANCE_ID"

# Cleanup function — always terminate the instance
cleanup() {
    echo ""
    echo "==> Terminating spot instance $INSTANCE_ID..."
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" --output text > /dev/null 2>&1 || true
    echo "  Instance terminated."
}
trap cleanup EXIT

# Wait for instance to be running
echo "==> Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text \
    --region "$AWS_REGION")

if [ "$PUBLIC_IP" = "None" ] || [ -z "$PUBLIC_IP" ]; then
    echo "ERROR: Instance has no public IP. Check subnet/VPC configuration."
    exit 1
fi

echo "  Public IP: $PUBLIC_IP"

# ── Wait for SSH ──────────────────────────────────────────────────────────────
echo "==> Waiting for SSH to become available..."
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -o LogLevel=ERROR"

for i in $(seq 1 30); do
    if ssh $SSH_OPTS -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "echo ok" 2>/dev/null; then
        echo "  SSH ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: SSH not available after 150s"
        exit 1
    fi
    sleep 5
done

# Helper: run command on EC2
run_remote() {
    ssh $SSH_OPTS -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "$@"
}

# ── Bootstrap environment ──────────────────────────────────────────────────────
echo "==> Bootstrapping EC2 environment..."
run_remote bash -s <<'BOOTSTRAP'
set -euo pipefail

# Amazon Linux 2023: install Python 3.12, git, gcc
sudo dnf install -y -q python3.12 python3.12-pip python3.12-devel git gcc 2>/dev/null || \
    sudo dnf install -y -q python3 python3-pip python3-devel git gcc

if command -v python3.12 &>/dev/null; then
    PYTHON=python3.12
else
    PYTHON=python3
fi
echo "Using: $($PYTHON --version)"

# SSH for GitHub
mkdir -p ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
BOOTSTRAP

echo "==> Cloning repositories (branch: $BRANCH)..."
# Clone all repos needed for backtest (backtester imports executor + predictor + flow-doctor)
for REPO in alpha-engine-backtester alpha-engine alpha-engine-predictor flow-doctor; do
    echo "  Cloning $REPO..."
    ssh -A $SSH_OPTS -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" \
        "git clone --depth 1 --branch $BRANCH git@github.com:cipher813/$REPO.git /home/ec2-user/$REPO" 2>/dev/null || {
        HTTPS_URL="https://github.com/cipher813/$REPO.git"
        run_remote "git clone --depth 1 --branch $BRANCH $HTTPS_URL /home/ec2-user/$REPO"
    }
done

echo "==> Installing Python dependencies..."
run_remote bash -s <<'DEPS'
set -euo pipefail
cd /home/ec2-user/alpha-engine-backtester

if command -v python3.12 &>/dev/null; then
    PIP="python3.12 -m pip"
else
    PIP="python3 -m pip"
fi

$PIP install --upgrade pip -q
$PIP install -q -r requirements.txt

# Install flow-doctor from bundled source (not on PyPI)
$PIP install -q -e /home/ec2-user/flow-doctor

# Also install predictor deps (needed for GBM inference + feature computation)
cd /home/ec2-user/alpha-engine-predictor
if [ -f requirements.txt ]; then
    $PIP install -q -r requirements.txt 2>/dev/null || true
fi

# Force numpy<2 after all deps (pyarrow compiled against numpy 1.x)
$PIP install -q 'numpy<2'

echo "Dependencies installed."
DEPS

# ── Copy config files ──────────────────────────────────────────────────────────
echo "==> Uploading config.yaml and .env..."
scp $SSH_OPTS -i "$KEY_FILE" \
    "$REPO_ROOT/config.yaml" \
    ec2-user@"$PUBLIC_IP":/home/ec2-user/alpha-engine-backtester/config.yaml

if [ -f "$ENV_FILE" ]; then
    scp $SSH_OPTS -i "$KEY_FILE" \
        "$ENV_FILE" \
        ec2-user@"$PUBLIC_IP":/home/ec2-user/alpha-engine-backtester/.env
fi

# Copy executor config (needed for simulation).
# Try EC2 path first (when launched from always-on EC2), then local dev path.
EXECUTOR_CONFIG=""
for candidate in \
    "$HOME/alpha-engine/config/risk.yaml" \
    "$HOME/Development/alpha-engine/config/risk.yaml"; do
    if [ -f "$candidate" ]; then
        EXECUTOR_CONFIG="$candidate"
        break
    fi
done

if [ -n "$EXECUTOR_CONFIG" ]; then
    echo "  Uploading risk.yaml from $EXECUTOR_CONFIG"
    run_remote "mkdir -p /home/ec2-user/alpha-engine/config"
    scp $SSH_OPTS -i "$KEY_FILE" \
        "$EXECUTOR_CONFIG" \
        ec2-user@"$PUBLIC_IP":/home/ec2-user/alpha-engine/config/risk.yaml
else
    echo "  WARNING: risk.yaml not found — simulation will be skipped"
fi

# Copy predictor config (needed for predictor backtest).
PREDICTOR_CONFIG=""
for candidate in \
    "$HOME/alpha-engine-predictor/config/predictor.yaml" \
    "$HOME/Development/alpha-engine-predictor/config/predictor.yaml"; do
    if [ -f "$candidate" ]; then
        PREDICTOR_CONFIG="$candidate"
        break
    fi
done

if [ -n "$PREDICTOR_CONFIG" ]; then
    echo "  Uploading predictor.yaml from $PREDICTOR_CONFIG"
    run_remote "mkdir -p /home/ec2-user/alpha-engine-predictor/config"
    scp $SSH_OPTS -i "$KEY_FILE" \
        "$PREDICTOR_CONFIG" \
        ec2-user@"$PUBLIC_IP":/home/ec2-user/alpha-engine-predictor/config/predictor.yaml
else
    echo "  WARNING: predictor.yaml not found — predictor backtest will be skipped"
fi

# Bootstrap predictor data cache (slim cache parquets + sector_map required for backtest)
echo "==> Downloading predictor slim cache from S3 (~25 MB)..."
run_remote bash -s <<'CACHE'
set -euo pipefail
CACHE_DIR="/home/ec2-user/alpha-engine-predictor/data/cache"
mkdir -p "$CACHE_DIR"
if command -v aws &>/dev/null; then
    aws s3 cp s3://alpha-engine-research/predictor/price_cache/sector_map.json "$CACHE_DIR/sector_map.json" 2>/dev/null || true
    aws s3 sync s3://alpha-engine-research/predictor/price_cache_slim/ "$CACHE_DIR/" --quiet 2>/dev/null || true
fi
echo "Predictor cache dir: $(ls "$CACHE_DIR"/*.parquet 2>/dev/null | wc -l) parquet files"
CACHE

# ── Build env export command ─────────────────────────────────────────────────
ENV_SOURCE='set -a; [ -f /home/ec2-user/alpha-engine-backtester/.env ] && source /home/ec2-user/alpha-engine-backtester/.env; set +a; export XDG_CACHE_HOME=/tmp;'

# Determine python binary on remote
REMOTE_PYTHON=$(run_remote "command -v python3.12 || command -v python3")

# ── Smoke test ────────────────────────────────────────────────────────────────
if [ "$RUN_MODE" = "smoke-only" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  SMOKE TEST"
    echo "═══════════════════════════════════════════════════════════════"

    run_remote bash -s <<SMOKE
set -euo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

$REMOTE_PYTHON backtest.py --mode signal-quality --log-level INFO 2>&1 | tail -30

echo ""
echo "Smoke test complete."
SMOKE

    echo "==> Smoke-only mode — instance will be terminated."
    exit 0
fi

# ── Full backtest ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  FULL BACKTEST (--mode $BACKTEST_MODE)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_remote bash -s <<BACKTEST
set -euo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

echo "Starting backtest at \$(date)"
$REMOTE_PYTHON backtest.py --mode $BACKTEST_MODE --upload --log-level INFO 2>&1

echo ""
echo "Backtest complete at \$(date)"
BACKTEST

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Backtest complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"

# Emit CloudWatch heartbeat on successful completion
aws cloudwatch put-metric-data \
  --namespace "AlphaEngine" \
  --metric-name "Heartbeat" \
  --dimensions "Process=backtester" \
  --value 1 --unit "Count" \
  --region "${AWS_REGION:-us-east-1}" 2>/dev/null \
  && echo "Heartbeat emitted: backtester" \
  || echo "WARNING: Failed to emit heartbeat (non-fatal)"

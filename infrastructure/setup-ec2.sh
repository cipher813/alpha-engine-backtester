#!/bin/bash
# One-time setup for alpha-engine-backtester on the same EC2 instance as the executor.
#
# Deploy from local:
#   git push origin main
#   ae "bash ~/alpha-engine-backtester/infrastructure/setup-ec2.sh"
#
# Or if the repo isn't cloned yet, SSH in first:
#   ae
#   git clone https://github.com/brianmcmahon/alpha-engine-backtester.git
#   bash ~/alpha-engine-backtester/infrastructure/setup-ec2.sh

set -euo pipefail

REPO_DIR="/home/ec2-user/alpha-engine-backtester"

echo "=== Alpha Engine Backtester — EC2 setup ==="

# ── 1. Pull latest code ───────────────────────────────────────────────────────
cd "$REPO_DIR"
git pull

# ── 2. Create virtualenv (separate from executor's venv — different deps) ─────
if [ ! -d ".venv" ]; then
    echo "Creating virtualenv..."
    python3.11 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

# ── 3. Create log file ────────────────────────────────────────────────────────
sudo touch /var/log/backtester.log
sudo chown ec2-user:ec2-user /var/log/backtester.log

# ── 4. Register cron job ──────────────────────────────────────────────────────
bash "$REPO_DIR/infrastructure/add-cron.sh"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Test run:"
echo "  cd $REPO_DIR && .venv/bin/python backtest.py --mode signal-quality"
echo ""
echo "Logs:"
echo "  tail -f /var/log/backtester.log"

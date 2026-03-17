#!/bin/bash
# Register the weekly backtester cron job.
# Safe to run multiple times — replaces existing entry.
#
# Schedule: Mondays at 08:00 UTC (1 hour after predictor training starts at 07:00)
# This ensures the backtester runs against the freshly trained GBM model.
# (EC2 started at 06:45 UTC by EventBridge ae-backtester-start rule)
#
# Usage:
#   GMAIL_APP_PASSWORD=xxx bash infrastructure/add-cron.sh

set -euo pipefail

GMAIL_PW="${GMAIL_APP_PASSWORD:-}"

if [ -z "$GMAIL_PW" ]; then
    echo "ERROR: GMAIL_APP_PASSWORD not set. Pass it as env var or export it first."
    exit 1
fi

CRON_LINE="0 8 * * 1  cd /home/ec2-user/alpha-engine-backtester && git pull --ff-only >> /var/log/backtester.log 2>&1 && cd /home/ec2-user/alpha-engine && git pull --ff-only >> /var/log/backtester.log 2>&1 && cd /home/ec2-user/alpha-engine-backtester && GMAIL_APP_PASSWORD=${GMAIL_PW} .venv/bin/python backtest.py --mode all --upload --stop-instance >> /var/log/backtester.log 2>&1"

# Remove existing backtester entry, then add new one
EXISTING=$(crontab -l 2>/dev/null || true)
FILTERED=$(echo "$EXISTING" | grep -v "alpha-engine-backtester" || true)

{
    echo "$FILTERED"
    echo "$CRON_LINE"
} | crontab -

echo "Backtester cron job registered: Mondays 08:00 UTC"
echo ""
echo "Current crontab:"
crontab -l

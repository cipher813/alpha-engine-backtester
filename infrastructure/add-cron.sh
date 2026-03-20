#!/bin/bash
# Register the weekly backtester cron job.
# Safe to run multiple times — replaces existing entry.
#
# The backtester runs on a spot instance launched from the always-on EC2.
# The spot_backtest.sh script handles the full lifecycle:
#   launch spot → clone repos → install deps → run backtest → terminate
#
# Schedule: Mondays at 08:00 UTC (1 hour after predictor training starts at 07:00)
# This ensures the backtester runs against the freshly trained GBM model.
#
# Secrets sourced from ~/.alpha-engine.env (shared with executor).
#
# Usage:
#   bash infrastructure/add-cron.sh

set -euo pipefail

ENV_FILE="/home/ec2-user/.alpha-engine.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: ${ENV_FILE} not found."
    echo "Create it with GMAIL_APP_PASSWORD, then chmod 600."
    exit 1
fi

SOURCE_ENV=". ${ENV_FILE} &&"

# Launch spot instance for the full backtest (10y data, param sweep, upload results)
CRON_LINE="0 8 * * 1  cd /home/ec2-user/alpha-engine-backtester && git pull --ff-only >> /var/log/backtester.log 2>&1 && ${SOURCE_ENV} bash infrastructure/spot_backtest.sh >> /var/log/backtester.log 2>&1"

# Remove existing backtester entry, then add new one
EXISTING=$(crontab -l 2>/dev/null || true)
FILTERED=$(echo "$EXISTING" | grep -v "alpha-engine-backtester" || true)

{
    echo "$FILTERED"
    echo "$CRON_LINE"
} | crontab -

echo "Backtester cron job registered: Mondays 08:00 UTC"
echo "  Mode: spot instance (launched from always-on EC2)"
echo "  Secrets: sourced from ${ENV_FILE}"
echo ""
echo "Current crontab:"
crontab -l

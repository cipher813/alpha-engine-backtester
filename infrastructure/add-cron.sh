#!/bin/bash
# Register the weekly backtester cron job.
# Safe to run multiple times — won't add duplicate entries.
#
# Schedule: Sundays at 14:00 UTC = 9:00am ET = 6:00am PT

CRON_LINE="0 14 * * 0  cd /home/ec2-user/alpha-engine-backtester && .venv/bin/python backtest.py --mode signal-quality --upload >> /var/log/backtester.log 2>&1"

if crontab -l 2>/dev/null | grep -qF "alpha-engine-backtester"; then
    echo "Backtester cron job already registered — no change."
    crontab -l | grep "alpha-engine-backtester"
else
    echo "Adding backtester cron job..."
    (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
    echo "Done. Current crontab:"
    crontab -l
fi

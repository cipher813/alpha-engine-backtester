#!/bin/bash
# Register the weekly backtester cron job.
# Safe to run multiple times — removes old entry if schedule changed.
#
# Schedule: Mondays at 07:00 UTC = Sunday ~11pm PDT / ~10pm PST
# (EC2 started at 06:45 UTC by EventBridge ae-backtester-start rule)

CRON_LINE="0 7 * * 1  cd /home/ec2-user/alpha-engine-backtester && .venv/bin/python backtest.py --mode signal-quality --upload >> /var/log/backtester.log 2>&1"

if crontab -l 2>/dev/null | grep -qF "alpha-engine-backtester"; then
    # Check if schedule matches
    EXISTING=$(crontab -l 2>/dev/null | grep "alpha-engine-backtester")
    if echo "$EXISTING" | grep -qF "0 7 * * 1"; then
        echo "Backtester cron job already registered with correct schedule — no change."
        echo "  $EXISTING"
    else
        echo "Updating backtester cron schedule..."
        echo "  Old: $EXISTING"
        # Remove old entry and add new one
        (crontab -l 2>/dev/null | grep -v "alpha-engine-backtester"; echo "$CRON_LINE") | crontab -
        echo "  New: $CRON_LINE"
        echo "Done."
    fi
else
    echo "Adding backtester cron job..."
    (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
    echo "Done. Current crontab:"
    crontab -l
fi

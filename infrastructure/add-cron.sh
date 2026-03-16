#!/bin/bash
# Register the weekly backtester cron job.
# Safe to run multiple times — removes old entry if schedule changed.
#
# Schedule: Mondays at 08:00 UTC (1 hour after predictor training starts at 07:00)
# This ensures the backtester runs against the freshly trained GBM model.
# (EC2 started at 06:45 UTC by EventBridge ae-backtester-start rule)

CRON_LINE="0 8 * * 1  cd /home/ec2-user/alpha-engine-backtester && GMAIL_APP_PASSWORD=obopkmmizphlfplz .venv/bin/python backtest.py --mode all --upload --stop-instance >> /var/log/backtester.log 2>&1"

if crontab -l 2>/dev/null | grep -qF "alpha-engine-backtester"; then
    EXISTING=$(crontab -l 2>/dev/null | grep "alpha-engine-backtester")
    if echo "$EXISTING" | grep -q "^0 8 \* \* 1"; then
        echo "Backtester cron job already registered with correct schedule — no change."
        echo "  $EXISTING"
    else
        echo "Updating backtester cron schedule..."
        echo "  Old: $EXISTING"
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

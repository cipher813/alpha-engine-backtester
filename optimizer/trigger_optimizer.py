"""
trigger_optimizer.py — auto-disable underperforming entry triggers.

Reads trigger scorecard results from analysis/trigger_scorecard.py.
If a trigger type has consistently negative entry timing alpha over
sufficient samples, recommends disabling it by writing a disabled_triggers
list to config/executor_params.json on S3.

Min-data gate: requires >= 50 trades per trigger type before recommending
any disabling. Until then, all triggers remain active.
"""

import json
import logging
from datetime import date

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

S3_PARAMS_KEY = "config/executor_params.json"

_MIN_TRADES_PER_TRIGGER = 50
_MAX_NEGATIVE_ALPHA_PCT = -0.005  # -0.5% avg alpha → candidate for disabling
_MAX_UNFAVORABLE_SLIPPAGE_BPS = 20  # 20bps mean unfavorable slippage

_cfg: dict = {}


def init_config(config: dict) -> None:
    global _cfg
    _cfg = config.get("trigger_optimizer", {})


def analyze(trigger_scorecard: dict) -> dict:
    """
    Analyze trigger scorecard and recommend which triggers to disable.

    Args:
        trigger_scorecard: dict from trigger_scorecard.compute_trigger_scorecard()

    Returns:
        dict with status, recommendations, and disable list.
    """
    if not trigger_scorecard or trigger_scorecard.get("status") != "ok":
        return {"status": "insufficient_data", "note": "No trigger scorecard available"}

    triggers = trigger_scorecard.get("triggers", [])
    if not triggers:
        return {"status": "insufficient_data", "note": "No trigger data in scorecard"}

    min_trades = _cfg.get("min_trades_per_trigger", _MIN_TRADES_PER_TRIGGER)
    alpha_threshold = _cfg.get("max_negative_alpha_pct", _MAX_NEGATIVE_ALPHA_PCT)
    slippage_threshold = _cfg.get("max_unfavorable_slippage_bps", _MAX_UNFAVORABLE_SLIPPAGE_BPS)

    recommendations = []
    disable_list = []
    total_evaluated = 0

    for t in triggers:
        name = t.get("trigger", "unknown")
        n = t.get("n_trades", 0)
        avg_alpha = t.get("avg_realized_alpha", 0)
        avg_slippage_vs_open = t.get("avg_slippage_vs_open_pct", 0)

        if n < min_trades:
            recommendations.append({
                "trigger": name,
                "action": "keep",
                "reason": f"insufficient data ({n} < {min_trades} trades)",
                "n_trades": n,
            })
            continue

        total_evaluated += 1
        should_disable = False
        reasons = []

        if avg_alpha is not None and avg_alpha < alpha_threshold:
            should_disable = True
            reasons.append(f"negative avg alpha ({avg_alpha:.3%})")

        if avg_slippage_vs_open is not None and avg_slippage_vs_open * 10000 > slippage_threshold:
            reasons.append(f"high slippage ({avg_slippage_vs_open * 10000:.0f}bps)")

        win_rate = t.get("win_rate_vs_spy")
        if win_rate is not None and win_rate < 0.40 and n >= min_trades:
            should_disable = True
            reasons.append(f"low win rate ({win_rate:.0%})")

        action = "disable" if should_disable else "keep"
        if should_disable:
            disable_list.append(name)

        recommendations.append({
            "trigger": name,
            "action": action,
            "reasons": reasons,
            "n_trades": n,
            "avg_alpha": round(avg_alpha, 4) if avg_alpha is not None else None,
            "win_rate": round(win_rate, 3) if win_rate is not None else None,
        })

    # Never disable ALL triggers — always keep at least time_expiry as fallback
    if len(disable_list) >= len([t for t in triggers if t.get("n_trades", 0) >= min_trades]):
        disable_list = [d for d in disable_list if d != "time_expiry"]
        for r in recommendations:
            if r["trigger"] == "time_expiry" and r["action"] == "disable":
                r["action"] = "keep"
                r["reasons"] = ["preserved as fallback — cannot disable all triggers"]

    return {
        "status": "ok" if total_evaluated > 0 else "insufficient_data",
        "total_evaluated": total_evaluated,
        "recommendations": recommendations,
        "disabled_triggers": disable_list,
        "min_trades_threshold": min_trades,
    }


def apply(result: dict, bucket: str) -> dict:
    """Write disabled_triggers list to executor_params.json on S3."""
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    disable_list = result.get("disabled_triggers", [])

    # Read current executor params
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=S3_PARAMS_KEY)
        current = json.loads(obj["Body"].read())
    except Exception:
        current = {}

    current_disabled = current.get("disabled_triggers", [])
    if set(disable_list) == set(current_disabled):
        return {"applied": False, "reason": "no change from current disabled list"}

    current["disabled_triggers"] = disable_list
    current["disabled_triggers_updated_at"] = str(date.today())

    body = json.dumps(current, indent=2)
    s3.put_object(Bucket=bucket, Key=S3_PARAMS_KEY, Body=body, ContentType="application/json")
    logger.info("Disabled triggers updated in S3: %s", disable_list)

    return {"applied": True, "disabled_triggers": disable_list}

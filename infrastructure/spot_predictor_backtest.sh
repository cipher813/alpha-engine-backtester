#!/usr/bin/env bash
# infrastructure/spot_predictor_backtest.sh — Predictor-backtest stage
# (predictor_pipeline: GBM inference + 10y ArcticDB read + predictor param
# sweep, plus Phase 4a/b/c) on a spot EC2 instance.
#
# alpha-engine-config-I4442: one of five scripts split out of the
# spot_backtest.sh monolith, mapped 1:1 to the Saturday SF's
# PredictorBacktest state, which currently calls:
#
#   spot_backtest.sh --mode=predictor-backtest --no-pit-parity \
#       --skip-stages=parity,evaluator{preflight_args}
#
# Runs AFTER the Backtester state — reads the backtest/{RUN_DATE}/ sweep
# artifacts Backtester just wrote and writes its own predictor_sweep_df /
# predictor_stats to the same prefix. This PR does not wire the SF to this
# script (separate nousergon-data PR, sequenced after); spot_backtest.sh
# stays the currently-wired path, unchanged.
#
# Usage:
#   ./infrastructure/spot_predictor_backtest.sh
#   ./infrastructure/spot_predictor_backtest.sh --preflight-only
#   ./infrastructure/spot_predictor_backtest.sh --run-date 2026-08-08
#   ./infrastructure/spot_predictor_backtest.sh --help

set -euo pipefail
export HOME="${HOME:-/home/ec2-user}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPOT_STAGE_NAME="predictor-backtest"

# shellcheck source=./_spot_common.sh
source "$SCRIPT_DIR/_spot_common.sh"
spot_common_init_defaults
_ORIG_ARGS=("$@")

usage() {
    cat <<'EOF'
spot_predictor_backtest.sh — Predictor-backtest stage (GBM inference + 10y
ArcticDB read + predictor param sweep, Phase 4a/b/c) on a spot EC2 instance.
Equivalent to:
  spot_backtest.sh --mode=predictor-backtest --no-pit-parity --skip-stages=parity,evaluator

Depends on the Backtester stage's backtest/{RUN_DATE}/ output already
existing in S3 — the stage preflight asserts this before spending on a spot.

Flags:
  --preflight-only        Boot + deps + smoke harness only, exit 0, zero spend
  --run-date DATE         Override RUN_DATE (default: today, normalized to NYSE trading day)
  --branch BRANCH         Git branch the spot clones (default: main)
  --instance-type TYPE    Override instance-type rotation (default: >=16GB RAM floor)
  --dry-run
  --skip-phases LIST / --only-phases LIST / --force / --force-phases LIST
  --skip-phase4-evaluations / --use-vectorized-sweep
  --help                  Print this and exit 0 (no AWS calls made)
EOF
}

spot_common_parse_flags "$@"
if [ "$SHOW_HELP" = "1" ]; then
    usage
    exit 0
fi
spot_common_compute_phase_flags

echo "═══════════════════════════════════════════════════════════════"
echo "  Predictor-Backtest Spot Run (stage=backtest, mode=predictor-backtest) — $(date +%Y-%m-%d)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Branch        : $BRANCH"
echo "  Preflight-only: $PREFLIGHT_ONLY"
echo "  Spot attempt  : $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS"
echo ""

spot_common_normalize_run_date

# ── Preflight checks ─────────────────────────────────────────────────────────
if [ "$PREFLIGHT_ONLY" != "1" ] && [ ! -f "$REPO_ROOT/config.yaml" ]; then
    echo "ERROR: config.yaml not found — copy from config.yaml.example"
    exit 1
fi
spot_common_resolve_executor_config
# This stage RUNS predictor_pipeline — unlike the monolith's uniform
# soft-skip, predictor.yaml is REQUIRED here: without it this stage cannot
# produce a real result, so fail loud before spend rather than silently
# writing nothing.
spot_common_resolve_predictor_config 1

# ── Stage-specific preflight (weekly-sf-policy.md §2.2) ──────────────────────
# PredictorBacktest runs after Backtester — it reads the sweep artifacts
# Backtester just wrote. Assert that upstream output exists BEFORE
# provisioning a spot: this is the exact class of miss the 2026-07-25
# incident named (a downstream stage's code path unexercised until deep
# into the run).
preflight_predictor_backtest() {
    echo "==> Stage preflight: predictor-backtest"
    if ! spot_common_s3_prefix_nonempty "backtest/${RUN_DATE}/"; then
        echo "ERROR: s3://${S3_BUCKET}/backtest/${RUN_DATE}/ is empty or unreachable." >&2
        echo "       This stage reads the Backtester stage's sweep output from that prefix — it must run and succeed first." >&2
        echo "       Failing before spend rather than discovering this mid-run." >&2
        exit 1
    fi
    echo "  stage preflight OK (backtest/${RUN_DATE}/ has upstream Backtester output)."
}
preflight_predictor_backtest

echo "==> Dispatcher pre-launch preflight (fail-fast before provisioning spot)..."
spot_common_pre_launch_preflight \
    "$REPO_ROOT/backtest.py" \
    "$REPO_ROOT/preflight.py" \
    "$REPO_ROOT/pipeline_common.py" \
    "$REPO_ROOT/synthetic/predictor_backtest.py"

# This mode runs predictor_pipeline (peak RSS ~2.8 GB) — apply the >=16 GB
# instance floor (I3280) unless the operator passed an explicit --instance-type.
spot_common_apply_large_universe_ram_floor
echo "  Instance types: $INSTANCE_TYPES"

spot_common_launch_instance
spot_common_install_cleanup_trap

echo "==> Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

spot_common_stage_configs
spot_common_wait_for_ssm_agent
spot_common_bootstrap
spot_common_install_deps
spot_common_fetch_predictor_cache
spot_common_build_env_source
spot_common_run_preflight_only_and_maybe_exit

# ── Stage: backtest (mode=predictor-backtest) ────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  PREDICTOR-BACKTEST (--mode predictor-backtest)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_ssm "predictor-backtest" "$MAX_RUNTIME_SECONDS" <<BACKTEST
set -eo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

RUN_DATE="${RUN_DATE}"

echo "▶ stage=backtest START at \$(date -u +%H:%M:%S)"
_BT_RC=0
$REMOTE_PYTHON -u backtest.py --mode predictor-backtest --date "\${RUN_DATE}" --upload --log-level INFO $BACKTEST_SKIP_PHASE4_FLAG $BACKTEST_PHASE_FLAGS 2>&1 || _BT_RC=\$?
if [ "\$_BT_RC" -ne 0 ]; then
    # config-I7258: preserve the REAL exit code (a bare \`exit 1\` here
    # launders rc=137/OOM into a generic 1 before krepis.ssm_dispatcher
    # (>=0.60.0) can classify it via SSM's ResponseCode).
    case "\$_BT_RC" in
        137|-9) echo "ERROR: backtest.py SIGKILLed (rc=\$_BT_RC) — likely OOM on \$(hostname) (\$(curl -s -m 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo unknown-instance-type)). Spot run marked FAILED — check flow-doctor alerts." >&2 ;;
        124|-14|143) echo "ERROR: backtest.py timed out (rc=\$_BT_RC) on \$(hostname). Spot run marked FAILED — check flow-doctor alerts." >&2 ;;
        *) echo "ERROR: backtest.py failed (rc=\$_BT_RC). Spot run marked FAILED — check flow-doctor alerts." >&2 ;;
    esac
    exit "\$_BT_RC"
fi
echo "▶ stage=backtest END at \$(date -u +%H:%M:%S)"

echo ""
echo "Predictor-backtest stage complete at \$(date)"
BACKTEST

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Predictor-backtest complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"

# No CloudWatch heartbeat emitted here: unlike Process=backtester and
# Process=evaluator, no Process=predictor-backtest alarm exists today.
# Emitting an unsubscribed metric repeats the exact class of drift
# config-I5786 documented (a heartbeat with no alarm is a metric with no
# subscriber, observability-policy.md §5). Tracked as a follow-up:
# alpha-engine-config-I6710 (per-stage heartbeat + alarm for the newly-
# independent PredictorBacktest/PortfolioOptimizerBacktest/Parity stages).

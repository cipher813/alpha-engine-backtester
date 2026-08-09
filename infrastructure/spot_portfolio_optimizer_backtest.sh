#!/usr/bin/env bash
# infrastructure/spot_portfolio_optimizer_backtest.sh — Portfolio-optimizer
# stage (portfolio_optimizer_gate + cov_estimator_sweep + gamma_sweep) on a
# spot EC2 instance.
#
# alpha-engine-config-I4442: one of five scripts split out of the
# spot_backtest.sh monolith, mapped 1:1 to the Saturday SF's
# PortfolioOptimizerBacktest state, which currently calls:
#
#   spot_backtest.sh --mode=portfolio-optimizer-backtest --no-pit-parity \
#       --skip-stages=parity,evaluator{preflight_args}
#
# Runs AFTER the PredictorBacktest state. Each of its three phases re-runs
# the predictor pipeline (GBM inference + 10y ArcticDB read) internally.
# Reads/writes the same backtest/{RUN_DATE}/ prefix. This PR does not wire
# the SF to this script (separate nousergon-data PR, sequenced after);
# spot_backtest.sh stays the currently-wired path, unchanged.
#
# Usage:
#   ./infrastructure/spot_portfolio_optimizer_backtest.sh
#   ./infrastructure/spot_portfolio_optimizer_backtest.sh --preflight-only
#   ./infrastructure/spot_portfolio_optimizer_backtest.sh --run-date 2026-08-08
#   ./infrastructure/spot_portfolio_optimizer_backtest.sh --help

set -euo pipefail
export HOME="${HOME:-/home/ec2-user}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPOT_STAGE_NAME="portfolio-optimizer"

# shellcheck source=./_spot_common.sh
source "$SCRIPT_DIR/_spot_common.sh"
spot_common_init_defaults
_ORIG_ARGS=("$@")

usage() {
    cat <<'EOF'
spot_portfolio_optimizer_backtest.sh — Portfolio-optimizer stage
(portfolio_optimizer_gate + cov_estimator_sweep + gamma_sweep) on a spot EC2
instance. Equivalent to:
  spot_backtest.sh --mode=portfolio-optimizer-backtest --no-pit-parity --skip-stages=parity,evaluator

Depends on the PredictorBacktest stage's backtest/{RUN_DATE}/ predictor
sweep output already existing in S3 — the stage preflight asserts this
before spending on a spot.

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
echo "  Portfolio-Optimizer-Backtest Spot Run (stage=backtest, mode=portfolio-optimizer-backtest) — $(date +%Y-%m-%d)"
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
# This stage re-runs predictor_pipeline internally in each of its three
# phases — predictor.yaml is REQUIRED, same reasoning as
# spot_predictor_backtest.sh.
spot_common_resolve_predictor_config 1

# ── Stage-specific preflight (weekly-sf-policy.md §2.2) ──────────────────────
# PortfolioOptimizerBacktest runs after PredictorBacktest — it reads the
# predictor sweep artifacts that stage just wrote. Assert that upstream
# output exists BEFORE provisioning a spot.
preflight_portfolio_optimizer_backtest() {
    echo "==> Stage preflight: portfolio-optimizer-backtest"
    if ! spot_common_s3_prefix_nonempty "backtest/${RUN_DATE}/"; then
        echo "ERROR: s3://${S3_BUCKET}/backtest/${RUN_DATE}/ is empty or unreachable." >&2
        echo "       This stage reads the PredictorBacktest stage's predictor-sweep output from that prefix — it must run and succeed first." >&2
        echo "       Failing before spend rather than discovering this mid-run." >&2
        exit 1
    fi
    echo "  stage preflight OK (backtest/${RUN_DATE}/ has upstream PredictorBacktest output)."
}
preflight_portfolio_optimizer_backtest

echo "==> Dispatcher pre-launch preflight (fail-fast before provisioning spot)..."
spot_common_pre_launch_preflight \
    "$REPO_ROOT/backtest.py" \
    "$REPO_ROOT/preflight.py" \
    "$REPO_ROOT/pipeline_common.py" \
    "$REPO_ROOT/synthetic/predictor_backtest.py"

# This mode re-runs predictor_pipeline internally — apply the >=16 GB
# instance floor (I3280) unless the operator passed an explicit --instance-type.
spot_common_apply_predictor_ram_floor
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

# ── Stage: backtest (mode=portfolio-optimizer-backtest) ─────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  PORTFOLIO-OPTIMIZER-BACKTEST (--mode portfolio-optimizer-backtest)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_ssm "portfolio-optimizer" "$MAX_RUNTIME_SECONDS" <<BACKTEST
set -eo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

RUN_DATE="${RUN_DATE}"

echo "▶ stage=backtest START at \$(date -u +%H:%M:%S)"
if ! $REMOTE_PYTHON -u backtest.py --mode portfolio-optimizer-backtest --date "\${RUN_DATE}" --upload --log-level INFO $BACKTEST_SKIP_PHASE4_FLAG $BACKTEST_PHASE_FLAGS 2>&1; then
    echo "ERROR: backtest.py failed. Spot run marked FAILED — check flow-doctor alerts." >&2
    exit 1
fi
echo "▶ stage=backtest END at \$(date -u +%H:%M:%S)"

echo ""
echo "Portfolio-optimizer-backtest stage complete at \$(date)"
BACKTEST

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Portfolio-optimizer-backtest complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"

# No CloudWatch heartbeat emitted here — see spot_predictor_backtest.sh's
# identical comment. Tracked: alpha-engine-config-I6710.

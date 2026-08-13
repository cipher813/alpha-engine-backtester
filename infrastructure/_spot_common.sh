#!/usr/bin/env bash
# infrastructure/_spot_common.sh — shared spot-EC2 infrastructure for the
# per-stage backtester launchers (spot_backtester.sh, spot_predictor_backtest.sh,
# spot_portfolio_optimizer_backtest.sh, spot_parity.sh, spot_evaluator.sh).
#
# alpha-engine-config-I4442 (weekly-sf-policy.md §2.1 — atomicity): the
# Saturday SF's five backtest-family states are already independent SF
# states, but until this split they all called the SAME 1600+-line
# `spot_backtest.sh` monolith with different --mode/--skip-stages/
# --pit-parity-enabled flags. Bundling collapsed the blast radius and the
# diagnostic radius into one file: a syntax error or an unbound variable
# in one mode's code path could break every mode, and there was no way to
# preflight a single stage in isolation. This file extracts the parts that
# are genuinely IDENTICAL across all five stages — spot launch, bootstrap,
# dependency install, config staging, SSM transport, cleanup/relaunch,
# error-artifact publishing — so each stage script only has to carry its
# own stage-specific SSM payload and its own preflight.
#
# `spot_backtest.sh` itself is UNCHANGED and stays the currently-wired SF
# path; the SF cutover to these new scripts is a separate nousergon-data
# PR, sequenced after this one. Per policy-shared-code, this stays a
# repo-local sourced file (pure-Bash primitives may stay mirrored) rather
# than moving into nousergon-lib in this PR — nousergon-lib is the right
# home on SECOND adoption (crucible-data / crucible-predictor's spot
# launchers carry near-identical launch/bootstrap/SSM-transport blocks;
# lifting this into the lib is the natural next step once a second repo
# needs it, tracked as a follow-up rather than done speculatively here).
#
# MUST be sourced, never executed directly — it defines functions and sets
# defaults into the caller's global scope; it has no entry point of its own.
#
# shellcheck disable=SC2034
# This file sets many variables (SHOW_HELP, BACKTEST_SKIP_PHASE4_FLAG, etc.)
# that are only READ by the stage script that sources it — shellcheck
# analyzes this file in isolation and cannot see that cross-file usage.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: _spot_common.sh must be sourced, not executed directly." >&2
    echo "       e.g. source \"\$SCRIPT_DIR/_spot_common.sh\"" >&2
    exit 1
fi

# ── Unconditional global defaults ────────────────────────────────────────────
# Every var a downstream function reads is initialized HERE, unconditionally,
# regardless of which flags the caller later parses or which code path runs.
# Motivating defect (2026-07-25 weekly SF recovery, alpha-engine-config-I4442):
# a var initialized only inside one mode's heredoc branch was unbound under
# `set -u` in every OTHER mode — `_BACKTEST_WAS_SKIPPED` in the monolith hit
# exactly this class and was later fixed by moving its init ahead of
# `set -euo pipefail` (see spot_backtest.sh's own `export ... MUST precede
# set -euo pipefail` comment). Every stage-carried variable in this file and
# in each spot_*.sh caller is set here or at the top of the caller — never
# inside an `if`/`case`/heredoc branch on first use.
spot_common_init_defaults() {
    AWS_REGION="${AWS_REGION:-us-east-1}"
    S3_BUCKET="${S3_BUCKET:-alpha-engine-research}"
    BRANCH="${BRANCH:-main}"
    # Capacity-resilient instance-type fallback set (2026-05-22 incident:
    # the Evaluator invocation hit InsufficientInstanceCapacity for c5.large
    # in subnet-e07166ec / us-east-1f). All 2 vCPU / 4-8 GB RAM — equivalent
    # for the backtester (memory-bound).
    INSTANCE_TYPES="${INSTANCE_TYPES:-c5.large,m5.large,c6i.large,c5a.large}"
    INSTANCE_TYPE=""  # --instance-type X collapses INSTANCE_TYPES to single value
    AMI_ID="ami-0c421724a94bba6d6"      # Amazon Linux 2023 x86_64
    # Spot-side watchdog budget. Kept at the monolith's combined-run value
    # (7200s) for every split script in THIS PR — right-sizing each stage's
    # own budget to its own p95 (weekly-sf-policy.md §4) is deferred to the
    # SF-cutover PR, where the SF state's own executionTimeout is set
    # alongside it; until then this stays a conservative shared ceiling.
    MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-7200}"
    # KEY_NAME kept ONLY as a launch attribute — nothing in this script SSHs
    # in. Communication is via SSM (2026-05-27 SSH→SSM migration).
    KEY_NAME="alpha-engine-key"
    # alpha-engine-config#3018: SECURITY_GROUP / SUBNETS are EC2 *launch*
    # attributes, not an IAM policy surface — ec2:RunInstances on
    # alpha-engine-executor-role is granted with Resource:"*", so neither
    # value gates access or can "drift" against an IAM policy document.
    # Duplicated verbatim across the data/predictor/backtester spot
    # launchers (same default VPC vpc-566f002e, same SG) — accepted-by-design
    # as plain launch config, intentionally out of IAM-as-code scope.
    SECURITY_GROUP="sg-03cd3c4bd91e610b0"
    SUBNETS="${SUBNETS:-subnet-a61ec0fb,subnet-1e58307a,subnet-789d3857,subnet-c670118d,subnet-7cff7c43,subnet-e07166ec}"
    # IAM_PROFILE backs alpha-engine-executor-role, tracked+applied+drift-
    # checked from crucible-executor/infrastructure/iam/alpha-engine-executor-role/.
    IAM_PROFILE="alpha-engine-executor-profile"
    LIB_PYTHON="${LIB_PYTHON:-/home/ec2-user/alpha-engine-dashboard/.venv/bin/python}"

    # Common flag-parsed values — every script surfaces this same subset of
    # the monolith's flag surface (branch/instance-type/run-date/dry-run/
    # preflight-only/phase-control/freeze-evaluator). Stage-selecting flags
    # (--mode, --skip-stages, --pit-parity-enabled, --smoke-only) do NOT
    # exist on these scripts — each script IS one stage, hardcoded.
    SHOW_HELP=0
    PREFLIGHT_ONLY=0
    SKIP_PHASE4="${SKIP_PHASE4_EVALUATIONS:-false}"
    SKIP_PHASES="${SKIP_PHASES:-}"
    ONLY_PHASES="${ONLY_PHASES:-}"
    FORCE_ALL="${FORCE_ALL:-false}"
    FORCE_PHASES="${FORCE_PHASES:-}"
    DRY_RUN="${DRY_RUN:-false}"
    FREEZE_EVALUATOR="${FREEZE_EVALUATOR:-false}"
    USE_VECTORIZED_SWEEP="${USE_VECTORIZED_SWEEP:-false}"
    EVAL_HALF="${EVAL_HALF:-all}"
    RUN_DATE="${RUN_DATE:-$(date -u +%Y-%m-%d)}"

    # #883 bounded mid-run spot-reclaim relaunch — env-only, no CLI flag on
    # the monolith either. MAX_SPOT_ATTEMPTS=4 preserves the prior
    # RECLAIM_RELAUNCH_MAX=3 budget (3 relaunches = 4 total attempts).
    MAX_SPOT_ATTEMPTS="${MAX_SPOT_ATTEMPTS:-4}"
    SPOT_ATTEMPT="${SPOT_ATTEMPT:-1}"
    SF_EXECUTION_TIMEOUT="${SF_EXECUTION_TIMEOUT:-}"

    # Derived flag strings, built by spot_common_compute_phase_flags() after
    # parsing — initialized empty here so `set -u` never trips if a caller
    # (e.g. spot_parity.sh / spot_evaluator.sh) never calls that builder.
    BACKTEST_SKIP_PHASE4_FLAG=""
    BACKTEST_PHASE_FLAGS=""

    # Populated later in the launch sequence; initialized here so any early
    # exit path (trap cleanup) can reference them without tripping `set -u`.
    INSTANCE_ID=""
    RUN_ID=""
    S3_STAGING_PREFIX=""
    S3_STAGING=""
    LAST_SSM_DESC=""
    EXECUTOR_CONFIG=""
    PREDICTOR_CONFIG=""
    STAGED_PREDICTOR_CONFIG=0
    ENV_SOURCE=""
    REMOTE_PYTHON=""
}

# ── Flag parsing ──────────────────────────────────────────────────────────────
# Common subset of the monolith's flag surface. Unknown flags hard-fail
# (no-silent-fails) — same as the monolith. Callers pass "$@" through.
spot_common_parse_flags() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h) SHOW_HELP=1; shift ;;
            --preflight-only) PREFLIGHT_ONLY=1; shift ;;
            --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
            --instance-type=*) INSTANCE_TYPE="${1#*=}"; shift ;;
            --branch) BRANCH="$2"; shift 2 ;;
            --branch=*) BRANCH="${1#*=}"; shift ;;
            --run-date) RUN_DATE="$2"; shift 2 ;;
            --run-date=*) RUN_DATE="${1#*=}"; shift ;;
            --skip-phase4-evaluations) SKIP_PHASE4="true"; shift ;;
            --skip-phases) SKIP_PHASES="$2"; shift 2 ;;
            --skip-phases=*) SKIP_PHASES="${1#*=}"; shift ;;
            --only-phases) ONLY_PHASES="$2"; shift 2 ;;
            --only-phases=*) ONLY_PHASES="${1#*=}"; shift ;;
            --force) FORCE_ALL="true"; shift ;;
            --force-phases) FORCE_PHASES="$2"; shift 2 ;;
            --force-phases=*) FORCE_PHASES="${1#*=}"; shift ;;
            --dry-run) DRY_RUN="true"; shift ;;
            --freeze-evaluator) FREEZE_EVALUATOR="true"; shift ;;
            --use-vectorized-sweep) USE_VECTORIZED_SWEEP="true"; shift ;;
            --eval-half) EVAL_HALF="$2"; shift 2 ;;
            --eval-half=*) EVAL_HALF="${1#*=}"; shift ;;
            *)
                echo "Unknown flag: $1" >&2
                echo "(stage-selecting flags --mode / --skip-stages / --pit-parity-enabled / --smoke-only do not exist on this script — it IS one stage. Use spot_backtest.sh directly for those.)" >&2
                exit 1
                ;;
        esac
    done

    case "$EVAL_HALF" in
        all|diagnostics|optimize) ;;
        *)
            echo "ERROR: unknown --eval-half='$EVAL_HALF'" >&2
            echo "       Valid values: all diagnostics optimize" >&2
            exit 1
            ;;
    esac
}

# Builds BACKTEST_SKIP_PHASE4_FLAG / BACKTEST_PHASE_FLAGS from the parsed
# phase-control vars. Called by the three backtest-family stage scripts
# (spot_backtester.sh, spot_predictor_backtest.sh,
# spot_portfolio_optimizer_backtest.sh) only — parity/evaluator don't pass
# these through to backtest.py.
spot_common_compute_phase_flags() {
    if [ "$SKIP_PHASE4" = "true" ]; then
        BACKTEST_SKIP_PHASE4_FLAG="--skip-phase4-evaluations"
    else
        BACKTEST_SKIP_PHASE4_FLAG=""
    fi
    BACKTEST_PHASE_FLAGS=""
    if [ -n "$SKIP_PHASES" ]; then
        BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --skip-phases=$SKIP_PHASES"
    fi
    if [ -n "$ONLY_PHASES" ]; then
        BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --only-phases=$ONLY_PHASES"
    fi
    if [ "$FORCE_ALL" = "true" ]; then
        BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --force"
    fi
    if [ -n "$FORCE_PHASES" ]; then
        BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --force-phases=$FORCE_PHASES"
    fi
    if [ "$DRY_RUN" = "true" ]; then
        BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --dry-run"
    fi
    if [ "$USE_VECTORIZED_SWEEP" = "true" ]; then
        BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --use-vectorized-sweep"
    fi
}

# Stages that materialise the ~900-ticker universe need >=16 GB RAM (I3280:
# 8 GB instances can dip to ~6 GB available under OS overhead, leaving zero
# margin above the 6.0 GB headroom guard). Skipped when the operator passes an
# explicit --instance-type.
#
# RENAMED 2026-08-13 from `spot_common_apply_predictor_ram_floor`. The old name
# named the wrong thing, and the misnomer had already cost two production OOMs:
#
#   * spot_backtester.sh was left off the floor on the reasoning "param-sweep
#     does not run predictor_pipeline -> stays on the cheap default rotation".
#     OOM-killed 2026-08-13 (config-I7216, PR653/PR657).
#   * spot_evaluator.sh carried the SAME sentence and was OOM-killed the SAME
#     DAY on a 4 GB c5.large, immediately after
#     `Load complete: 921 price tickers, 903 feature tickers`
#     (execution watch-rerun-2026-08-13-2, instance i-077d2a5479affe1d3).
#
# The driver is the ArcticDB universe read, not the GBM tensor. Not loading the
# tensor does not make a stage cheap, and every launcher that reads the full
# universe needs this whether or not `predictor_pipeline` appears anywhere in
# it. The name now says the condition an author must actually check.
spot_common_apply_large_universe_ram_floor() {
    local floor_types="m5.xlarge,m6i.xlarge,m5a.xlarge,c5.2xlarge,c6i.2xlarge"
    if [ -z "$INSTANCE_TYPE" ]; then
        echo "  Stage reads the full ~900-ticker universe -> applying >=16 GB instance floor"
        INSTANCE_TYPES="$floor_types"
    fi
    if [ -n "$INSTANCE_TYPE" ]; then
        INSTANCE_TYPES="$INSTANCE_TYPE"
    fi
}

spot_common_collapse_instance_type() {
    if [ -n "$INSTANCE_TYPE" ]; then
        INSTANCE_TYPES="$INSTANCE_TYPE"
    fi
}

# ── DATE_CONVENTIONS: normalize RUN_DATE to the NYSE trading day ────────────
# Single dispatcher-side chokepoint, BEFORE RUN_DATE is threaded into the
# stage's --date and any bash s3 path. Defensive: keep the calendar value if
# the lib call fails — a normalization miss must not abort the launch; the
# python entry points re-normalize idempotently as a backstop.
spot_common_normalize_run_date() {
    local _run_date_td
    _run_date_td="$("$LIB_PYTHON" -c "import datetime as d; from nousergon_lib import trading_calendar as tc; x=d.date.fromisoformat('${RUN_DATE}'[:10]); print(x.isoformat() if tc.is_trading_day(x) else tc.previous_trading_day(x).isoformat())" 2>/dev/null || true)"
    if [ -n "$_run_date_td" ]; then
        if [ "$_run_date_td" != "$RUN_DATE" ]; then
            echo "==> Normalized RUN_DATE ${RUN_DATE} (calendar) -> ${_run_date_td} (trading day) per DATE_CONVENTIONS"
        fi
        RUN_DATE="$_run_date_td"
    else
        echo "WARNING: trading-day normalization of RUN_DATE=${RUN_DATE} failed — keeping calendar value (python entry points will re-normalize)" >&2
    fi
}

# ── Dispatcher-side pre-launch preflight (L4485) ─────────────────────────────
# Fail fast on the DISPATCHER, before provisioning a spot. Two cheap
# (<2s, no deps) checks that catch the two cheapest-to-miss classes at
# second zero: a SyntaxError in a load-bearing entrypoint, and a
# requirements.txt lib pin below preflight.py's own floor. Callers pass the
# list of entrypoints THEIR stage actually imports.
spot_common_pre_launch_preflight() {
    local py
    py="$LIB_PYTHON"
    [ -x "$py" ] || py="$(command -v python3 || echo python3)"

    if ! "$py" -c 'import ast,sys; [ast.parse(open(f).read(), filename=f) for f in sys.argv[1:]]' "$@" 2>/tmp/prelaunch_syntax.err; then
        echo "ERROR: pre-launch syntax check FAILED — a SyntaxError would crash the spot ~15 min into boot+deps. Fix before launching:" >&2
        cat /tmp/prelaunch_syntax.err >&2
        exit 1
    fi

    local pin floor lowest
    pin=$(grep -oE '@v[0-9]+\.[0-9]+\.[0-9]+' "$REPO_ROOT/requirements.txt" | head -1 | tr -d '@v')
    floor=$(grep -oE 'MIN_LIB_VERSION[[:space:]]*=[[:space:]]*"[0-9.]+"' "$REPO_ROOT/preflight.py" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if [ -n "$pin" ] && [ -n "$floor" ]; then
        lowest=$(printf '%s\n%s\n' "$floor" "$pin" | sort -V | head -1)
        if [ "$lowest" != "$floor" ]; then
            echo "ERROR: requirements.txt alpha-engine-lib pin v$pin < preflight.py MIN_LIB_VERSION $floor." >&2
            echo "       The spot's pip install would pull a version the code rejects. Bump the pin or the floor." >&2
            exit 1
        fi
        echo "  pre-launch: lib pin v$pin >= MIN_LIB_VERSION $floor OK"
    else
        echo "  pre-launch: WARNING — could not parse lib pin (pin='$pin' floor='$floor'); skipping pin cross-check" >&2
    fi

    local dirty
    dirty=$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | grep -E '\.(py|sh)$' || true)
    if [ -n "$dirty" ]; then
        echo "  pre-launch: WARNING — uncommitted tracked .py/.sh changes; the spot clones --branch $BRANCH and will NOT see these:" >&2
        echo "$dirty" | sed 's/^/      /' >&2
    fi
    git -C "$REPO_ROOT" fetch --quiet origin "$BRANCH" 2>/dev/null || true
    local lhead rhead
    lhead=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)
    rhead=$(git -C "$REPO_ROOT" rev-parse "origin/$BRANCH" 2>/dev/null || true)
    if [ -n "$lhead" ] && [ -n "$rhead" ] && ! git -C "$REPO_ROOT" merge-base --is-ancestor "$lhead" "$rhead" 2>/dev/null; then
        echo "  pre-launch: WARNING — local HEAD ($lhead) is not in origin/$BRANCH; the spot clones origin/$BRANCH and will run WITHOUT your local commits. Push first." >&2
    fi

    local cfg_real cfg_git_root cfg_rel cfg_dirty
    if [ -L "$REPO_ROOT/config.yaml" ]; then
        cfg_real=$(readlink -f "$REPO_ROOT/config.yaml" 2>/dev/null || true)
        if [ -n "$cfg_real" ]; then
            cfg_git_root=$(git -C "$(dirname "$cfg_real")" rev-parse --show-toplevel 2>/dev/null || true)
            if [ -z "$cfg_git_root" ]; then
                echo "  pre-launch: WARNING — config.yaml symlinks to $cfg_real, which is NOT inside a git repo; operator flags there have no audit trail and will vanish on rebuild." >&2
            else
                cfg_rel="${cfg_real#"$cfg_git_root"/}"
                if ! git -C "$cfg_git_root" ls-files --error-unmatch "$cfg_rel" >/dev/null 2>&1; then
                    echo "  pre-launch: WARNING — config.yaml symlinks to $cfg_real, which is NOT git-tracked in $cfg_git_root; operator flags there have no audit trail and will vanish on rebuild." >&2
                else
                    cfg_dirty=$(git -C "$cfg_git_root" status --porcelain -- "$cfg_rel" 2>/dev/null || true)
                    if [ -n "$cfg_dirty" ]; then
                        echo "  pre-launch: WARNING — config.yaml ($cfg_real) has uncommitted changes not captured in git ($cfg_git_root); these operator flags will NOT survive a rebuild of this symlink. Commit + PR the change:" >&2
                        echo "$cfg_dirty" | sed 's/^/      /' >&2
                    fi
                fi
            fi
        fi
    fi

    echo "  pre-launch preflight OK."
}

# ── Config resolution ────────────────────────────────────────────────────────
# Executor risk.yaml: needed by every stage (the executor sim/backtest path
# loads it at process startup regardless of --mode). Fails loud if missing —
# silently falling back to risk.yaml.example produces placeholder bucket
# names and an ArcticDB KeyNotFoundException deep in the run.
spot_common_resolve_executor_config() {
    local experiment_id="${ALPHA_ENGINE_EXPERIMENT_ID:-reference}"
    EXECUTOR_CONFIG=""
    local candidate
    for candidate in \
        "$HOME/alpha-engine-config/experiments/$experiment_id/executor/risk.yaml" \
        "$HOME/Development/alpha-engine-config/experiments/$experiment_id/executor/risk.yaml" \
        "$HOME/alpha-engine-config/executor/risk.yaml" \
        "$HOME/Development/alpha-engine-config/executor/risk.yaml" \
        "$HOME/alpha-engine/config/risk.yaml" \
        "$HOME/Development/alpha-engine/config/risk.yaml"; do
        if [ -f "$candidate" ]; then
            EXECUTOR_CONFIG="$candidate"
            break
        fi
    done
    if [ -z "$EXECUTOR_CONFIG" ]; then
        echo "ERROR: executor risk.yaml not found in any search path:" >&2
        echo "  ~/alpha-engine-config/experiments/$experiment_id/executor/risk.yaml" >&2
        echo "  ~/Development/alpha-engine-config/experiments/$experiment_id/executor/risk.yaml" >&2
        echo "  ~/alpha-engine-config/executor/risk.yaml" >&2
        echo "  ~/Development/alpha-engine-config/executor/risk.yaml" >&2
        echo "  ~/alpha-engine/config/risk.yaml (legacy)" >&2
        echo "  ~/Development/alpha-engine/config/risk.yaml (legacy)" >&2
        exit 1
    fi
}

# Predictor predictor.yaml. `required=1` (spot_predictor_backtest.sh /
# spot_portfolio_optimizer_backtest.sh — these stages run predictor_pipeline
# and cannot produce a real result without it) hard-fails when missing,
# per-stage preflight assertion rather than the monolith's uniform soft-skip
# (the monolith couldn't know per-invocation whether the mode needed it; each
# split script does). `required=0` (spot_backtester.sh / spot_parity.sh /
# spot_evaluator.sh) keeps the monolith's soft-skip-with-sentinel behavior.
spot_common_resolve_predictor_config() {
    local required="${1:-0}"
    PREDICTOR_CONFIG=""
    local candidate
    for candidate in \
        "$HOME/alpha-engine-predictor/config/predictor.yaml" \
        "$HOME/Development/alpha-engine-predictor/config/predictor.yaml"; do
        if [ -f "$candidate" ]; then
            PREDICTOR_CONFIG="$candidate"
            break
        fi
    done
    # alpha-engine-config-I7216: stage it from the private config repo before
    # declaring it missing. predictor.yaml is gitignored (the .example pattern),
    # so a fresh dispatcher only has it because SOMETHING copied it there — and
    # until now the only thing that did was ResearchPredictorParallel, three
    # `cp` invocations buried in the weekly SF's PredictorTraining branch.
    #
    # That made this stage's prerequisite a SIDE EFFECT of a different stage.
    # Measured 2026-08-13: a mechanical rerun (weekly_sf_rerun.py) correctly
    # derived skip_predictor_training, because that stage had already completed
    # — and PredictorBacktest then died here on a config the skipped stage
    # would have staged. The recovery path is exactly the path that skips
    # completed stages, so the dependency is invisible precisely when it bites.
    # PredictorBacktest writes the live entry feed
    # (predictor/research_free_backfill/), so this blocked the cohort refresh.
    #
    # Staging here rather than adding a fourth `cp` to the SF keeps the fix at
    # the layer that DECLARES the requirement: the stage that needs the file
    # obtains it, instead of every caller remembering to.
    if [ -z "$PREDICTOR_CONFIG" ]; then
        local _ref="$HOME/alpha-engine-config/experiments/reference/predictor/predictor.yaml"
        local _dest="$HOME/alpha-engine-predictor/config/predictor.yaml"
        if [ -f "$_ref" ] && [ -d "$(dirname "$_dest")" ]; then
            echo "  predictor.yaml absent — staging from alpha-engine-config reference (config-I7216)"
            # `rm -f` then plain `cp`, NOT `cp --remove-destination`: the SF's
            # own three copies use the GNU flag, which exists only to replace a
            # SYMLINKED destination and is unsupported by BSD cp. This form is
            # equivalent on Linux and also runs on a developer's machine, which
            # is what let this staging path be exercised before it shipped.
            if rm -f "$_dest" && cp "$_ref" "$_dest"; then
                PREDICTOR_CONFIG="$_dest"
            else
                echo "  WARNING: staging predictor.yaml from $_ref failed" >&2
            fi
        fi
    fi

    if [ -z "$PREDICTOR_CONFIG" ]; then
        if [ "$required" = "1" ]; then
            echo "ERROR: predictor.yaml not found in any search path — this stage runs predictor_pipeline and cannot produce a real result without it:" >&2
            echo "  ~/alpha-engine-predictor/config/predictor.yaml" >&2
            echo "  ~/Development/alpha-engine-predictor/config/predictor.yaml" >&2
            echo "  and staging from ~/alpha-engine-config/experiments/reference/predictor/predictor.yaml did not succeed" >&2
            exit 1
        fi
        echo "  WARNING: predictor.yaml not found — predictor backtest will be skipped"
        STAGED_PREDICTOR_CONFIG=0
    else
        STAGED_PREDICTOR_CONFIG=1
    fi
}

# ── Launch spot instance ──────────────────────────────────────────────────────
# Capacity-resilient launch via krepis.ec2_spot (rotates instance_type x
# subnet on InsufficientInstanceCapacity etc). Direct fix for the 2026-05-22
# incident (InsufficientInstanceCapacity for c5.large in us-east-1f).
spot_common_launch_instance() {
    echo "==> Requesting spot instance (lib CLI rotation: types=[$INSTANCE_TYPES], subnets=[$SUBNETS])..."
    local ec2_spot_rc=0
    INSTANCE_ID=$("$LIB_PYTHON" -m krepis.ec2_spot launch \
        --types "$INSTANCE_TYPES" \
        --subnets "$SUBNETS" \
        --image-id "$AMI_ID" \
        --key-name "$KEY_NAME" \
        --security-group "$SECURITY_GROUP" \
        --iam-profile "$IAM_PROFILE" \
        --name "alpha-engine-${SPOT_STAGE_NAME}-$(date +%Y%m%d)" \
        --region "$AWS_REGION") || ec2_spot_rc=$?
    if [ "$ec2_spot_rc" -ne 0 ] || [ -z "$INSTANCE_ID" ]; then
        if [ "$ec2_spot_rc" -eq 64 ]; then
            echo "ERROR: capacity exhausted across all instance_type x subnet combinations" >&2
        fi
        if [ "$ec2_spot_rc" -eq 0 ]; then
            # rc=0 with an EMPTY instance id = the launch layer produced
            # nothing. `${ec2_spot_rc:-1}` defaults only when UNSET — a
            # captured 0 passed through would be a silent success. An empty
            # id must always fail loud (config#1646).
            echo "ERROR: ec2_spot launch exited 0 without an instance id — failing loud (config#1646)" >&2
            ec2_spot_rc=1
        fi
        exit "$ec2_spot_rc"
    fi
    echo "  Instance ID: $INSTANCE_ID"

    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${INSTANCE_ID}"
    S3_STAGING_PREFIX="tmp/spot_${SPOT_STAGE_NAME}/${RUN_ID}"
    S3_STAGING="s3://${S3_BUCKET}/${S3_STAGING_PREFIX}"
}

# ── Cleanup / reclaim-relaunch / error-artifact publishing ──────────────────
# Always terminate the instance + clean S3 staging, with diagnostics on
# failure, and re-exec on a CONFIRMED spot reclaim (#883). Installs a
# `trap cleanup EXIT` — call once, after INSTANCE_ID is set.
spot_common_install_cleanup_trap() {
    # shellcheck disable=SC2329
    # invoked via `trap cleanup EXIT` below — shellcheck doesn't associate
    # a same-function trap registration as a call site.
    cleanup() {
        local exit_code=$?
        local _will_relaunch=0 _alert_sev="error"
        echo ""
        echo "==> Dispatcher EXIT (code=$exit_code)"
        local state="<not yet provisioned>" reason_code="<none>" state_reason="<none>"
        if [ "$exit_code" -ne 0 ]; then
            local last_desc="${LAST_SSM_DESC:-<none — failed before any SSM call>}"
            echo "    last run_ssm: $last_desc"
            if [ -n "${INSTANCE_ID:-}" ]; then
                local _desc
                _desc=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" --query 'Reservations[0].Instances[0].[State.Name,StateReason.Code,StateTransitionReason]' --output text 2>/dev/null || true)
                state=$(printf '%s' "$_desc" | cut -f1)
                reason_code=$(printf '%s' "$_desc" | cut -f2)
                state_reason=$(printf '%s' "$_desc" | cut -f3-)
                [ -z "$state" ] && state="<lookup-failed>"
                [ -z "$reason_code" ] && reason_code="<none>"
                [ -z "$state_reason" ] && state_reason="<none>"
                echo "    spot state: $state"
                echo "    spot state-reason-code: $reason_code"
                echo "    spot state-transition-reason: $state_reason"
            fi
            # See alpha-engine-config-I7009 — migrated off the exit-code contract to --json.
            if [ -n "${INSTANCE_ID:-}" ] && [ "$SPOT_ATTEMPT" -lt "$MAX_SPOT_ATTEMPTS" ]; then
                local _decide_json="" _decide_rc=0
                _decide_json="$("$LIB_PYTHON" -m krepis.ec2_spot relaunch-decision \
                    --instance-id "$INSTANCE_ID" \
                    --region "$AWS_REGION" \
                    --attempt "$SPOT_ATTEMPT" \
                    --max-attempts "$MAX_SPOT_ATTEMPTS" \
                    ${SF_EXECUTION_TIMEOUT:+--sf-execution-timeout "$SF_EXECUTION_TIMEOUT" --per-attempt-seconds "$MAX_RUNTIME_SECONDS"} \
                    --json \
                    2>/dev/null)" || _decide_rc=$?
                if [ "$_decide_rc" -ne 0 ]; then
                    echo "    spot relaunch-decision: CLI failed to answer (rc=$_decide_rc) — treating as hold" >&2
                else
                    local _relaunch=""
                    _relaunch="$(printf '%s' "$_decide_json" | "$LIB_PYTHON" -c 'import json,sys; print("1" if json.load(sys.stdin).get("relaunch") else "0")')"
                    echo "    spot relaunch-decision (attempt $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS): $_decide_json"
                    if [ "$_relaunch" = "1" ]; then
                        _will_relaunch=1
                        _alert_sev="warning"
                    fi
                fi
            fi
            # Independent-channel surveillance (SNS + flow-doctor forum
            # topics). Best-effort: keeps cleanup running even if Python /
            # lib / SNS / flow-doctor are unreachable — stdout diagnostic
            # above is primary. This IS the "error-artifact publishing"
            # shared infra named in alpha-engine-config-I4442.
            local _alert_python _alert_msg
            _alert_msg="exit_code=$exit_code last_run_ssm='$last_desc' spot_state=$state spot_reason_code='$reason_code' spot_transition_reason='$state_reason' instance_id=${INSTANCE_ID:-<none>} will_relaunch=$_will_relaunch"
            if [ -x "$(dirname "$0")/../.venv/bin/python" ]; then
                _alert_python="$(dirname "$0")/../.venv/bin/python"
            else
                _alert_python="$(command -v python3 || command -v python || echo python)"
            fi
            (cd "$REPO_ROOT" && "$_alert_python" -c "
import sys
from ops_alerts import publish_ops_alert
publish_ops_alert(
    sys.argv[1],
    severity=sys.argv[2],
    source='alpha-engine-backtester/${SPOT_STAGE_NAME}',
)
" "$_alert_msg" "$_alert_sev") \
                > /dev/null 2>&1 || echo "    (ops alert fan-out failed; primary stdout diagnostic above is the surface)"
        fi
        echo "==> Terminating spot instance $INSTANCE_ID..."
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" --output text > /dev/null 2>&1 || true
        aws s3 rm "$S3_STAGING" --recursive --quiet 2>/dev/null || true
        echo "  Instance terminated; S3 staging cleaned."
        if [ "$_will_relaunch" = "1" ]; then
            echo "==> Spot RECLAIMED by AWS (reason_code='$reason_code' state='$state' transition='$state_reason') — relaunching on a fresh spot (attempt $((SPOT_ATTEMPT + 1))/$MAX_SPOT_ATTEMPTS)"
            trap - EXIT
            SPOT_ATTEMPT=$((SPOT_ATTEMPT + 1)) exec bash "$0" ${_ORIG_ARGS[@]+"${_ORIG_ARGS[@]}"}
        fi
        # CRITICAL: re-exit with the captured status — a bash EXIT trap that
        # ends on a successful command (the `|| true` cleanup steps) would
        # otherwise leave the script exiting 0, masking a Failed SSM step as
        # success to the orchestration wrapper.
        exit "$exit_code"
    }
    trap cleanup EXIT
}

# ── Stage config staging ─────────────────────────────────────────────────────
spot_common_stage_configs() {
    echo "==> Staging configs to ${S3_STAGING}/"
    aws s3 cp "$REPO_ROOT/config.yaml" "${S3_STAGING}/config.yaml" --region "$AWS_REGION" --quiet
    echo "  staged config.yaml"
    aws s3 cp "$EXECUTOR_CONFIG" "${S3_STAGING}/risk.yaml" --region "$AWS_REGION" --quiet
    echo "  staged risk.yaml from $EXECUTOR_CONFIG"
    if [ -n "$PREDICTOR_CONFIG" ]; then
        aws s3 cp "$PREDICTOR_CONFIG" "${S3_STAGING}/predictor.yaml" --region "$AWS_REGION" --quiet
        echo "  staged predictor.yaml from $PREDICTOR_CONFIG"
    fi
}

# ── Wait for the SSM agent to register ───────────────────────────────────────
spot_common_wait_for_ssm_agent() {
    echo "==> Waiting for SSM agent to come Online..."
    local i ping
    for i in $(seq 1 36); do  # 36 x 5s = 180s budget
        ping=$(aws ssm describe-instance-information \
            --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
            --query 'InstanceInformationList[0].PingStatus' \
            --output text --region "$AWS_REGION" 2>/dev/null || true)
        if [ "$ping" = "Online" ]; then
            echo "  SSM agent Online."
            return 0
        fi
        if [ "$i" -eq 36 ]; then
            echo "ERROR: SSM agent not Online after 180s (instance $INSTANCE_ID)"
            exit 1
        fi
        sleep 5
    done
}

# ── SSM dispatch primitive (lib chokepoint) ──────────────────────────────────
# run_ssm "<description>" [timeout_seconds] <<HEREDOC ... HEREDOC
# --diagnostics-bucket/--diagnostics-prefix activate the lib's failure-record
# writer: s3://${S3_BUCKET}/_spot_diagnostics/ae-${SPOT_STAGE_NAME}/{date}.json
# on terminal non-Success. Best-effort inside the lib; inner SSM exit always
# preserved.
run_ssm() {
    local description="$1" timeout_s="${2:-3600}"
    LAST_SSM_DESC="$description"
    "$LIB_PYTHON" -m krepis.ssm_dispatcher run \
        --instance-id "$INSTANCE_ID" \
        --description "${SPOT_STAGE_NAME}: $description" \
        --timeout "$timeout_s" \
        --output-bucket "$S3_BUCKET" \
        --output-key-prefix "${S3_STAGING_PREFIX}/ssm-output" \
        --region "$AWS_REGION" \
        --diagnostics-bucket "$S3_BUCKET" \
        --diagnostics-prefix "_spot_diagnostics/ae-${SPOT_STAGE_NAME}" \
        --script-stdin
}

# ── Bootstrap spot: watchdog + python + git + clone + fetch configs ─────────
spot_common_bootstrap() {
    echo "==> Bootstrapping spot (watchdog, python, clone, configs)..."
    run_ssm "bootstrap" 600 <<BOOTSTRAP
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=${AWS_REGION} AWS_DEFAULT_REGION=${AWS_REGION}

systemd-run --on-active=${MAX_RUNTIME_SECONDS} --unit=alpha-engine-watchdog \
    --description='alpha-engine spot hard-timeout' /sbin/shutdown -h now

dnf install -y -q python3.12 python3.12-pip python3.12-devel git gcc 2>/dev/null || \
    dnf install -y -q python3 python3-pip python3-devel git gcc
command -v python3.12 >/dev/null && PYTHON_BIN=python3.12 || PYTHON_BIN=python3
echo "Using: \$(\$PYTHON_BIN --version)"

git clone --depth 1 --branch ${BRANCH} https://github.com/nousergon/crucible-backtester.git /home/ec2-user/alpha-engine-backtester
git clone --depth 1 --branch ${BRANCH} https://github.com/nousergon/crucible-executor.git /home/ec2-user/alpha-engine
git clone --depth 1 --branch ${BRANCH} https://github.com/nousergon/crucible-predictor.git /home/ec2-user/alpha-engine-predictor

aws s3 cp ${S3_STAGING}/config.yaml /home/ec2-user/alpha-engine-backtester/config.yaml --region ${AWS_REGION} --quiet
echo "Fetched config.yaml"

mkdir -p /home/ec2-user/alpha-engine/config
aws s3 cp ${S3_STAGING}/risk.yaml /home/ec2-user/alpha-engine/config/risk.yaml --region ${AWS_REGION} --quiet
echo "Fetched risk.yaml"

if [ "${STAGED_PREDICTOR_CONFIG}" = "1" ]; then
    mkdir -p /home/ec2-user/alpha-engine-predictor/config
    aws s3 cp ${S3_STAGING}/predictor.yaml /home/ec2-user/alpha-engine-predictor/config/predictor.yaml --region ${AWS_REGION} --quiet
    echo "Fetched predictor.yaml"
else
    echo "predictor.yaml NOT staged (predictor backtest will be skipped)"
fi

echo "Bootstrap complete: 3 repos cloned, 3-4 configs fetched from ${S3_STAGING}."
BOOTSTRAP
}

# ── Install python dependencies ──────────────────────────────────────────────
spot_common_install_deps() {
    echo "==> Installing Python dependencies..."
    run_ssm "deps" 1200 <<DEPS
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=${AWS_REGION} AWS_DEFAULT_REGION=${AWS_REGION}
cd /home/ec2-user/alpha-engine-backtester

command -v python3.12 >/dev/null && PIP="python3.12 -m pip" || PIP="python3 -m pip"

\$PIP install --upgrade pip -q
\$PIP install -q -r requirements.txt

# The predictor checkout is CODE-ONLY (config#3031): its requirements.txt is
# deliberately NOT installed here — co-installing two repos' requirements
# into one resolver namespace let predictor's numpy floor silently override
# the backtester's numpy cap (numba/vectorbt hard ceiling). Every library the
# in-process predictor replay needs at runtime is declared in the
# backtester's OWN requirements.txt.
cd /home/ec2-user/alpha-engine-predictor
if [ ! -d "/home/ec2-user/alpha-engine-predictor/model" ]; then
    echo "FATAL: predictor checkout missing (code-only sys.path dependency)" >&2
    exit 1
fi

cd /home/ec2-user/alpha-engine-backtester
PYBIN="\${PIP% -m pip}"
\$PYBIN -c "import nousergon_lib.quant.stats.multiple_testing, nousergon_lib.quant" || {
    echo "FATAL: nousergon-lib is missing quant.stats — a co-installed sibling repo's pin likely downgraded it below v0.49.0. Resolved version:" >&2
    \$PIP show nousergon-lib | grep -E '^Version:' >&2 || true
    exit 1
}
\$PIP show nousergon-lib | grep -E '^Version:'

\$PYBIN -c "from synthetic.predictor_backtest import run; from synthetic.production_signal_backtest import build_production_signal_inputs" || {
    echo "FATAL: predictor modules missing or failed to import" >&2
    exit 1
}

# Fail-loud numpy-2 consistency guard (config#2815). Asserts the exact
# import chains that have broken production runs before, so any future
# co-installed pin that downgrades numpy breaks LOUD here at deps time
# (seconds) instead of deep into the run.
\$PYBIN -c "import numpy, scipy.sparse, lightgbm, numba, vectorbt; assert int(numpy.__version__.split('.')[0]) >= 2, 'numpy '+numpy.__version__+' < 2.0 is inconsistent with the numpy-2-built scipy/cvxpy stack (config#2815)'; print('numpy-2 guard OK: numpy='+numpy.__version__+' scipy='+scipy.__version__+' lightgbm='+lightgbm.__version__+' numba='+numba.__version__+' vectorbt='+vectorbt.__version__)" || {
    echo "FATAL: import-chain consistency check failed — a co-installed pin, stale downgrade, or numba/numpy ABI mismatch broke the scipy/lightgbm or numba/vectorbt import chain (config#2815, config-I3279). See traceback above." >&2
    exit 1
}

# Fail-loud pip-check dependency-consistency gate (config#2973). Gate on
# pip check's EXIT CODE, not on output emptiness — a clean env prints
# non-empty "No broken requirements found." and exits 0.
PIP_CHECK_ALLOWLIST=""
PIP_CHECK_RC=0
PIP_CHECK_OUT=\$(\$PYBIN -m pip check 2>&1) || PIP_CHECK_RC=\$?
if [ "\$PIP_CHECK_RC" -eq 0 ]; then
    echo "pip check: clean (exit 0)."
else
    if [ -z "\$PIP_CHECK_ALLOWLIST" ]; then
        PIP_CHECK_REMAIN="\$PIP_CHECK_OUT"
    else
        PIP_CHECK_REMAIN=\$(printf '%s\n' "\$PIP_CHECK_OUT" | grep -vFf <(printf '%s\n' "\$PIP_CHECK_ALLOWLIST") || true)
    fi
    if [ -n "\$PIP_CHECK_REMAIN" ]; then
        echo "FATAL: pip check reported non-allowlisted dependency conflicts:" >&2
        printf '%s\n' "\$PIP_CHECK_REMAIN" >&2
        exit 1
    fi
    echo "pip check: all reported conflicts are allowlisted."
fi

echo "Dependencies installed."
DEPS
}

# ── Predictor sector_map cache fetch ─────────────────────────────────────────
spot_common_fetch_predictor_cache() {
    echo "==> Downloading predictor sector_map from S3..."
    run_ssm "predictor-cache" 300 <<'CACHE'
set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp
CACHE_DIR="/home/ec2-user/alpha-engine-predictor/data/cache"
mkdir -p "$CACHE_DIR"
aws s3 cp s3://alpha-engine-research/reference/price_cache/sector_map.json "$CACHE_DIR/sector_map.json" 2>/dev/null \
    || aws s3 cp s3://alpha-engine-research/predictor/price_cache/sector_map.json "$CACHE_DIR/sector_map.json" 2>/dev/null \
    || true
echo "Predictor cache dir: sector_map.json $([ -f "$CACHE_DIR/sector_map.json" ] && echo present || echo MISSING)"
CACHE
}

# ── Build env export command ─────────────────────────────────────────────────
# PYTHONUNBUFFERED=1: line-buffering so SSM ships log lines as emitted.
# ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS=true: the sim hot loop would
# otherwise emit ~50k-200k per-decision S3 PUTs and blow the watchdog.
spot_common_build_env_source() {
    ENV_SOURCE='export XDG_CACHE_HOME=/tmp; export PYTHONUNBUFFERED=1; export ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS=true; export AWS_REGION=us-east-1; export AWS_DEFAULT_REGION=us-east-1 ALPHA_ENGINE_DEPLOYED=1; command -v python3.12 >/dev/null && PYTHON_BIN=python3.12 || PYTHON_BIN=python3; export PYTHON_BIN;'
    # shellcheck disable=SC2016
    # Deliberately single-quoted: this is a literal '$PYTHON_BIN' TOKEN
    # interpolated into each stage heredoc, resolved by the REMOTE spot's
    # own shell (via the PYTHON_BIN export in ENV_SOURCE above) — not by
    # this dispatcher.
    REMOTE_PYTHON='$PYTHON_BIN'
}

# ── Preflight-only (Friday shell_run dry path) ──────────────────────────────
# Runs ONLY backtest.py --mode=smoke (BacktesterPreflight + _runtime_smoke —
# lib-pin / imports / predictor-weights / universe-freshness, ~30-60s,
# read-only), then exit 0 — before ANY stage-specific spend. Returns 0 (does
# NOT exit) when PREFLIGHT_ONLY=0 so the caller continues into its own stage.
spot_common_run_preflight_only_and_maybe_exit() {
    if [ "$PREFLIGHT_ONLY" != "1" ]; then
        return 0
    fi
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  PREFLIGHT-ONLY (Friday shell_run dry path)"
    echo "  boot + deps done; running bootstrap-class smoke harness only,"
    echo "  then exit 0 — NO stage workload, ZERO external API calls,"
    echo "  ZERO S3/config writes."
    echo "═══════════════════════════════════════════════════════════════"
    run_ssm "preflight-only" 900 <<PREFLIGHT
set -eo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

echo "==> Preflight: backtest.py --mode=smoke"
$REMOTE_PYTHON -u backtest.py --mode=smoke --log-level INFO 2>&1
PREFLIGHT

    echo ""
    echo "==> Preflight-only PASSED — bootstrap-class smoke clean."
    echo "==> Instance will be terminated (no stage workload, no S3/config writes performed)."
    exit 0
}

# ── Per-stage CloudWatch heartbeat ───────────────────────────────────────────
spot_common_emit_heartbeat() {
    local _process="$1"
    aws cloudwatch put-metric-data \
        --namespace "AlphaEngine" \
        --metric-name "Heartbeat" \
        --dimensions "Process=${_process}" \
        --value 1 --unit "Count" \
        --region "${AWS_REGION:-us-east-1}" 2>/dev/null \
        && echo "Heartbeat emitted: ${_process}" \
        || echo "WARNING: Failed to emit heartbeat for ${_process} (non-fatal)"
}

# ── Cheap dispatcher-side S3 substrate checks (weekly-sf-policy.md §2.2) ────
# `aws s3 ls` on a single key/prefix — no spend, no spot launch. Each stage
# script's own preflight_<stage>() function calls these to assert its
# specific upstream artifact exists BEFORE provisioning a spot, rather than
# discovering the miss 15-40 min into boot+deps (the exact 2026-07-25
# Parity incident this issue is named for — parity failed after 37 minutes
# because its code path wasn't exercised until late in the monolith).
spot_common_s3_key_exists() {
    aws s3 ls "s3://${S3_BUCKET}/$1" --region "$AWS_REGION" >/dev/null 2>&1
}

spot_common_s3_prefix_nonempty() {
    local out
    out=$(aws s3 ls "s3://${S3_BUCKET}/$1" --region "$AWS_REGION" 2>/dev/null || true)
    [ -n "$out" ]
}

"""
Backtester preflight: connectivity + freshness checks run at the top of
each entrypoint before any real work starts.

Primitives live in ``alpha_engine_lib.preflight.BasePreflight``; this
module only composes them into a mode-specific sequence. See the
alpha-engine-lib README for the 2026-04-14 data-path failure mode that
motivated the library.

Modes:

- ``"backtest"`` — ``backtest.py`` entrypoint (weekly spot instance).
  Verifies that every module the full backtest call chain will import
  is actually importable now (imports, lib version, predictor weights)
  + S3 bucket reachable + ArcticDB ``macro/SPY`` fresh + the executor
  risk.yaml the simulate path will import resolves to a real config
  (not the placeholder .example template). 8-day threshold covers
  Fri→Mon weekly cadence + buffer.

  2026-04-21 incident motivated the import/version/weights additions:
  a Saturday SF dry-run burned ~80 minutes of c5.large compute before
  failing on ``No module named 'alpha_engine_lib.arcticdb'`` deep in
  ``_run_simulation_loop``. The three new preflight checks below all
  run in <2 seconds and would have caught the same bug at startup.
- ``"evaluate"`` — ``evaluate.py`` entrypoint. Reads simulation
  artifacts from S3 only, no ArcticDB. Keep the check cheap.
- ``"lambda_health"`` — daily predictor health check Lambda. Reads
  ``research.db`` + per-day metrics from S3. No ArcticDB.
"""

from __future__ import annotations

import importlib
import os

import yaml

from alpha_engine_lib.preflight import BasePreflight


# Placeholder prefix convention used by every repo's *.yaml.example
# template. A bucket/path value starting with this is definitionally a
# not-filled-in config and must never reach a live S3/ArcticDB read.
_PLACEHOLDER_PREFIX = "your-"

# Minimum alpha-engine-lib version the backtester depends on at runtime.
# Keep in sync with the ``@vX.Y.Z`` pin in ``requirements.txt``. Bump when
# a new symbol from the lib is imported by any backtest call path.
#
# Current floor: 0.1.4 — introduces ``alpha_engine_lib.arcticdb`` which
# ``backtest._run_simulation_loop`` depends on to filter historical
# signals against the current universe.
MIN_LIB_VERSION = "0.1.4"

# Modules whose imports are load-bearing for the backtest modes. Any
# missing or non-importable entry here would surface deep in the call
# chain; listing them explicitly here makes the failure show up in
# seconds at preflight instead of ~80 minutes into a spot run.
_CRITICAL_IMPORTS_BACKTEST = (
    # alpha-engine-lib submodules we directly call
    "alpha_engine_lib.arcticdb",
    "alpha_engine_lib.logging",
    "alpha_engine_lib.preflight",
    # executor modules — simulate path
    "executor.main",
    "executor.ibkr",
    # synthetic — predictor-backtest mode (10y GBM replay)
    "synthetic.predictor_backtest",
    # predictor model — loaded inside download_gbm_model
    "model.gbm_scorer",
)

# S3 keys the backtester's predictor-backtest mode HEADs before spending
# 10y × ~900 tickers on GBM inference. Missing means PredictorTraining
# has not populated the Layer-1A weights — investigate there.
_REQUIRED_PREDICTOR_WEIGHTS = (
    "predictor/weights/meta/momentum_model.txt",
    "predictor/weights/meta/momentum_model.txt.meta.json",
)

# Backtester-local modules that must be pre-imported before sibling repo
# paths land on sys.path. Rationale: the predictor repo ships its own
# ``store.arctic_reader`` with a different API; once predictor_path is
# at sys.path[0], ``from store.arctic_reader import load_universe_from_arctic``
# (in backtester's ``synthetic/predictor_backtest.py``) resolves to the
# wrong module and ImportErrors. In production these modules load via
# backtester's top-level imports BEFORE predictor_path is inserted;
# preflight runs before those top-level imports so we eagerly load
# them here to match the production sys.modules cache ordering.
_LOCAL_PREIMPORTS_BACKTEST = (
    "store.arctic_reader",
)

# Per-ticker ArcticDB freshness threshold (calendar days). The executor's
# hard-fail ATR/VWAP guards require last_date >= run_date - 1 trading day;
# 5 calendar days covers a long weekend + buffer for holidays. Tickers
# stale beyond this fail preflight rather than burning 2 hours of spot
# compute in predictor-backtest mode's inner loop.
_UNIVERSE_MAX_STALE_DAYS = 5

# Universe-freshness check parallelism — trades a small bump in per-ticker
# memory for wall-clock time. 20 threads → ~10 sec for 900 tickers on a
# c5.large. ArcticDB's internal I/O is thread-safe; we only read the
# ``.index[-1]`` value so even with 900 symbols the total data touched
# is tiny.
_UNIVERSE_FRESHNESS_THREADS = 20


class BacktesterPreflight(BasePreflight):
    """Preflight checks for the three backtester entrypoints."""

    def __init__(
        self,
        bucket: str,
        mode: str,
        executor_paths: list[str] | None = None,
        predictor_paths: list[str] | None = None,
    ):
        super().__init__(bucket)
        if mode not in ("backtest", "evaluate", "lambda_health"):
            raise ValueError(f"BacktesterPreflight: unknown mode {mode!r}")
        self.mode = mode
        # backtest.py passes config["executor_paths"] + config["predictor_paths"]
        # here so preflight can (a) validate executor's risk.yaml will load with
        # real values, and (b) add both repo roots to sys.path before
        # _check_imports so ``from executor.main`` / ``from model.gbm_scorer``
        # resolve the same way they do in ``_setup_simulation`` later. Without
        # the sys.path inserts, _check_imports fires ModuleNotFoundError on
        # ``executor``/``model`` even when everything is set up correctly.
        self.executor_paths = executor_paths or []
        self.predictor_paths = predictor_paths or []

    def run(self) -> None:
        self.check_env_vars("AWS_REGION")
        self.check_s3_bucket()

        if self.mode == "backtest":
            # Environment checks first — cheapest, and catch the class
            # of failure where the spot's pip install didn't pull the
            # pin we expected. ~2 seconds total. All three would have
            # caught the 2026-04-21 80-minute burn.
            self._check_lib_version()
            self._check_imports()
            self._check_predictor_weights()
            # synthetic/predictor_backtest.py reads from ArcticDB. SPY
            # lives in the ``macro`` library (market-wide series); its
            # freshness is a sufficient signal that the ArcticDB write
            # path upstream (alpha-engine-data DailyData) is healthy.
            # 8-day threshold: weekly cadence + 1 day buffer.
            self.check_arcticdb_fresh("macro", "SPY", max_stale_days=8)
            # Per-ticker freshness scan across the universe library —
            # macro.SPY can be fresh while individual tickers (ASGN,
            # MOH 2026-04-21) silently stop receiving daily_append writes.
            # Executor's load_atr_14_pct / load_daily_vwap guards fire
            # two hours deep into predictor-backtest mode when that
            # happens. Scanning here catches it at preflight in ~10s.
            self._check_universe_freshness(max_stale_days=_UNIVERSE_MAX_STALE_DAYS)
            # backtest.py's simulate path imports executor.main, which
            # in turn imports executor.config_loader and reads the
            # executor's risk.yaml. If it resolves to the placeholder
            # .example template (or to a file with "your-*" bucket
            # names), every downstream S3/ArcticDB read fails deep in
            # the executor-sim call chain. Caught at preflight so the
            # operator sees the real cause in <1s. Hit 2026-04-20.
            self._check_executor_config()

    # ── Environment primitives (added 2026-04-21 post-80min-burn) ────────

    def _check_lib_version(self) -> None:
        """Fail if the installed alpha_engine_lib is older than the
        minimum the backtester's call chain needs.

        Triggers when the spot's pip install silently fell back to a
        cached older version (or when requirements.txt was bumped but
        MIN_LIB_VERSION here wasn't — same bug in the other direction).
        """
        import alpha_engine_lib
        from packaging.version import Version

        installed = getattr(alpha_engine_lib, "__version__", None)
        if not installed:
            raise RuntimeError(
                "Pre-flight: alpha_engine_lib has no __version__ "
                "attribute — likely a broken install. Re-run the pip "
                "install step on this host."
            )
        if Version(installed) < Version(MIN_LIB_VERSION):
            raise RuntimeError(
                f"Pre-flight: alpha_engine_lib {installed} < required "
                f"{MIN_LIB_VERSION}. Spot's pip install may have pulled "
                "a stale cached version, or requirements.txt drifted "
                "from MIN_LIB_VERSION in preflight.py. The 2026-04-21 "
                "Saturday SF dry-run burned ~80 min on this exact class "
                "of failure before surfacing a deep-call-chain import "
                "error — this check catches it at startup."
            )

    def _check_imports(self) -> None:
        """Actually import every module the deep call chain relies on.

        A Python ImportError from inside ``_run_simulation_loop`` (or
        any of the other deep-call-stack sites) takes minutes-to-hours
        of spot time to surface because nothing before that point tries
        to import the module. Surfacing it at preflight is worth the
        ~1 second of extra import cost at startup.

        ``executor.main`` / ``executor.ibkr`` / ``model.gbm_scorer`` /
        ``synthetic.predictor_backtest`` are only importable once the
        alpha-engine + alpha-engine-predictor repo roots are on
        ``sys.path`` — normally done inside ``backtest._setup_simulation``.
        Preflight does the same inserts first so the import check
        matches production import resolution.

        **Sibling-repo collision defense:** the backtester and predictor
        both ship a ``store/arctic_reader.py`` with different APIs (the
        backtester's has ``load_universe_from_arctic``; the predictor's
        does not). Once predictor_path lands at sys.path[0], Python
        resolves ``store.arctic_reader`` to the predictor version and
        ``synthetic.predictor_backtest``'s top-level import of
        ``load_universe_from_arctic`` fails. In production this
        doesn't bite because backtester's top-level modules have
        already loaded (and cached in sys.modules) before predictor_path
        is prepended. Preflight runs before any top-level backtester
        imports, so we eagerly pre-load the local ``store`` modules
        here to match that production ordering.

        Mode-specific list: only ``backtest`` mode imports the executor
        and predictor repos. ``evaluate`` / ``lambda_health`` have
        their own narrower call chains (validated by their own preflight
        branches as needed).
        """
        import sys

        # Pre-import backtester-local modules that have same-name
        # siblings in executor/predictor repos. Cache wins regardless
        # of subsequent sys.path insert order.
        for local in _LOCAL_PREIMPORTS_BACKTEST:
            try:
                importlib.import_module(local)
            except ImportError as exc:
                raise RuntimeError(
                    f"Pre-flight: could not import local module "
                    f"{local!r} — backtester's own code is broken. "
                    f"Underlying error: {exc}"
                ) from exc

        for candidates in (self.executor_paths, self.predictor_paths):
            for p in candidates:
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)
                    break  # first hit wins, matches backtest.py behavior

        for name in _CRITICAL_IMPORTS_BACKTEST:
            try:
                importlib.import_module(name)
            except ImportError as exc:
                raise RuntimeError(
                    f"Pre-flight: could not import {name!r} — would "
                    "have crashed deep in the backtest call chain. "
                    "Check requirements.txt pin + that pip install "
                    "completed successfully on this host. If "
                    "executor/predictor imports fail, check config.yaml "
                    "``executor_paths`` / ``predictor_paths`` resolve "
                    f"to real directories on this host. Underlying "
                    f"error: {exc}"
                ) from exc

    def _check_predictor_weights(self) -> None:
        """S3 HEAD on the Layer-1A momentum GBM weights + metadata.

        ``synthetic/predictor_backtest.py::download_gbm_model`` reads
        these keys near the start of predictor-backtest mode. If they
        don't exist, fail now (seconds) instead of after the
        universe-data load (minutes) and report the named upstream
        owner in the error.
        """
        import boto3
        s3 = boto3.client("s3", region_name=self.region)
        for key in _REQUIRED_PREDICTOR_WEIGHTS:
            try:
                s3.head_object(Bucket=self.bucket, Key=key)
            except Exception as exc:
                raise RuntimeError(
                    f"Pre-flight: required key s3://{self.bucket}/{key} "
                    "is missing or unreadable. The Layer-1A momentum "
                    "GBM backtest requires this file; Saturday SF's "
                    "PredictorTraining step must populate "
                    "predictor/weights/meta/momentum_model.txt every "
                    f"run — investigate there. Underlying error: {exc}"
                ) from exc

    def _check_universe_freshness(self, max_stale_days: int) -> None:
        """Scan every ticker in the ArcticDB ``universe`` library; hard-fail
        if any ticker's last_date is older than ``max_stale_days`` calendar
        days from today.

        Motivation (2026-04-21 ~17:19 PT dry-run): backtester main completed
        with real portfolio stats + param-sweep results, then ``predictor-
        backtest`` mode's executor-simulate loop hit
        ``load_atr_14_pct failed validation — stale tickers ['ASGN (2026-04-01)',
        'MOH (2026-04-01)']`` ~2 hours in. Same class as the SNDK incident
        earlier that day: individual tickers stop getting daily_append
        writes while macro.SPY stays current, so
        ``check_arcticdb_fresh("macro", "SPY", ...)`` reports healthy.

        Runs the scan concurrently — 20 threads × ~900 tickers × one
        ``tail(1)`` read each ≈ 5-10 sec on c5.large. Caller pays that
        cost once at preflight, avoids burning 2 hours of spot compute
        before the same condition surfaces deep in the call chain.

        Raises ``RuntimeError`` with the stale ticker list + last_dates
        so the operator sees exactly which upstream writes failed.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from datetime import date, timedelta

        try:
            from alpha_engine_lib.arcticdb import open_universe_lib
        except ImportError as exc:
            raise RuntimeError(
                f"Pre-flight: alpha_engine_lib.arcticdb not importable for "
                f"universe-freshness scan: {exc}"
            ) from exc

        lib = open_universe_lib(self.bucket, region=self.region)
        symbols = list(lib.list_symbols())
        if not symbols:
            raise RuntimeError(
                f"Pre-flight: ArcticDB universe on bucket {self.bucket!r} "
                "has zero symbols — data pipeline upstream has not written "
                "anything. Investigate alpha-engine-data DataPhase1."
            )

        today = date.today()
        cutoff = today - timedelta(days=max_stale_days)

        def _last_date_for(sym: str) -> tuple[str, date | None, str | None]:
            """Return (symbol, last_date, error_msg)."""
            try:
                # tail(1) reads just the last row — avoids pulling 10y of
                # features per ticker for what is ultimately a one-datetime
                # freshness check. ~20ms per symbol typical.
                df = lib.tail(sym, n=1).data
                if df.empty:
                    return sym, None, "empty frame"
                last_ts = df.index[-1]
                # Normalize timezone / strip time component for clean
                # date comparison — matches check_arcticdb_fresh semantics.
                import pandas as pd
                last_date = pd.Timestamp(last_ts)
                if last_date.tzinfo is not None:
                    last_date = last_date.tz_convert("UTC").tz_localize(None)
                return sym, last_date.date(), None
            except Exception as exc:
                return sym, None, str(exc)

        stale: list[tuple[str, date]] = []
        errored: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=_UNIVERSE_FRESHNESS_THREADS) as pool:
            for sym, last_date, err in pool.map(_last_date_for, symbols):
                if err is not None:
                    errored.append((sym, err))
                elif last_date is None:
                    errored.append((sym, "no last_date"))
                elif last_date < cutoff:
                    stale.append((sym, last_date))

        # Errored tickers are themselves a red flag — can't verify freshness.
        if errored:
            sample = [f"{s}({e[:40]})" for s, e in errored[:5]]
            raise RuntimeError(
                f"Pre-flight: {len(errored)} universe ticker(s) in ArcticDB "
                f"could not be read for freshness check. Sample: {sample}. "
                "Treated as fatal because a silent read error here would "
                "mask exactly the kind of per-ticker write skip this scan "
                "exists to catch."
            )

        if stale:
            # Sort by stalest first so the operator sees the worst offenders.
            stale.sort(key=lambda x: x[1])
            summary = [f"{sym} (last={d.isoformat()})" for sym, d in stale[:10]]
            more = f" (+{len(stale) - 10} more)" if len(stale) > 10 else ""
            raise RuntimeError(
                f"Pre-flight: {len(stale)}/{len(symbols)} universe ticker(s) "
                f"have stale ArcticDB data (older than {max_stale_days} "
                f"calendar days, cutoff={cutoff.isoformat()}). "
                f"Top offenders: {summary}{more}. "
                "Same class as 2026-04-21 ASGN/MOH failure: daily_append "
                "skipped these tickers, executor guards would abort the "
                "backtester ~2 hours in. Backfill via polygon one-shot "
                "or investigate daily_append skip logic before re-running."
            )

    # ── Mode-specific primitives ─────────────────────────────────────────

    def _check_executor_config(self) -> None:
        """Validate the executor risk.yaml the simulate path will load.

        Mirrors executor/config_loader.py's canonical search order,
        minus the removed `.example` fallback (alpha-engine#73). Fails
        if no real risk.yaml is reachable, or if the loaded config
        carries placeholder bucket values, or if the executor's
        signals_bucket disagrees with the backtester's (both must read
        the same bucket or the backtest measures data-source drift
        instead of logic drift).
        """
        # If the caller didn't give us an executor repo root, skip —
        # executor's own import-time config_loader now hard-fails on
        # miss (alpha-engine#73), so this is a defense-in-depth check
        # rather than the sole safeguard.
        executor_root = next(
            (p for p in self.executor_paths if os.path.isdir(p)),
            None,
        )
        if executor_root is None:
            return

        candidate_paths = [
            os.path.expanduser("~/alpha-engine-config/executor/risk.yaml"),
            os.path.realpath(
                os.path.join(executor_root, "..", "alpha-engine-config", "executor", "risk.yaml")
            ),
            os.path.realpath(os.path.join(executor_root, "config", "risk.yaml")),
        ]
        resolved = next((p for p in candidate_paths if os.path.isfile(p)), None)
        if resolved is None:
            raise RuntimeError(
                "Pre-flight: executor risk.yaml not found in any of:\n  "
                + "\n  ".join(candidate_paths)
                + "\nBacktester simulate path will hard-fail on import. Clone "
                  "alpha-engine-config next to the alpha-engine repo, or populate "
                  "alpha-engine/config/risk.yaml from the .example template. The "
                  ".example is intentionally NOT a fallback (see alpha-engine#73)."
            )

        try:
            with open(resolved) as f:
                loaded = yaml.safe_load(f) or {}
        except Exception as exc:
            raise RuntimeError(
                f"Pre-flight: executor risk.yaml at {resolved} failed to parse: {exc}"
            ) from exc

        for key in ("signals_bucket", "trades_bucket"):
            value = loaded.get(key)
            if not isinstance(value, str) or not value:
                raise RuntimeError(
                    f"Pre-flight: executor risk.yaml at {resolved} is missing required "
                    f"key {key!r} or has an empty value."
                )
            if value.startswith(_PLACEHOLDER_PREFIX):
                raise RuntimeError(
                    f"Pre-flight: executor risk.yaml at {resolved} has placeholder "
                    f"{key}={value!r}. This is the .example template (or a copy that "
                    "wasn't filled in). Downstream ArcticDB/S3 reads would hit "
                    "nonexistent buckets — matches the 2026-04-20 KeyNotFoundException "
                    "incident."
                )

        executor_signals_bucket = loaded["signals_bucket"]
        if executor_signals_bucket != self.bucket:
            raise RuntimeError(
                f"Pre-flight: executor signals_bucket={executor_signals_bucket!r} does "
                f"not match backtester signals_bucket={self.bucket!r}. Simulate mode "
                "replays archived signals through the executor — both must read from "
                "the same S3 bucket or the backtest measures data-source drift instead "
                "of logic drift."
            )

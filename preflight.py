"""
Backtester preflight: connectivity + freshness checks run at the top of
each entrypoint before any real work starts.

Primitives live in ``alpha_engine_lib.preflight.BasePreflight``; this
module only composes them into a mode-specific sequence. See the
alpha-engine-lib README for the 2026-04-14 data-path failure mode that
motivated the library.

Modes:

- ``"backtest"`` — ``backtest.py`` entrypoint (weekly spot instance).
  S3 bucket reachable + ArcticDB ``macro/SPY`` fresh. The predictor-
  backtest stage reads ArcticDB directly; a stale universe would
  silently degrade the 10y synthetic signal pipeline. 8-day threshold
  covers Fri→Mon weekly cadence + buffer.
- ``"evaluate"`` — ``evaluate.py`` entrypoint. Reads simulation
  artifacts from S3 only, no ArcticDB. Keep the check cheap.
- ``"lambda_health"`` — daily predictor health check Lambda. Reads
  ``research.db`` + per-day metrics from S3. No ArcticDB.
"""

from __future__ import annotations

from alpha_engine_lib.preflight import BasePreflight


class BacktesterPreflight(BasePreflight):
    """Preflight checks for the three backtester entrypoints."""

    def __init__(self, bucket: str, mode: str):
        super().__init__(bucket)
        if mode not in ("backtest", "evaluate", "lambda_health"):
            raise ValueError(f"BacktesterPreflight: unknown mode {mode!r}")
        self.mode = mode

    def run(self) -> None:
        self.check_env_vars("AWS_REGION")
        self.check_s3_bucket()

        if self.mode == "backtest":
            # synthetic/predictor_backtest.py reads from ArcticDB. SPY
            # lives in the ``macro`` library (market-wide series); its
            # freshness is a sufficient signal that the ArcticDB write
            # path upstream (alpha-engine-data DailyData) is healthy.
            # 8-day threshold: weekly cadence + 1 day buffer.
            self.check_arcticdb_fresh("macro", "SPY", max_stale_days=8)

"""
Backtester preflight: connectivity + freshness checks run at the top of
each entrypoint before any real work starts.

Primitives live in ``alpha_engine_lib.preflight.BasePreflight``; this
module only composes them into a mode-specific sequence. See the
alpha-engine-lib README for the 2026-04-14 data-path failure mode that
motivated the library.

Modes:

- ``"backtest"`` — ``backtest.py`` entrypoint (weekly spot instance).
  S3 bucket reachable + ArcticDB ``macro/SPY`` fresh + the executor
  risk.yaml the simulate path will import resolves to a real config
  (not the placeholder .example template). 8-day threshold covers
  Fri→Mon weekly cadence + buffer.
- ``"evaluate"`` — ``evaluate.py`` entrypoint. Reads simulation
  artifacts from S3 only, no ArcticDB. Keep the check cheap.
- ``"lambda_health"`` — daily predictor health check Lambda. Reads
  ``research.db`` + per-day metrics from S3. No ArcticDB.
"""

from __future__ import annotations

import os

import yaml

from alpha_engine_lib.preflight import BasePreflight


# Placeholder prefix convention used by every repo's *.yaml.example
# template. A bucket/path value starting with this is definitionally a
# not-filled-in config and must never reach a live S3/ArcticDB read.
_PLACEHOLDER_PREFIX = "your-"


class BacktesterPreflight(BasePreflight):
    """Preflight checks for the three backtester entrypoints."""

    def __init__(
        self,
        bucket: str,
        mode: str,
        executor_paths: list[str] | None = None,
    ):
        super().__init__(bucket)
        if mode not in ("backtest", "evaluate", "lambda_health"):
            raise ValueError(f"BacktesterPreflight: unknown mode {mode!r}")
        self.mode = mode
        # backtest.py passes config["executor_paths"] here so preflight
        # can locate the executor repo root and validate its risk.yaml
        # will load with real values. Optional: omitted by evaluate.py
        # and lambda_health (they don't import the executor).
        self.executor_paths = executor_paths or []

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
            # backtest.py's simulate path imports executor.main, which
            # in turn imports executor.config_loader and reads the
            # executor's risk.yaml. If it resolves to the placeholder
            # .example template (or to a file with "your-*" bucket
            # names), every downstream S3/ArcticDB read fails deep in
            # the executor-sim call chain. Caught at preflight so the
            # operator sees the real cause in <1s. Hit 2026-04-20.
            self._check_executor_config()

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

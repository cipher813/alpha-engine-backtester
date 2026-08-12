"""attestation.py — the backtester's RUNTIME numeric correctness verdict.

WHY THIS EXISTS
---------------
The L4593 correctness battery already proves the simulation engine computes the
right numbers: leg (a) null calibration, leg (b) the independent NumPy oracle
(``synthetic/reference_sim.py``), leg (c) golden benchmarks against published
external truth. **All three run in CI.**

The weekly backtest does not run in CI. It runs on a spot instance that
``pip install -r requirements.txt``s from scratch at boot. ``vectorbt~=0.28.5`` is
a compatible-release specifier, and ``numpy`` / ``pandas`` / ``numba`` resolve
transitively alongside it — so the engine that produces every live number can be a
different build from the one CI proved correct, and nothing in the pipeline
notices. A fill-price, fee-side or NAV-marking change in a patch release would
bias **every** backtest identically: the optimizer would still emit a
plausible-looking ``executor_params.json``, the Report Card would still render a
grade, and no artifact anywhere would say the arithmetic had moved.

`sf-pipeline-policy.md` §2.3a names this exact shape: a **data artifact** missing
makes a consumer fail visibly; a **correctness verdict** missing makes every
consumer succeed *as though the check had passed*. This module is that verdict for
the backtester — computed **where the numbers are computed**, in the deployed
interpreter, on the deployed wheels, on every cycle.

WHAT IT ASSERTS
---------------
A small battery of hand-computable scenarios driven through the **production**
path (``vectorbt_bridge.orders_to_portfolio`` + ``portfolio_stats``), compared
against literals derived **by hand** (not by running this code) plus one
cross-check against the independent oracle. Each covers a distinct accounting axis
a systematic bug rides on:

===========================  ==============================================
``pnl_no_fees``              fill price + share accounting + NAV marking
``fee_charged_both_sides``   fee side/sign (entry AND exit, on notional)
``drawdown_peak_to_trough``  running-peak drawdown definition
``alpha_over_active_window`` benchmark window alignment (the 2026-05-24 bug)
``oracle_nav_agreement``     whole NAV path vs a from-scratch accountant
===========================  ==============================================

The literals are reproduced from ``tests/test_closed_form_scenarios.py``'s
hand-computed layer, which documents the arithmetic for each. They are duplicated
here **deliberately**: the CI test proves the engine on the CI runner, this module
proves it on the box that produced the week's numbers. Neither substitutes for the
other, and the shared literals are what make a divergence between the two
diagnostic rather than confusing.

CONTRACT
--------
``run_attestation()`` **never raises**. Any failure — an import error, an engine
that crashes, an unexpected exception — resolves to ``verdict == "UNKNOWN"`` with
the error class and message recorded. This is the one legitimate broad catch in
this module: the failure mode swallowed is "the attestation itself could not run";
the primary deliverable (the weekly backtest's own artifacts) survives untouched;
and the recording surface is the emitted ``attestation.json`` body plus an ERROR
log line. A verdict-producing stage that dies must not kill the stages that do not
depend on it (§2.3a), and must equally not let the ones that do proceed unmarked —
hence ``UNKNOWN``, never a silent pass.

Consumers: ``reporter.save`` always-emits ``backtest/{run_date}/attestation.json``;
``crucible-evaluator``'s Backtester tile grades it as a **critical** component and
the Report Card carries the verdict at top level. See `sf-pipeline-policy.md` §2.3a
rules 1–3.
"""

from __future__ import annotations

import logging
import platform
import time
from typing import Callable, NamedTuple

logger = logging.getLogger(__name__)

SCHEMA = "backtest_attestation-1.0.0"

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

#: Comparison tolerances. These are *closed-form* identities, so agreement is
#: expected to near machine precision — the bands are deliberately far tighter
#: than any plausible accounting bug. A relative band of 1e-9 catches a 1bp
#: systematic bias five orders of magnitude before it would move a grade.
_RTOL = 1e-9
_ATOL = 1e-12

_INIT_CASH = 1_000_000.0


class Scenario(NamedTuple):
    """One known-answer check.

    ``expected`` is a literal derived by hand (see the module docstring); ``compute``
    drives the production path and returns the comparable observed number. They are
    kept separate — rather than ``compute`` returning a bool — so the emitted
    artifact carries both numbers and a later divergence is diagnosable from the
    artifact alone (`principles.md` §2.1).
    """

    name: str
    description: str
    expected: float
    compute: Callable[[], float]
    rtol: float = _RTOL
    atol: float = _ATOL


def verdict_is_pass(verdict: str | None) -> bool:
    """True only for an explicit PASS.

    §2.3a rule 2: ``UNKNOWN`` — and anything else, including ``None`` and a
    truthy-looking ``"ok"`` — withholds the guarantee. Consumers call this rather
    than testing truthiness so the "missing reads as pass" bug cannot be written.
    """
    return verdict == PASS


# ════════════════════════════════════════════════════════════════════════════
# Scenarios
# ════════════════════════════════════════════════════════════════════════════

def _bdays(n: int):
    import pandas as pd

    return pd.bdate_range("2024-01-01", periods=n)


def _enter(date: str, ticker: str, shares: float, price: float) -> dict:
    return {"date": date, "ticker": ticker, "action": "ENTER",
            "shares": shares, "price_at_order": price}


def _exit(date: str, ticker: str, shares: float, price: float) -> dict:
    return {"date": date, "ticker": ticker, "action": "EXIT",
            "shares": shares, "price_at_order": price}


def _pnl_no_fees() -> float:
    """Buy 10 @ 100 (day 0), sell 10 @ 130 (day 3), zero fees.

    P&L = 10 * (130 - 100) = +300 on 1,000,000 → total_return = 3e-4 exactly.
    """
    import pandas as pd

    from vectorbt_bridge import orders_to_portfolio, portfolio_stats

    idx = _bdays(5)
    prices = pd.DataFrame({"AAA": [100., 110., 120., 130., 140.]}, index=idx)
    orders = [_enter("2024-01-01", "AAA", 10, 100.),
              _exit("2024-01-04", "AAA", 10, 130.)]
    pf = orders_to_portfolio(orders, prices, init_cash=_INIT_CASH, fees=0.0)
    return float(portfolio_stats(pf)["total_return"])


def _fee_charged_both_sides() -> float:
    """Same trade at 10 bps. Entry fee = 1000 * 0.001 = 1.0; exit fee =
    1300 * 0.001 = 1.3. Net P&L = 300 - 1.0 - 1.3 = 297.7.

    A fee applied on one side only, or on shares rather than notional, moves this.
    """
    import pandas as pd

    from vectorbt_bridge import orders_to_portfolio, portfolio_stats

    idx = _bdays(5)
    prices = pd.DataFrame({"AAA": [100., 110., 120., 130., 140.]}, index=idx)
    orders = [_enter("2024-01-01", "AAA", 10, 100.),
              _exit("2024-01-04", "AAA", 10, 130.)]
    pf = orders_to_portfolio(orders, prices, init_cash=_INIT_CASH, fees=0.001)
    return float(portfolio_stats(pf)["total_return"])


def _drawdown_peak_to_trough() -> float:
    """Hold 1000 sh through 100→120→130→110→90→140.

    Peak NAV at price 130 (1,030,000); trough at price 90 (990,000).
    max_drawdown = -40,000 / 1,030,000 = -0.03883495145631068.
    """
    import pandas as pd

    from vectorbt_bridge import orders_to_portfolio, portfolio_stats

    idx = _bdays(6)
    prices = pd.DataFrame({"AAA": [100., 120., 130., 110., 90., 140.]}, index=idx)
    orders = [_enter("2024-01-01", "AAA", 1000, 100.),
              _exit("2024-01-08", "AAA", 1000, 140.)]
    pf = orders_to_portfolio(orders, prices, init_cash=_INIT_CASH, fees=0.0)
    return float(portfolio_stats(pf)["max_drawdown"])


def _alpha_over_active_window() -> float:
    """total_alpha = total_return - spy_return over the ACTIVE window.

    The portfolio is flat until day 1; SPY runs 400 → 440 (+10%) with all of the
    move outside the flat prefix. Anchoring the benchmark on the full wrapper
    instead of the active window is the exact class that produced
    ``alpha_vs_ew_high_vol: -954%`` on 2026-05-24. Expected = 3e-4 - 0.10.
    """
    import pandas as pd

    from vectorbt_bridge import orders_to_portfolio, portfolio_stats

    idx = _bdays(5)
    prices = pd.DataFrame({"AAA": [100., 110., 120., 130., 140.]}, index=idx)
    spy = pd.Series([400., 400., 400., 400., 440.], index=idx)
    orders = [_enter("2024-01-01", "AAA", 10, 100.),
              _exit("2024-01-04", "AAA", 10, 130.)]
    pf = orders_to_portfolio(orders, prices, init_cash=_INIT_CASH, fees=0.0)
    return float(portfolio_stats(pf, spy_prices=spy)["total_alpha"])


def _oracle_nav_agreement() -> float:
    """Max |NAV_production - NAV_oracle| / init_cash over a two-asset, fee-and-
    slippage-bearing scenario. Expected 0 — the oracle never imports vectorbt.

    This is the only check whose expectation is not a hand-derived literal; its
    independence comes from the second implementation rather than from arithmetic.
    """
    import numpy as np
    import pandas as pd

    from synthetic.reference_sim import simulate_reference
    from vectorbt_bridge import orders_to_portfolio

    idx = _bdays(8)
    prices = pd.DataFrame({
        "AAA": [100., 105., 110., 108., 112., 120., 118., 125.],
        "BBB": [50., 48., 52., 55., 53., 51., 60., 58.],
    }, index=idx)
    orders = [
        _enter("2024-01-02", "AAA", 500, 105.),
        _enter("2024-01-03", "BBB", 1000, 52.),
        _exit("2024-01-05", "AAA", 500, 112.),
        _exit("2024-01-09", "BBB", 1000, 60.),
    ]
    fees, slippage_bps = 0.0015, 5.0
    pf = orders_to_portfolio(
        orders, prices, init_cash=_INIT_CASH, fees=fees, slippage_bps=slippage_bps,
    )
    ref = simulate_reference(
        orders, prices, init_cash=_INIT_CASH, fees=fees, slippage_bps=slippage_bps,
    )
    prod_nav = np.asarray(pf.value().to_numpy(), dtype=np.float64)
    ref_nav = np.asarray(ref.nav, dtype=np.float64)
    if prod_nav.shape != ref_nav.shape:
        # A shape mismatch is a real divergence, not a comparison error — report
        # it as an out-of-band deviation so the check FAILs rather than raising.
        return float("inf")
    return float(np.max(np.abs(prod_nav - ref_nav)) / _INIT_CASH)


def _SCENARIOS() -> list[Scenario]:
    """The battery. A callable (not a module constant) so a test can substitute
    it, and so no engine import happens at import time of this module."""
    return [
        Scenario(
            name="pnl_no_fees",
            description="10 sh @100 → @130, no fees: total_return = 300/1e6",
            expected=300.0 / _INIT_CASH,
            compute=_pnl_no_fees,
        ),
        Scenario(
            name="fee_charged_both_sides",
            description="same trade @10bps: 300 - 1.0 - 1.3 = 297.7",
            expected=297.7 / _INIT_CASH,
            compute=_fee_charged_both_sides,
        ),
        Scenario(
            name="drawdown_peak_to_trough",
            description="peak 1,030,000 → trough 990,000: -40000/1030000",
            expected=-40_000.0 / 1_030_000.0,
            compute=_drawdown_peak_to_trough,
        ),
        Scenario(
            name="alpha_over_active_window",
            description="total_return 3e-4 minus SPY +10% on the active window",
            expected=300.0 / _INIT_CASH - 0.10,
            compute=_alpha_over_active_window,
        ),
        Scenario(
            name="oracle_nav_agreement",
            description="max NAV deviation vs synthetic.reference_sim (no vectorbt)",
            expected=0.0,
            compute=_oracle_nav_agreement,
            atol=1e-9,
        ),
    ]


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def _engine_versions() -> dict:
    """Name the build that was attested. A verdict that does not identify the
    engine cannot explain a later divergence."""
    versions = {"python": platform.python_version()}
    for mod in ("vectorbt", "numpy", "pandas", "numba"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001 — a version probe never blocks
            versions[mod] = f"<unavailable: {type(exc).__name__}>"
    return versions


def run_attestation(run_date: str | None = None) -> dict:
    """Run the known-answer battery on the deployed engine and return the verdict.

    Never raises — see the module docstring's CONTRACT section. Returns a dict
    conforming to ``SCHEMA``.
    """
    started = time.monotonic()
    checks: list[dict] = []
    try:
        scenarios = _SCENARIOS()
    except Exception as exc:  # noqa: BLE001 — see CONTRACT: this becomes UNKNOWN
        logger.error(
            "attestation: battery could not be constructed (%s: %s) — verdict UNKNOWN. "
            "Every consumer must withhold the correctness guarantee this cycle.",
            type(exc).__name__, exc, exc_info=True,
        )
        return {
            "schema": SCHEMA,
            "run_date": run_date,
            "status": "error",
            "verdict": UNKNOWN,
            "checks": [],
            "n_checks": 0,
            "n_failed": 0,
            "engine": _engine_versions(),
            "error_class": type(exc).__name__,
            "error_msg": str(exc)[:500],
            "wall_clock_seconds": round(time.monotonic() - started, 3),
        }

    for sc in scenarios:
        record = {
            "name": sc.name,
            "description": sc.description,
            "expected": sc.expected,
            "observed": None,
            "abs_error": None,
            "rtol": sc.rtol,
            "atol": sc.atol,
            "passed": False,
        }
        try:
            observed = float(sc.compute())
            band = max(sc.atol, sc.rtol * abs(sc.expected))
            err = abs(observed - sc.expected)
            record["observed"] = observed
            record["abs_error"] = err
            record["passed"] = bool(err <= band)
            if not record["passed"]:
                logger.error(
                    "attestation check FAILED: %s expected=%r observed=%r abs_error=%r band=%r",
                    sc.name, sc.expected, observed, err, band,
                )
        except Exception as exc:  # noqa: BLE001 — a check that could not run is UNKNOWN
            record["errored"] = True
            record["error_class"] = type(exc).__name__
            record["error_msg"] = str(exc)[:500]
            logger.error(
                "attestation check ERRORED: %s (%s: %s)", sc.name, type(exc).__name__, exc,
                exc_info=True,
            )
        checks.append(record)

    # Outcome taxonomy: a check that DISAGREED is evidence the numbers are wrong
    # (FAIL); a check that could not RUN is absence of evidence (UNKNOWN). Both
    # withhold the guarantee, but only the first accuses the engine — collapsing
    # them would make an environment problem read as a correctness regression.
    n_failed = sum(1 for c in checks if not c["passed"] and not c.get("errored"))
    n_errored = sum(1 for c in checks if c.get("errored"))
    if n_failed:
        verdict = FAIL
    elif n_errored or not checks:
        verdict = UNKNOWN
    else:
        verdict = PASS
    result = {
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "ok",
        "verdict": verdict,
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": n_failed,
        "n_errored": n_errored,
        "engine": _engine_versions(),
        "wall_clock_seconds": round(time.monotonic() - started, 3),
    }
    if verdict == UNKNOWN:
        logger.error(
            "attestation UNKNOWN — %d/%d known-answer checks could not run. The "
            "correctness guarantee is WITHHELD this cycle (never granted by default).",
            n_errored, len(checks),
        )
    elif verdict == PASS:
        logger.info(
            "attestation PASS — %d/%d known-answer checks agreed on %s (vectorbt %s, numpy %s)",
            len(checks), len(checks), result["engine"]["python"],
            result["engine"]["vectorbt"], result["engine"]["numpy"],
        )
    else:
        logger.error(
            "attestation FAIL — %d/%d known-answer checks DISAGREE with their "
            "hand-derived expectation. THIS RUN'S NUMBERS ARE NOT TRUSTWORTHY; "
            "every consumer must withhold the guarantee.",
            n_failed, len(checks),
        )
    return result

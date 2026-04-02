"""
end_to_end.py — Full-pipeline attribution across all six decision boundaries.

Joins universe_returns with scanner_evaluations, team_candidates, cio_evaluations,
predictor_outcomes, executor_shadow_book, and trades to compute lift at each stage
of the pipeline. Answers: "Did this step improve on what it was given?"

Decision boundaries (upstream → downstream):
  1. Scanner filter:  900 → 50-70
  2. Sector teams:    50-70 → 12-18
  3. CIO promotion:   12-18 → ~5-8
  4. Predictor veto:  ~25 → ~20
  5. Executor trading: ~20 → ~15

All tables join on (ticker, eval_date). Every downstream table is a strict subset
of the one above it.

Writer: backtester (weekly, after universe_returns is populated).
Output: lift summary dict + optional e2e_attribution.csv on S3.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def compute_lift_metrics(
    research_db_path: str,
    trades_db_path: str | None = None,
    eval_date: str | None = None,
) -> dict:
    """
    Compute lift at each decision boundary for the given eval_date(s).

    Args:
        research_db_path: path to research.db (universe_returns, scanner_evaluations,
                          team_candidates, cio_evaluations, predictor_outcomes)
        trades_db_path: path to trades.db (trades, executor_shadow_book).
                        Optional — executor lift is skipped if not available.
        eval_date: optional filter. If None, computes across all available dates.

    Returns dict with:
        status: "ok" | "insufficient_data" | "error"
        n_dates: number of eval dates analyzed
        scanner_lift: {passing_avg, universe_avg, lift, n_passing, n_universe}
        team_lift: [{team_id, pick_avg, sector_avg, lift, n_picks}, ...]
        cio_lift: {advance_avg, all_recs_avg, lift, n_advance, n_recs}
        predictor_lift: {up_avg, all_avg, down_avg, lift, n_up, n_down, n_all}
        executor_lift: {traded_avg, approved_avg, lift, n_traded, n_approved}
        pipeline_lift: {traded_avg, universe_avg, lift}
    """
    if not Path(research_db_path).exists():
        return {"status": "error", "error": f"research.db not found at {research_db_path}"}

    conn = sqlite3.connect(research_db_path)

    # Check if universe_returns has data
    try:
        ur_count = conn.execute("SELECT COUNT(*) FROM universe_returns").fetchone()[0]
    except sqlite3.OperationalError:
        conn.close()
        return {"status": "error", "error": "universe_returns table does not exist"}

    if ur_count == 0:
        conn.close()
        return {"status": "insufficient_data", "error": "universe_returns is empty"}

    date_filter = ""
    params: list = []
    if eval_date:
        date_filter = " WHERE eval_date = ?"
        params = [eval_date]

    try:
        # Load universe_returns as base
        ur = pd.read_sql_query(
            f"SELECT * FROM universe_returns{date_filter} ORDER BY eval_date, ticker",
            conn, params=params,
        )

        n_dates = ur["eval_date"].nunique()
        if n_dates == 0:
            conn.close()
            return {"status": "insufficient_data", "error": "no data for specified date"}

        # Only use rows with 5d returns populated
        ur = ur[ur["return_5d"].notna()]
        if ur.empty:
            conn.close()
            return {"status": "insufficient_data", "error": "no rows with return_5d populated"}

        result: dict = {
            "status": "ok",
            "n_dates": n_dates,
            "n_universe_rows": len(ur),
        }

        # 1. Scanner lift
        result["scanner_lift"] = _scanner_lift(conn, ur, date_filter, params)

        # 2. Team lift
        result["team_lift"] = _team_lift(conn, ur, date_filter, params)

        # 3. CIO lift
        result["cio_lift"] = _cio_lift(conn, ur, date_filter, params)

        # 4. Predictor lift
        result["predictor_lift"] = _predictor_lift(conn, ur, date_filter, params)

        # 5. Executor lift (requires trades.db)
        if trades_db_path and Path(trades_db_path).exists():
            result["executor_lift"] = _executor_lift(trades_db_path, ur)
        else:
            result["executor_lift"] = {"status": "skipped", "reason": "trades.db not available"}

        # 6. Full pipeline lift
        result["pipeline_lift"] = _pipeline_lift(ur, result)

        conn.close()
        return result

    except Exception as e:
        conn.close()
        logger.error("end_to_end.compute_lift_metrics failed: %s", e)
        return {"status": "error", "error": str(e)}


def build_attribution_table(
    research_db_path: str,
    trades_db_path: str | None = None,
    eval_date: str | None = None,
) -> pd.DataFrame:
    """
    Build the full-pipeline attribution table — one row per ticker per eval_date.

    Joins all evaluation tables to produce a wide DataFrame for analysis and CSV export.
    """
    if not Path(research_db_path).exists():
        return pd.DataFrame()

    conn = sqlite3.connect(research_db_path)

    date_filter = ""
    params: list = []
    if eval_date:
        date_filter = " WHERE ur.eval_date = ?"
        params = [eval_date]

    try:
        query = f"""
        SELECT
            ur.ticker,
            ur.eval_date,
            ur.sector,
            ur.close_price,
            ur.return_5d,
            ur.return_10d,
            ur.spy_return_5d,
            ur.spy_return_10d,
            ur.beat_spy_5d,
            ur.beat_spy_10d,
            ur.sector_etf,
            ur.sector_etf_return_5d,
            ur.beat_sector_5d,
            se.tech_score,
            se.scan_path,
            se.quant_filter_pass,
            se.liquidity_pass,
            se.volatility_pass,
            se.balance_sheet_pass,
            se.filter_fail_reason,
            tc.team_id,
            tc.quant_rank,
            tc.quant_score AS team_quant_score,
            tc.qual_score AS team_qual_score,
            tc.team_recommended,
            ce.combined_score AS cio_combined_score,
            ce.macro_shift AS cio_macro_shift,
            ce.final_score AS cio_final_score,
            ce.cio_decision,
            ce.cio_conviction,
            ce.cio_rank,
            po.predicted_direction,
            po.prediction_confidence,
            po.p_up,
            po.p_down,
            po.actual_5d_return AS predictor_actual_5d
        FROM universe_returns ur
        LEFT JOIN scanner_evaluations se
            ON ur.ticker = se.ticker AND ur.eval_date = se.eval_date
        LEFT JOIN team_candidates tc
            ON ur.ticker = tc.ticker AND ur.eval_date = tc.eval_date
        LEFT JOIN cio_evaluations ce
            ON ur.ticker = ce.ticker AND ur.eval_date = ce.eval_date
        LEFT JOIN predictor_outcomes po
            ON ur.ticker = po.symbol AND ur.eval_date = po.prediction_date
        {date_filter}
        ORDER BY ur.eval_date, ur.ticker
        """
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Join trades if available
        if trades_db_path and Path(trades_db_path).exists():
            trades_conn = sqlite3.connect(trades_db_path)
            try:
                trades = pd.read_sql_query(
                    "SELECT ticker, date AS eval_date, action, fill_price, "
                    "realized_return_pct, trigger_type, exit_type "
                    "FROM trades WHERE action = 'ENTER'",
                    trades_conn,
                )
                if not trades.empty:
                    df = df.merge(trades, on=["ticker", "eval_date"], how="left")

                # Shadow book
                try:
                    shadow = pd.read_sql_query(
                        "SELECT ticker, date AS eval_date, block_reason, "
                        "intended_position_pct "
                        "FROM executor_shadow_book",
                        trades_conn,
                    )
                    if not shadow.empty:
                        df = df.merge(
                            shadow, on=["ticker", "eval_date"], how="left",
                            suffixes=("", "_shadow"),
                        )
                except sqlite3.OperationalError:
                    pass  # shadow book table may not exist yet
            finally:
                trades_conn.close()

        return df

    except Exception as e:
        logger.error("build_attribution_table failed: %s", e)
        conn.close()
        return pd.DataFrame()


def format_lift_report(metrics: dict) -> list[str]:
    """Format lift metrics as markdown lines for inclusion in the weekly report."""
    lines = ["## Pipeline evaluation — Decision boundary lift"]

    if metrics.get("status") != "ok":
        lines.append(f"\n> {metrics.get('status', 'unknown')}: {metrics.get('error', '')}")
        return lines

    lines.append(f"\n*{metrics['n_dates']} evaluation dates, {metrics['n_universe_rows']} universe rows*\n")

    lines.append("| Decision | Population | Avg 5d return | Baseline 5d return | Lift | n |")
    lines.append("|----------|------------|---------------|-------------------|------|---|")

    sl = metrics.get("scanner_lift", {})
    if sl and sl.get("status") != "skipped":
        lines.append(
            f"| Scanner filter | {sl.get('n_passing', '?')} / {sl.get('n_universe', '?')} | "
            f"{_pct(sl.get('passing_avg'))} | {_pct(sl.get('universe_avg'))} | "
            f"{_pct(sl.get('lift'))} | {sl.get('n_passing', '')} |"
        )

    cl = metrics.get("cio_lift", {})
    if cl and cl.get("status") != "skipped":
        lines.append(
            f"| CIO promotion | {cl.get('n_advance', '?')} / {cl.get('n_recs', '?')} | "
            f"{_pct(cl.get('advance_avg'))} | {_pct(cl.get('all_recs_avg'))} | "
            f"{_pct(cl.get('lift'))} | {cl.get('n_advance', '')} |"
        )

    pl = metrics.get("predictor_lift", {})
    if pl and pl.get("status") != "skipped":
        lines.append(
            f"| Predictor (UP) | {pl.get('n_up', '?')} / {pl.get('n_all', '?')} | "
            f"{_pct(pl.get('up_avg'))} | {_pct(pl.get('all_avg'))} | "
            f"{_pct(pl.get('lift'))} | {pl.get('n_up', '')} |"
        )

    el = metrics.get("executor_lift", {})
    if el and el.get("status") != "skipped":
        lines.append(
            f"| Executor trading | {el.get('n_traded', '?')} / {el.get('n_approved', '?')} | "
            f"{_pct(el.get('traded_avg'))} | {_pct(el.get('approved_avg'))} | "
            f"{_pct(el.get('lift'))} | {el.get('n_traded', '')} |"
        )

    pipl = metrics.get("pipeline_lift", {})
    if pipl and pipl.get("status") != "skipped":
        lines.append(
            f"| **Full pipeline** | — | "
            f"{_pct(pipl.get('traded_avg'))} | {_pct(pipl.get('universe_avg'))} | "
            f"**{_pct(pipl.get('lift'))}** | — |"
        )

    # Team lift breakdown
    tl = metrics.get("team_lift", [])
    if tl and not isinstance(tl, dict):
        lines.append("\n### Sector team lift\n")
        lines.append("| Team | Pick avg 5d | Sector avg 5d | Lift | n picks |")
        lines.append("|------|-------------|---------------|------|---------|")
        for t in tl:
            lines.append(
                f"| {t.get('team_id', '?')} | {_pct(t.get('pick_avg'))} | "
                f"{_pct(t.get('sector_avg'))} | {_pct(t.get('lift'))} | "
                f"{t.get('n_picks', '')} |"
            )

    return lines


# ── Internal lift computations ───────────────────────────────────────────────

def _scanner_lift(conn, ur: pd.DataFrame, date_filter: str, params: list) -> dict:
    """Scanner filter lift: passing stocks vs. full universe."""
    try:
        se_filter = date_filter.replace("eval_date", "se.eval_date") if date_filter else ""
        se = pd.read_sql_query(
            f"SELECT ticker, eval_date, quant_filter_pass FROM scanner_evaluations se{se_filter}",
            conn, params=params,
        )
        if se.empty:
            return {"status": "skipped", "reason": "scanner_evaluations empty"}

        merged = ur.merge(se, on=["ticker", "eval_date"], how="inner")
        passing = merged[merged["quant_filter_pass"] == 1]

        universe_avg = float(merged["return_5d"].mean())
        passing_avg = float(passing["return_5d"].mean()) if not passing.empty else None
        lift = (passing_avg - universe_avg) if passing_avg is not None else None

        return {
            "universe_avg": round(universe_avg, 4),
            "passing_avg": round(passing_avg, 4) if passing_avg is not None else None,
            "lift": round(lift, 4) if lift is not None else None,
            "n_universe": len(merged),
            "n_passing": len(passing),
        }
    except sqlite3.OperationalError:
        return {"status": "skipped", "reason": "scanner_evaluations table not found"}


def _team_lift(conn, ur: pd.DataFrame, date_filter: str, params: list) -> list[dict] | dict:
    """Sector team lift: team picks vs. own sector average."""
    try:
        tc_filter = date_filter.replace("eval_date", "tc.eval_date") if date_filter else ""
        tc = pd.read_sql_query(
            f"SELECT ticker, eval_date, team_id, team_recommended FROM team_candidates tc{tc_filter}",
            conn, params=params,
        )
        if tc.empty:
            return {"status": "skipped", "reason": "team_candidates empty"}

        merged = ur.merge(tc, on=["ticker", "eval_date"], how="inner")
        results = []

        for team_id in sorted(merged["team_id"].unique()):
            team_data = merged[merged["team_id"] == team_id]
            picks = team_data[team_data["team_recommended"] == 1]

            sector_avg = float(team_data["return_5d"].mean())
            pick_avg = float(picks["return_5d"].mean()) if not picks.empty else None
            lift = (pick_avg - sector_avg) if pick_avg is not None else None

            results.append({
                "team_id": team_id,
                "sector_avg": round(sector_avg, 4),
                "pick_avg": round(pick_avg, 4) if pick_avg is not None else None,
                "lift": round(lift, 4) if lift is not None else None,
                "n_candidates": len(team_data),
                "n_picks": len(picks),
            })

        return results
    except sqlite3.OperationalError:
        return {"status": "skipped", "reason": "team_candidates table not found"}


def _cio_lift(conn, ur: pd.DataFrame, date_filter: str, params: list) -> dict:
    """CIO lift: ADVANCE stocks vs. all sector recommendations."""
    try:
        ce_filter = date_filter.replace("eval_date", "ce.eval_date") if date_filter else ""
        ce = pd.read_sql_query(
            f"SELECT ticker, eval_date, cio_decision FROM cio_evaluations ce{ce_filter}",
            conn, params=params,
        )
        if ce.empty:
            return {"status": "skipped", "reason": "cio_evaluations empty"}

        merged = ur.merge(ce, on=["ticker", "eval_date"], how="inner")
        advance = merged[merged["cio_decision"] == "ADVANCE"]

        all_recs_avg = float(merged["return_5d"].mean())
        advance_avg = float(advance["return_5d"].mean()) if not advance.empty else None
        lift = (advance_avg - all_recs_avg) if advance_avg is not None else None

        reject = merged[merged["cio_decision"] == "REJECT"]
        reject_avg = float(reject["return_5d"].mean()) if not reject.empty else None

        return {
            "all_recs_avg": round(all_recs_avg, 4),
            "advance_avg": round(advance_avg, 4) if advance_avg is not None else None,
            "reject_avg": round(reject_avg, 4) if reject_avg is not None else None,
            "lift": round(lift, 4) if lift is not None else None,
            "n_recs": len(merged),
            "n_advance": len(advance),
            "n_reject": len(reject),
        }
    except sqlite3.OperationalError:
        return {"status": "skipped", "reason": "cio_evaluations table not found"}


def _predictor_lift(conn, ur: pd.DataFrame, date_filter: str, params: list) -> dict:
    """Predictor lift: UP-predicted vs. all portfolio stocks."""
    try:
        po_filter = date_filter.replace("eval_date = ?", "prediction_date = ?") if date_filter else ""
        po = pd.read_sql_query(
            f"SELECT symbol AS ticker, prediction_date AS eval_date, "
            f"predicted_direction, prediction_confidence "
            f"FROM predictor_outcomes{po_filter}",
            conn, params=params,
        )
        if po.empty:
            return {"status": "skipped", "reason": "predictor_outcomes empty"}

        merged = ur.merge(po, on=["ticker", "eval_date"], how="inner")
        if merged.empty:
            return {"status": "skipped", "reason": "no matching predictor_outcomes in universe_returns"}

        up = merged[merged["predicted_direction"] == "UP"]
        down = merged[merged["predicted_direction"] == "DOWN"]

        all_avg = float(merged["return_5d"].mean())
        up_avg = float(up["return_5d"].mean()) if not up.empty else None
        down_avg = float(down["return_5d"].mean()) if not down.empty else None
        lift = (up_avg - all_avg) if up_avg is not None else None

        return {
            "all_avg": round(all_avg, 4),
            "up_avg": round(up_avg, 4) if up_avg is not None else None,
            "down_avg": round(down_avg, 4) if down_avg is not None else None,
            "lift": round(lift, 4) if lift is not None else None,
            "n_all": len(merged),
            "n_up": len(up),
            "n_down": len(down),
        }
    except sqlite3.OperationalError:
        return {"status": "skipped", "reason": "predictor_outcomes table not found"}


def _executor_lift(trades_db_path: str, ur: pd.DataFrame) -> dict:
    """Executor lift: traded returns vs. approved (non-blocked) entries."""
    try:
        trades_conn = sqlite3.connect(trades_db_path)
        trades = pd.read_sql_query(
            "SELECT ticker, date AS eval_date, realized_return_pct "
            "FROM trades WHERE action = 'ENTER'",
            trades_conn,
        )

        # Shadow book (blocked entries)
        try:
            shadow = pd.read_sql_query(
                "SELECT ticker, date AS eval_date, block_reason "
                "FROM executor_shadow_book",
                trades_conn,
            )
        except sqlite3.OperationalError:
            shadow = pd.DataFrame()

        trades_conn.close()

        if trades.empty:
            return {"status": "skipped", "reason": "no ENTER trades"}

        # Merge trades with universe_returns for forward returns
        traded = ur.merge(trades, on=["ticker", "eval_date"], how="inner")

        # Approved = traded + not blocked
        # "Approved" baseline is all portfolio stocks that weren't blocked
        approved = ur.merge(
            trades[["ticker", "eval_date"]], on=["ticker", "eval_date"], how="inner"
        )
        if not shadow.empty:
            blocked = ur.merge(shadow[["ticker", "eval_date"]], on=["ticker", "eval_date"], how="inner")
            approved = pd.concat([approved, blocked])

        traded_avg = float(traded["return_5d"].mean()) if not traded.empty else None
        approved_avg = float(approved["return_5d"].mean()) if not approved.empty else None
        lift = (traded_avg - approved_avg) if traded_avg is not None and approved_avg is not None else None

        return {
            "traded_avg": round(traded_avg, 4) if traded_avg is not None else None,
            "approved_avg": round(approved_avg, 4) if approved_avg is not None else None,
            "lift": round(lift, 4) if lift is not None else None,
            "n_traded": len(traded),
            "n_approved": len(approved),
        }
    except Exception as e:
        return {"status": "skipped", "reason": str(e)}


def _pipeline_lift(ur: pd.DataFrame, result: dict) -> dict:
    """Full pipeline lift: best available traded return vs. universe average."""
    universe_avg = float(ur["return_5d"].mean())

    # Use executor traded_avg if available, otherwise CIO advance_avg
    el = result.get("executor_lift", {})
    cl = result.get("cio_lift", {})

    traded_avg = el.get("traded_avg") if el.get("status") != "skipped" else None
    if traded_avg is None:
        traded_avg = cl.get("advance_avg") if cl.get("status") != "skipped" else None

    if traded_avg is None:
        return {"status": "skipped", "reason": "no downstream returns available"}

    return {
        "universe_avg": round(universe_avg, 4),
        "traded_avg": round(traded_avg, 4),
        "lift": round(traded_avg - universe_avg, 4),
    }


def _pct(val) -> str:
    """Format a decimal return as percentage string."""
    if val is None:
        return "—"
    return f"{val * 100:+.2f}%"

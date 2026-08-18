"""Class guard: no ``config["_x"]`` key is READ that nothing WRITES (config-I7616).

The bug class, three live instances in one 40-line block of ``evaluate.py``:

    prices_for_metrics = config.get("_prices")            # written by nothing
    ohlc_for_metrics   = config.get("_ohlcv_by_ticker")   # written by nothing
    spy_prices         = config.get("_spy_prices")        # written by nothing

The leading underscore marks a key the pipeline threads through the config dict
at runtime rather than one an operator sets in ``config.yaml``. That convention
is exactly what makes the defect invisible: the key is absent from every YAML
file *by design*, so its absence at runtime looks normal, ``.get`` returns
``None``, and the consumer degrades to ``insufficient_data`` forever. Three
downstream metrics — ``team_metrics``, ``portfolio_excursion`` and the
``risk_ratio_ci`` monitor the evaluator grades as CRITICAL — were dead for
months behind exactly that, and the enclosing ``except Exception`` meant nothing
ever raised.

Per-instance fixes do not survive the class: config-I7600 wired ONE of the three
keys and left the other two dead. This is the structural check. It reads source,
not runtime, so it fires on the commit that introduces the next one.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Directories with no production config-threading (and no obligation to write).
_SKIP_DIRS = {
    ".git", ".venv", "venv", "build", "dist", "__pycache__",
    "node_modules", ".worktrees", ".pytest_cache",
}

#: Variable names that hold the pipeline config dict at a read site. Narrow on
#: purpose: this guard is about the config dict threaded through the backtester,
#: not about every dict in the repo that happens to have an underscored key.
_CONFIG_NAMES = {
    "config", "cfg", "_cfg", "base_config", "combo_config", "combo",
    "sim_config", "config_a", "config_b", "run_config", "self._cfg",
}


def _is_config_receiver(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _CONFIG_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _CONFIG_NAMES
    return False


def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _source_files() -> list[pathlib.Path]:
    out = []
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        out.append(path)
    return sorted(out)


def _scan() -> tuple[dict[str, set[str]], set[str]]:
    """Return ``({read_key: {"file:line", ...}}, {written_key, ...})``.

    A WRITE is any construction that can put the key into a dict the pipeline
    passes on — subscript assignment, a dict literal, ``setdefault``, or an
    ``update``/``**`` merge. Deliberately generous: a false "written" costs a
    missed detection of one key, while a false "unwritten" would make the guard
    noise and get it deleted.
    """
    reads: dict[str, set[str]] = {}
    writes: set[str] = set()

    for path in _source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:  # pragma: no cover — a broken file fails elsewhere
            continue
        rel = path.relative_to(REPO_ROOT)
        is_test = rel.parts and rel.parts[0] == "tests"

        for node in ast.walk(tree):
            # ── writes ───────────────────────────────────────────────────────
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        key = _const_str(target.slice)
                        if key and key.startswith("_"):
                            writes.add(key)
            if isinstance(node, ast.Dict):
                for k in node.keys:
                    key = _const_str(k) if k is not None else None
                    if key and key.startswith("_"):
                        writes.add(key)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "setdefault" and node.args:
                    key = _const_str(node.args[0])
                    if key and key.startswith("_"):
                        writes.add(key)

            # ── reads (production source only) ───────────────────────────────
            if is_test:
                continue
            key = None
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and _is_config_receiver(node.func.value)
            ):
                key = _const_str(node.args[0])
            elif isinstance(node, ast.Subscript) and _is_config_receiver(node.value):
                key = _const_str(node.slice)
            if key and key.startswith("_"):
                reads.setdefault(key, set()).add(f"{rel}:{node.lineno}")

    return reads, writes


def test_scanner_finds_the_known_runtime_threaded_keys() -> None:
    """The guard is only worth anything if it can see real reads.

    A source scanner that silently matches nothing passes forever. Pin a key
    that is genuinely threaded through the config at runtime, so a refactor that
    breaks the AST patterns fails HERE rather than by going quiet.
    """
    reads, writes = _scan()
    assert reads, "scanner found no config['_*'] reads at all — it is broken"
    assert "_phase_registry" in writes
    assert "_portfolio_stats" in reads


def test_no_private_config_key_is_read_without_a_writer() -> None:
    reads, writes = _scan()
    orphans = {k: sorted(v) for k, v in reads.items() if k not in writes}
    assert not orphans, (
        "These config keys are READ but nothing in the repository WRITES them, "
        "so they resolve to None on every run and the consumer degrades "
        "silently (config-I7616). Wire the key from what the pipeline already "
        "persisted for the date, or delete the read:\n"
        + "\n".join(f"  {k}: {sites}" for k, sites in sorted(orphans.items()))
    )


@pytest.mark.parametrize("key", ["_prices", "_ohlcv_by_ticker", "_spy_prices"])
def test_the_three_config_i7616_keys_are_gone(key: str) -> None:
    """The specific instances, pinned so they cannot come back by copy-paste."""
    reads, _ = _scan()
    assert key not in reads, (
        f"config[{key!r}] is read again at {sorted(reads[key])}. Nothing writes "
        f"it; config-I7616 replaced these reads with phase-marker loads "
        f"(evaluate._load_price_matrix / _load_ohlcv_for_excursion / "
        f"_load_spy_prices)."
    )

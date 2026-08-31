"""Schema-contract test for `console.descriptor.yaml` (alpha-engine-config-I9497).

`console.descriptor.yaml` at the repo root is this component's onboarding onto
the fleet console (console-policy.md §2.6). This test is the producer-side
half of that contract: it loads the descriptor and validates it against a
vendored copy of nousergon-console's own schema — mirroring
`crucible-evaluator/tests/test_console_descriptor_schema.py`, the first
consumer of this pattern. This repo has no runtime dependency on
nousergon-console, so the schema is a fixture, not an import.

`tests/fixtures/component_descriptor.schema.json` and
`tests/fixtures/console_known_drivers.json` are copies of
`nousergon-console/console/schemas/component_descriptor.schema.json` and
`console/drivers/__init__.py::KNOWN_DRIVERS`, verified byte-identical against
nousergon-console main on 2026-08-31. Copies, not symlinks, because this is a
public AGPL repo and nousergon-console's working tree is not guaranteed
present at test time; re-sync by hand if the upstream schema changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DESCRIPTOR_PATH = REPO_ROOT / "console.descriptor.yaml"
SCHEMA_PATH = REPO_ROOT / "tests" / "fixtures" / "component_descriptor.schema.json"
KNOWN_DRIVERS_PATH = REPO_ROOT / "tests" / "fixtures" / "console_known_drivers.json"

OWNING_REPO = "crucible-backtester"
COMPONENT_ID = "crucible-backtester-trials"

#: Every field the fleet registry's own schema requires
#: (`nous-ergon-ops/scripts/observability_registry.py::REQUIRED_FIELDS`). A
#: repo-local descriptor gathered by `observability-registry-publish.yml` is
#: validated as a FULL registry row, not merely as console bindings — a row
#: missing one of these fails the fleet publish on `main`, after every
#: pre-merge gate has gone green (alpha-engine-config-I9002). Vendored as a
#: literal for the same reason the schema is: no dependency on nous-ergon-ops.
REGISTRY_REQUIRED_FIELDS = (
    "component_id", "owning_repo", "substrate", "origin", "lifecycle", "owner",
    "authority_tier", "signals", "log_location", "alert_channel",
    "severity_source", "console_surface", "retention",
)

REGISTRY_SIGNAL_CLASSES = ("execution", "cost", "resource", "data", "outcome")
REGISTRY_SIGNAL_VALUES = {"emitted", "absent", "n/a", "unknown"}


def _load_descriptor() -> dict:
    with open(DESCRIPTOR_PATH) as fh:
        return yaml.safe_load(fh)


def _load_schema() -> dict:
    with open(SCHEMA_PATH) as fh:
        return json.load(fh)


def _load_known_drivers() -> set[str]:
    with open(KNOWN_DRIVERS_PATH) as fh:
        return set(json.load(fh)["drivers"])


def _bindings(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def test_descriptor_validates_against_schema():
    jsonschema.validate(instance=_load_descriptor(), schema=_load_schema())


def test_descriptor_declares_its_own_component_id():
    """`component_id` is the one name every signal, log key, alert and console
    row uses (console-policy.md §3.6), assigned by the component owner and
    never re-minted by the console. Pinned so a rename has to be deliberate.
    """
    assert _load_descriptor()["component_id"] == COMPONENT_ID


def test_descriptor_owning_repo_matches_this_repo():
    """`gather_repo_descriptors.py`'s provenance check fails the whole gather
    when `declared_repo != repo` — and the gather is all-or-nothing, so one
    wrong `owning_repo` blocks EVERY fleet row from reaching the box.

    Compared against a LITERAL, not against `REPO_ROOT.name`: this repo is
    routinely checked out into a git worktree under `~/Development/.worktrees/`
    whose directory name carries the branch, so a directory-derived expectation
    fails in exactly the place the change is being made.
    """
    assert _load_descriptor()["owning_repo"] == OWNING_REPO


def test_descriptor_is_a_full_registry_row():
    """A gathered descriptor IS the registry row, not just its bindings.

    The failure this pins: a duplicate or incomplete row is green through
    every pre-merge gate and fails only once on `main`, where it blocks the
    fleet registry publish for everybody (alpha-engine-config-I9002 —
    measured, the console received no registry updates for nine minutes).
    """
    d = _load_descriptor()
    for field in REGISTRY_REQUIRED_FIELDS:
        assert d.get(field) not in (None, "", [], {}), f"missing required registry field {field!r}"
    for field in ("component_id", "owning_repo", "substrate", "origin", "lifecycle", "owner"):
        assert str(d[field]).strip().lower() != "unknown", (
            f"identity field {field!r} has no 'unknown' — a component nobody "
            "can place is a discovery, not a row"
        )
    for cls in REGISTRY_SIGNAL_CLASSES:
        entry = d["signals"][cls]
        status = entry["status"] if isinstance(entry, dict) else entry
        assert status in REGISTRY_SIGNAL_VALUES, f"signals.{cls} = {status!r}"
        if status in ("absent", "n/a", "unknown"):
            assert entry.get("reason"), (
                f"signals.{cls} is {status!r} with no reason — an absent "
                "signal with no stated reason is an unobserved one"
            )
    for claim in ("authority_tier", "log_location", "alert_channel",
                  "severity_source", "console_surface", "retention"):
        if str(d.get(claim, "")).strip().lower() == "unknown":
            assert d.get(f"{claim}_reason"), (
                f"{claim!r} is 'unknown' with no {claim}_reason — an "
                "unaudited claim is a work item only if it says so"
            )


def test_descriptor_only_names_registered_drivers():
    """A descriptor naming a driver that does not exist FAILS THE BUILD
    (console-policy.md §2.7) rather than rendering the component absent — a
    typo and a genuinely-gone component must not look the same.
    """
    d = _load_descriptor()
    known = _load_known_drivers()
    for key in ("runs", "artifacts", "metrics"):
        for binding in _bindings(d.get(key)):
            assert binding["driver"] in known, (
                f"{key} binding names driver {binding.get('driver')!r}, which "
                "is not in the vendored console driver registry."
            )


def test_metrics_carries_a_document_fields_observation():
    """THE reason this row cannot render a false transparency gap.

    A registry row with `lifecycle: in-service` and no observation renders
    UNREPORTED (`console/adapters/yaml_directory.py::_state_from_row`), which
    is §9.2's transparency-gap count itself. `document-fields` is the one
    driver that emits a COMPONENT entity under the descriptor's own
    `component_id` as an OBSERVATION claim, and observation outranks a
    declaration-without-lifecycle for `state`
    (`console/index/merge.py::_state_rank`, 1 vs 3). Drop this binding and
    this row silently becomes a transparency gap on the next console build,
    which is exactly the outcome I9497 was written to avoid causing.
    """
    d = _load_descriptor()
    doc_bindings = [b for b in _bindings(d.get("metrics")) if b["driver"] == "document-fields"]
    assert len(doc_bindings) == 1, "expected exactly one document-fields metrics binding"
    binding = doc_bindings[0]
    assert len(binding["documents"]) == 1
    assert binding["cadence_minutes"] > 0, (
        "without a cadence staleness is not computable (console-policy §5.2)"
    )


def test_every_bound_key_is_a_fixed_pointer():
    """No binding may name a DATE-TEMPLATED key.

    No console driver substitutes a date into a key, so a templated key
    renders ABSENT on every day that is not the day it was written — a
    detector that cries wolf rather than one that measures. The templated
    keys this repo genuinely produces belong in `produces`/`consumes`
    (lineage, nothing is read to satisfy them), never in a binding.
    """
    d = _load_descriptor()
    for key_name in ("runs", "artifacts", "metrics"):
        for binding in _bindings(d.get(key_name)):
            keys = [binding["key"]] if "key" in binding else []
            keys += [doc["key"] for doc in binding.get("documents", [])]
            for k in keys:
                assert "{" not in k and "}" not in k, (
                    f"{key_name} binding names templated key {k!r}"
                )


def test_artifact_bindings_declare_a_cadence():
    """§5.2: without a cadence staleness is not computable, and the console
    says so rather than assuming fresh. Every artifact bound here is a fixed
    pointer with a known rewrite rhythm, so there is no honest reason to omit
    one.
    """
    d = _load_descriptor()
    bindings = _bindings(d.get("artifacts"))
    assert bindings, "at least one artifact binding is expected"
    for binding in bindings:
        assert binding.get("cadence_minutes", 0) > 0


def test_no_binding_names_a_retired_artifact():
    """alpha-engine-config-I9620: a freshness binding on a deliberately
    retired artifact renders a declared decision as a permanent defect.

    `backtest/grade_history.json`'s producer was deleted in `3eba0bb`
    (PR678, 2026-08-16, config-I7474) and the artifact was tombstoned out of
    `ARTIFACT_REGISTRY.yaml` on 2026-08-25 (config-I8406). Binding it made
    this descriptor claim a producer that no longer exists.

    The list is explicit rather than derived because the tombstones live in
    a private repo this public one cannot read. Adding a retired key here is
    the cheap half of retiring one.
    """
    retired = {"backtest/grade_history.json"}
    d = _load_descriptor()
    for section in ("runs", "artifacts", "metrics"):
        for binding in _bindings(d.get(section)):
            keys = [binding["key"]] if "key" in binding else []
            keys += [doc["key"] for doc in binding.get("documents", [])]
            for k in keys:
                for r in retired:
                    assert r not in k, (
                        f"{section} binding names retired artifact {r!r}: {k}"
                    )
    for k in list(d.get("produces") or []) + list(d.get("consumes") or []):
        for r in retired:
            assert r not in k, f"lineage names retired artifact {r!r}: {k}"


def test_numeric_fields_declare_an_explicit_null_baseline():
    """§5.4: green means BETTER THAN THE BASELINE, and where there is no
    baseline there is no colour. alpha-engine-config-I9005 owns Crucible's
    viability bars and is RESERVED for Brian's ruling, so every numeric field
    here ships uncoloured — with `baseline: null` written down, which is the
    difference between "deliberately uncoloured" and "somebody forgot".
    """
    d = _load_descriptor()
    for binding in _bindings(d.get("metrics")):
        for doc in binding.get("documents", []):
            for name, spec in doc["fields"].items():
                if spec.get("render") == "text":
                    continue
                assert "baseline" in spec, f"{name} renders a value with no declared baseline"
                assert spec["baseline"] is None, f"{name} invents a baseline I9005 owns"
                assert spec.get("unit"), f"{name} is numeric with no declared unit"

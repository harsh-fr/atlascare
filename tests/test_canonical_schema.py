"""
tests/test_canonical_schema.py
==============================
Validate that the four CANONICAL data files supplied to the app conform to the
strict schemas in example_schema/. These four files are the only inputs the
evaluator provides; everything else is derived from them. If a swapped-in data
folder violates the schema, these tests fail loudly with the exact violations.

Validation uses the same dependency-free checker the startup bootstrap uses
(data.derive_support_files.validate_against_schema), so "tests pass" and "app
boots cleanly" stay in agreement.

Paths are resolved exactly like the repositories: the per-file env var if set,
otherwise data/<file>. So this validates whatever data the app will actually
read.
"""

import json
import os

import pytest

from data.derive_support_files import validate_against_schema

# key -> (env var used by the repos, default filename)
_CANONICAL = {
    "crm_cases":      ("CRM_DATA_PATH",       "crm_cases.json"),
    "orders":         ("ORDERS_DATA_PATH",    "orders.json"),
    "kb_articles":    ("KB_DATA_PATH",        "kb_articles.json"),
    "payment_config": ("PAYMENT_CONFIG_PATH", "payment_config.json"),
}

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _resolve(env_var: str, filename: str) -> str:
    return os.path.abspath(os.getenv(env_var) or os.path.join(_DATA_DIR, filename))


@pytest.mark.parametrize("key", list(_CANONICAL))
def test_canonical_file_present_and_parses(key):
    env_var, filename = _CANONICAL[key]
    path = _resolve(env_var, filename)
    assert os.path.exists(path), f"Canonical file missing: {path}"
    with open(path, "r", encoding="utf-8") as fh:
        try:
            json.load(fh)
        except json.JSONDecodeError as exc:
            pytest.fail(f"{filename} is not valid JSON: {exc}")


@pytest.mark.parametrize("key", list(_CANONICAL))
def test_canonical_file_matches_schema(key):
    env_var, filename = _CANONICAL[key]
    path = _resolve(env_var, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not present at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    violations = validate_against_schema(key, data)
    assert not violations, (
        f"{filename} violates example_schema/schema_{key}.json:\n  - "
        + "\n  - ".join(violations)
    )

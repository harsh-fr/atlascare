"""
run_tests.py
============
Pre-launch test runner for AtlasCare.

Run this before starting the server to verify everything is healthy.
Exits with code 0 on success, 1 on failure — safe to use in CI or
as a startup gate.

Usage:
    python run_tests.py              # run all tests
    python run_tests.py --fast       # skip slow edge case tests
    python run_tests.py --suite guardrails   # run one suite only
"""

import sys
import argparse
import subprocess
from pathlib import Path


SUITES = {
    "contracts":   "tests/test_contracts.py",
    "guardrails":  "tests/test_guardrails.py",
    "security":    "tests/test_security.py",
    "journeys":    "tests/test_journeys.py",
    "edge_cases":  "tests/test_edge_cases.py",
    "regression":  "tests/test_regression.py",
}

FAST_SUITES = ["contracts", "guardrails", "security", "regression"]


def run(suites: list[str], verbose: bool = True) -> bool:
    files = [SUITES[s] for s in suites if s in SUITES]
    if not files:
        print("No test files found.")
        return False

    cmd = [
        sys.executable, "-m", "pytest",
        *files,
        "--timeout=30",
        "-q" if not verbose else "-v",
        "--tb=short",
        "--no-header",
    ]

    print(f"\n{'='*60}")
    print(f"AtlasCare Pre-Launch Test Suite")
    print(f"Running: {', '.join(suites)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    success = result.returncode == 0

    print(f"\n{'='*60}")
    if success:
        print("✅ All tests passed — safe to launch the server.")
    else:
        print("❌ Tests failed — fix issues before launching.")
    print(f"{'='*60}\n")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AtlasCare pre-launch test runner.")
    parser.add_argument("--fast",    action="store_true", help="Skip slow edge case tests")
    parser.add_argument("--suite",   choices=list(SUITES.keys()), help="Run one suite only")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if args.suite:
        suites = [args.suite]
    elif args.fast:
        suites = FAST_SUITES
    else:
        suites = list(SUITES.keys())

    ok = run(suites, verbose=args.verbose)
    sys.exit(0 if ok else 1)
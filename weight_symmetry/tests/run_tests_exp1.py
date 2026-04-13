"""
Test runner: weight_symmetry/tests/run_tests_exp1.py
------------------------------------------------------
Runs Exp 1 unit tests via subprocess (avoids torch import in the runner process).

Usage:
    python weight_symmetry/tests/run_tests_exp1.py
    python weight_symmetry/tests/run_tests_exp1.py --fast   (skips smoke test)
"""

import sys
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Skip the end-to-end smoke test")
    args = parser.parse_args()

    helper = os.path.join(
        os.path.dirname(__file__), "helpers", "helper_exp1.py"
    )

    cmd = [sys.executable, helper]
    if args.fast:
        # Monkey-patch: skip smoke test by setting env var
        env = os.environ.copy()
        env["SKIP_SMOKE"] = "1"
    else:
        env = None

    print(f"[run_tests_exp1] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print("[run_tests_exp1] FAILED")
        sys.exit(1)
    else:
        print("[run_tests_exp1] ALL PASSED")

if __name__ == "__main__":
    main()

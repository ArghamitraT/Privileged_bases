"""Subprocess test runner; this process never imports torch.

Conda environment: mrl_env_cuda12
Usage:
    python NonLinearPB/NonLinearPBExperiments/tests/run_tests.py
"""

from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    for helper in ("helper_reproduction.py", "helper_mrl.py", "helper_phase0_rotation.py", "helper_pretrain.py",
                   "helper_tier1.py", "helper_hier_exp1.py"):
        status = subprocess.call([sys.executable, str(Path(__file__).with_name(helper))])
        if status:
            raise SystemExit(status)
    raise SystemExit(0)

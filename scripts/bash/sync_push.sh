#!/bin/bash
# sync_push.sh — sync env, then push.
# Use this when you may have installed new packages and want mrl_env.yml kept
# up to date. For a plain push with no env sync, use git_push.sh directly.
#
# Usage:
#   bash scripts/bash/sync_push.sh "commit message"
#   bash scripts/bash/sync_push.sh          # prompts for message
#
# What it does:
#   1. Detects pip packages installed in active env but missing from mrl_env.yml
#   2. Adds them to the yml pip section
#   3. Calls git_push.sh to commit and push

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
YML="${REPO_ROOT}/env/mrl_env.yml"

echo "============================================================"
echo "sync_push.sh — checking env against mrl_env.yml"
echo "============================================================"

# --- Step 1: detect new pip packages and update yml ---
python3 - "$YML" <<'EOF'
import sys, subprocess, re

yml_path = sys.argv[1]

with open(yml_path) as f:
    content = f.read()

# extract conda-managed package names (to avoid double-listing)
conda_pkgs = set()
for line in content.splitlines():
    line = line.strip()
    if line.startswith("- ") and not line.startswith("- pip:") and ":" not in line:
        name = re.split(r"[=><!]", line[2:].split("::")[-1])[0].strip().lower()
        if name:
            conda_pkgs.add(name)

# extract existing pip packages from yml
pip_in_yml = set()
in_pip = False
for line in content.splitlines():
    if "- pip:" in line:
        in_pip = True
        continue
    if in_pip:
        stripped = line.strip()
        if stripped.startswith("- ") and not stripped.startswith("- -"):
            name = re.split(r"[=><!]", stripped[2:])[0].strip().lower()
            pip_in_yml.add(name)
        elif stripped and not stripped.startswith("#") and not stripped.startswith("-"):
            in_pip = False  # left pip block

# get installed top-level pip packages (not deps of other packages)
result = subprocess.run(
    ["pip", "list", "--not-required", "--format=freeze"],
    capture_output=True, text=True
)
installed = {
    line.split("==")[0].lower()
    for line in result.stdout.strip().splitlines() if line
}

# skip infrastructure and conda-managed packages
skip = {"pip", "setuptools", "wheel", "pkg-resources", "packaging"} | conda_pkgs
new_pkgs = sorted(installed - pip_in_yml - skip)

if not new_pkgs:
    print("  mrl_env.yml is up to date — no new packages detected.")
    sys.exit(0)

print(f"  New packages detected: {new_pkgs}")

# insert new packages into pip section
lines = content.splitlines()
new_lines = []
in_pip = False
inserted = False
for line in lines:
    new_lines.append(line)
    if "- pip:" in line:
        in_pip = True
        continue
    if in_pip and not inserted:
        stripped = line.strip()
        if not stripped.startswith("- ") and stripped:
            # end of pip block — insert before this line
            for pkg in new_pkgs:
                indent = "    "
                new_lines.insert(-1, f"{indent}- {pkg}")
            inserted = True
            in_pip = False

# if pip block is at end of file
if in_pip and not inserted:
    indent = "    "
    for pkg in new_pkgs:
        new_lines.append(f"{indent}- {pkg}")

with open(yml_path, "w") as f:
    f.write("\n".join(new_lines) + "\n")

print(f"  Updated mrl_env.yml with: {new_pkgs}")
EOF

# --- Step 2: git push ---
echo ""
bash "${REPO_ROOT}/scripts/bash/git_push.sh" "$@"

#!/usr/bin/env bash
# =============================================================================
# git_push.sh — stage and push, no env sync.
# Use this for a plain commit+push. If you may have installed new packages
# and want mrl_env.yml updated too, use sync_push.sh instead.
#
# Handles:
#   - New and modified files for each extension (.py .sh .md .yaml .yml .txt)
#   - Deleted tracked files for each extension
#   - Extensions with no matching files (skipped gracefully)
#   - Nothing to commit (exits cleanly)
#
# Usage:
#   bash scripts/bash/git_push.sh "your commit message"   # message as argument
#   bash scripts/bash/git_push.sh                          # will prompt for message
# =============================================================================

set -euo pipefail

# ============================================================
# CONFIG — file extensions to stage
# ============================================================
EXTENSIONS=("py" "sh" "md" "yaml" "yml" "txt")
# ============================================================

# ---- Resolve git root (script may be called from anywhere) ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GIT_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "${GIT_ROOT}" ]]; then
    echo "ERROR: Not inside a git repository."
    exit 1
fi
cd "${GIT_ROOT}"

# ---- Get commit message ----
if [[ $# -ge 1 ]]; then
    COMMIT_MSG="$*"
else
    printf "Commit message: "
    read -r COMMIT_MSG
fi

if [[ -z "${COMMIT_MSG}" ]]; then
    echo "ERROR: Commit message cannot be empty."
    exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "============================================================"
echo "git_push.sh"
echo "  Git root : ${GIT_ROOT}"
echo "  Branch   : ${BRANCH}"
echo "  Message  : ${COMMIT_MSG}"
echo "============================================================"
echo ""

# ---- Stage new and modified files ----
TOTAL_STAGED=0

for ext in "${EXTENSIONS[@]}"; do
    FILE_COUNT=$(find . -path './.git' -prune -o \
                        -type f -name "*.${ext}" -print 2>/dev/null | wc -l)
    FILE_COUNT="${FILE_COUNT// /}"

    if [[ "${FILE_COUNT}" -gt 0 ]]; then
        find . -path './.git' -prune -o \
               -type f -name "*.${ext}" -print0 2>/dev/null \
            | xargs -0 git add --ignore-errors -- 2>/dev/null || true
        echo "  [+] Staged ${FILE_COUNT} .${ext} file(s)"
        TOTAL_STAGED=$(( TOTAL_STAGED + FILE_COUNT ))
    else
        echo "  [ ] No .${ext} files found — skipping"
    fi
done

# ---- Stage deleted tracked files ----
EXT_PATTERN=$(printf '%s|' "${EXTENSIONS[@]}")
EXT_PATTERN="${EXT_PATTERN%|}"

DELETED=$(git ls-files --deleted | grep -E "\.(${EXT_PATTERN})$" || true)
if [[ -n "${DELETED}" ]]; then
    DEL_COUNT=$(echo "${DELETED}" | wc -l)
    echo "${DELETED}" | tr '\n' '\0' | xargs -0 git rm --cached -- 2>/dev/null || true
    echo "  [-] Staged ${DEL_COUNT} deleted file(s)"
else
    echo "  [ ] No deleted files to stage"
fi

echo ""
echo "--- Staged changes ---"
git status --short

# ---- Check if anything is actually staged ----
if git diff --cached --quiet; then
    echo ""
    echo "Nothing staged to commit. Working tree is clean."
    exit 0
fi

# ---- Commit ----
echo ""
git commit -m "${COMMIT_MSG}"

# ---- Push ----
echo ""
echo "Pushing to origin/${BRANCH} ..."
git push origin "${BRANCH}"

echo ""
echo "============================================================"
echo "Done. Pushed ${TOTAL_STAGED} file(s) to origin/${BRANCH}."
echo "============================================================"

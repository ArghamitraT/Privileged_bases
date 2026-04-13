#!/usr/bin/env bash
# open_html.sh — serve exp13 HTML plots via a local HTTP server (VS Code friendly)
#
# Usage:
#   bash scripts/bash/open_html.sh                        # latest run, all HTML
#   bash scripts/bash/open_html.sh <run_dir>              # specific run folder
#   bash scripts/bash/open_html.sh <run_dir> <filter>     # filter by substring
#                                                         #   e.g. "mrl", "3d", "k03"
# Examples:
#   bash scripts/bash/open_html.sh exprmnt_2026_04_06__16_26_32
#   bash scripts/bash/open_html.sh exprmnt_2026_04_06__16_26_32 3d
#   bash scripts/bash/open_html.sh exprmnt_2026_04_06__16_26_32 mrl_k08
#
# In VS Code: the Ports tab will auto-detect the port — click the globe icon.
# Or: Ctrl+Shift+P -> "Simple Browser: Show" -> paste the URL printed below.

set -euo pipefail

RESULTS_ROOT="${HOME}/Mat_embedding_hyperbole/files/results"

# ── Resolve run directory ───────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    RUN_DIR="$1"
    if [[ ! -d "$RUN_DIR" ]]; then
        RUN_DIR="${RESULTS_ROOT}/${1}"
    fi
else
    RUN_DIR=$(ls -dt "${RESULTS_ROOT}"/exprmnt_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "ERROR: no run folders found in ${RESULTS_ROOT}" >&2
        exit 1
    fi
    echo "Using latest run: ${RUN_DIR}"
fi

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: run directory not found: ${RUN_DIR}" >&2
    exit 1
fi

# Search for html_viz/ — may be nested inside a --use-weights subfolder
HTML_DIR=$(find "$RUN_DIR" -type d -name "html_viz" | sort | tail -1)
if [[ -z "$HTML_DIR" ]]; then
    echo "ERROR: no html_viz/ folder found under ${RUN_DIR}" >&2
    echo "       Has exp13 been run with the current code?" >&2
    exit 1
fi
echo "html_viz: ${HTML_DIR}"

# ── Optional filter ─────────────────────────────────────────────────────────
FILTER="${2:-}"

# ── Collect matching files ───────────────────────────────────────────────────
if [[ -n "$FILTER" ]]; then
    mapfile -t FILES < <(find "$HTML_DIR" -maxdepth 1 -name "*.html" | grep "$FILTER" | sort)
else
    mapfile -t FILES < <(find "$HTML_DIR" -maxdepth 1 -name "*.html" | sort)
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No HTML files found (filter='${FILTER}') in ${HTML_DIR}"
    exit 0
fi

echo "Found ${#FILES[@]} file(s):"
for f in "${FILES[@]}"; do echo "  $(basename "$f")"; done

# ── Find a free port ─────────────────────────────────────────────────────────
PORT=""
for port in 8787 8788 8789 8790; do
    if ! lsof -i :"$port" 2>/dev/null | grep -q LISTEN; then
        PORT="$port"
        break
    fi
done

if [[ -z "$PORT" ]]; then
    echo "ERROR: no free port found in 8787-8790." >&2
    exit 1
fi

# ── Start HTTP server in the html_viz directory ──────────────────────────────
cd "$HTML_DIR"
python -m http.server "$PORT" >/dev/null 2>&1 &
PID=$!

echo ""
echo "Server PID : $PID  (port $PORT)"
echo "To stop    : kill $PID"
echo ""

# ── Print URLs ───────────────────────────────────────────────────────────────
echo "VS Code: Ports tab → globe icon, or Ctrl+Shift+P → 'Simple Browser: Show'"
echo ""
if [[ -n "$FILTER" ]]; then
    # Print individual URLs for filtered files
    for f in "${FILES[@]}"; do
        echo "  http://localhost:${PORT}/$(basename "$f")"
    done
else
    echo "  Index : http://localhost:${PORT}/"
    echo ""
    echo "  Individual files:"
    for f in "${FILES[@]}"; do
        echo "    http://localhost:${PORT}/$(basename "$f")"
    done
fi

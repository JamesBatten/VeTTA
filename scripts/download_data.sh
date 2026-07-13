#!/usr/bin/env bash
# Thin wrapper over the vetta dataset downloader CLI.
# Usage: scripts/download_data.sh [--dataset SSA_0.2 ...] [--data-dir data] [--list] [--smoke] [--dry-run]
set -euo pipefail
exec python -c 'import sys; from vetta.download import run_cli; sys.exit(run_cli())' "$@"

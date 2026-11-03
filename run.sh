#!/usr/bin/env bash
# Run a command inside the project's Python environment (WSL).
# Usage: bash run.sh python -m spareparts.train
set -euo pipefail
cd "$(dirname "$0")"
. ~/.venvs/eta/bin/activate
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$@"

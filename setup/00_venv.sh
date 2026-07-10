#!/usr/bin/env bash
# Create a host-side Python virtual environment at $REPRO_ROOT/.venv and install
# the host-side dependency used for model download (the huggingface_hub CLI).
# The headline RULER corpora are built by src/gen_ruler.py (stdlib only).
#
# WHY: on modern distros (Debian/Ubuntu 24.04+, many HPC login nodes) the system
# Python is "externally managed" (PEP 668), so `pip install ...` fails with
#   error: externally-managed-environment
# A venv sidesteps this entirely. All later setup/*.sh auto-activate this venv
# (config.env sources it if present), so you only run this once.
set -euo pipefail
cd "$(dirname "$0")/.."
source ./config.env

if [ ! -d "$VENV" ]; then
  echo "[00] creating venv at $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q "huggingface_hub[cli]"

echo "[00] venv ready: $(python -V) at $VENV"
echo "[00] HF CLI: $(command -v hf || command -v huggingface-cli || echo 'NOT FOUND')"
echo "[00] later setup/*.sh auto-activate this venv via config.env — continue with 01/02/03."

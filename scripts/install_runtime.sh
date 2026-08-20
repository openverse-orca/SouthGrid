#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "error: activate the environment created from environment-unitree.yml first" >&2
    exit 2
fi

python - <<'PY'
import sys

if sys.version_info[:3] != (3, 12, 13):
    raise SystemExit(
        f"error: expected Python 3.12.13, got {sys.version.split()[0]}; "
        "create a fresh environment from environment-unitree.yml"
    )
PY

# Install the resolved pip dependency set. NumPy and SciPy are supplied by Conda,
# and Python 3.12 uses the standard-library argparse module.
python -m pip install --no-deps --require-hashes -r requirements.txt

# --no-deps keeps pip from replacing the validated Conda NumPy/SciPy builds.
python -m pip install --no-deps "orca-gym==26.7.3"

# OrcaLab and OrcaGym use the same validated release version.
python -m pip install --no-deps "orca-lab==26.7.3"

# Install the source packages shipped with this repository.
python -m pip install --no-deps --no-build-isolation ./third_party/lerobot
python -m pip install --no-deps --no-build-isolation ./third_party/televuer
python -m pip install --no-deps --no-build-isolation ./third_party/openpi-client

python scripts/verify_environment.py

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

# requirements.txt is a complete resolved closure and carries its own --index-url /
# --extra-index-url lines, so the CPU-only PyTorch wheels resolve without extra flags.
# --no-deps keeps pip from re-resolving: NumPy and SciPy belong to Conda, and
# params-proto declares the PyPI argparse backport, which on Python 3.12 would shadow
# the standard library module and drop argparse.BooleanOptionalAction.
python -m pip install --no-deps --require-hashes -r requirements.txt

# --no-deps keeps pip from replacing the validated Conda NumPy/SciPy builds.
python -m pip install --no-deps "orca-gym==26.7.3"

# OrcaLab is the desktop application; the PyPI package is the whole thing, so
# there is no separate installer to run. --no-deps applies for the same reason as
# above, plus its pyparsing==3.2.5 pin, which would downgrade the version
# matplotlib is validated against. Every dependency it actually needs is in
# requirements.txt. orca-lab and orca-gym must stay on the same version: orca-lab
# declares an exact orca-gym pin, and a mismatch breaks the gRPC handshake.
python -m pip install --no-deps "orca-lab==26.7.3"

# Repository-owned sources are installed last. No developer-machine path or
# unpinned Git checkout is consulted.
python -m pip install --no-deps --no-build-isolation ./third_party/lerobot
python -m pip install --no-deps --no-build-isolation ./third_party/televuer
python -m pip install --no-deps --no-build-isolation ./third_party/openpi-client

python scripts/verify_environment.py

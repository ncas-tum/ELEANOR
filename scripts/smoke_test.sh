#!/usr/bin/env bash
# Install eleanor-ncas from the local package index (scripts/build.sh) the way
# users will, in clean containers, and check the installed builds:
#   pip install eleanor-ncas --extra-index-url <index>/<variant>
#   uv add eleanor-ncas --index <index>/<variant>
# Options: VARIANTS="cpu cu126", PY=3.14, GPU=--gpus=all
set -euo pipefail

PY=${PY:-3.11}
GPU=${GPU:-}
NET=eleanor-smoke

# Serve dist/index from a container on a private network, so the test
# containers reach it the same way on Linux, macOS and Docker Desktop/WSL
docker network create "$NET" > /dev/null 2>&1 || true
docker run -d --rm --name eleanor-index --network "$NET" \
  -v "$PWD/dist/index:/index:ro" python:3.12-slim \
  python -m http.server 8000 --directory /index > /dev/null
trap 'docker stop eleanor-index > /dev/null; docker network rm "$NET" > /dev/null' EXIT
sleep 2

for variant in ${VARIANTS:-cpu cu126 cu130}; do
  echo "=== $variant, Python $PY"
  # gcc: some dependencies (e.g. expelliarmus via tonic) have no wheel for
  # every Python version and compile on install
  # shellcheck disable=SC2086
  # The test index is plain HTTP (the real one is HTTPS), so trust its host
  docker run --rm $GPU --network "$NET" -e UV_HTTP_TIMEOUT=300 \
    -e PIP_TRUSTED_HOST=eleanor-index -e UV_INSECURE_HOST=eleanor-index \
    -v "$PWD/scripts/check_install.py:/check_install.py:ro" \
    ghcr.io/astral-sh/uv:bookworm-slim sh -ec "
      apt-get update -qq && apt-get install -y -qq gcc > /dev/null
      INDEX=http://eleanor-index:8000/$variant

      echo '--- pip install eleanor-ncas --extra-index-url \$INDEX'
      uv venv -q --seed --python $PY /tmp/pip-env
      /tmp/pip-env/bin/python -m pip install -q --disable-pip-version-check \
        eleanor-ncas --extra-index-url \$INDEX
      /tmp/pip-env/bin/python /check_install.py $variant

      echo '--- uv add eleanor-ncas --index \$INDEX'
      uv init -q --bare --python $PY /tmp/project && cd /tmp/project
      uv add -q eleanor-ncas --index \$INDEX
      uv run python /check_install.py $variant
    "
done

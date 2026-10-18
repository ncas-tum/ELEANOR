#!/usr/bin/env bash
# Local equivalent of .github/workflows/wheels.yml:
#   1. build one wheel per hardware variant              -> dist/raw/<variant>/
#   2. pin torch/torchvision/jax for that variant        -> dist/wheels/
#   3. generate the per-variant package index            -> dist/index/<variant>/
# Pick variants with e.g. VARIANTS="cpu cu126" scripts/build.sh (VARIANTS="" for
# none). Linux wheels build in Docker; on an Apple Silicon Mac,
# MACOS_VARIANTS="cpu mps" also builds the macOS wheels natively (Docker can't).
set -euo pipefail

BASE=$(uvx --from setuptools-scm python -m setuptools_scm)
BASE=${BASE%%+*}   # drop the git suffix (+g1234abc): the variant takes its place
PY=3.11            # the wheel is abi3, so the build Python only sets the minimum (3.11+)

#        variant  image                                    uv extra  CUDA architectures
VARIANT_SPECS="
cpu      almalinux:8                              cpu       -
cu126    nvidia/cuda:12.6.3-devel-rockylinux8     cuda12    7.0;7.5;8.0;8.6;8.9;9.0
cu130    nvidia/cuda:13.0.1-devel-rockylinux8     cuda13    7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX
"

build_variant() {
  local variant=$1 image=$2 extra=$3 archs=$4
  [[ $archs == - ]] && archs=""
  echo "=== $variant ($BASE+$variant)"

  docker build -f Dockerfile \
    --build-arg BASE_IMAGE="$image" \
    --build-arg UV_EXTRA="$extra" \
    --build-arg TORCH_CUDA_ARCH_LIST="$archs" \
    --build-arg PYTHON_VERSION="$PY" \
    --build-arg ELEANOR_VERSION="$BASE+$variant" \
    --build-arg MAX_JOBS=4 \
    --target export --output type=local,dest="dist/raw/$variant" .

  pin_variant "$variant" "$extra" dist/raw/"$variant"/*.whl
}

# Pin the exact torch/torchvision builds the variant was compiled against.
# They keep uv's markers, so the Linux and macOS cpu wheels match.
pin_variant() {
  local variant=$1 extra=$2 wheel=$3
  uv export --frozen --extra "$extra" --no-hashes --no-emit-project > dist/raw/pins-"$extra".txt
  python3 scripts/make_variant_wheel.py "$wheel" \
    --variant "$variant" --extra "$extra" --pins dist/raw/pins-"$extra".txt -o dist/wheels
}

# macOS arm64 wheel, built natively in its own venv (the project's .venv is
# left alone). macOS 14 is the minimum of torch's own arm64 wheels.
build_macos() {
  local variant=$1 extra=$2
  [[ $(uname -sm) == "Darwin arm64" ]] || { echo "error: macOS wheels need an Apple Silicon Mac"; exit 1; }
  echo "=== macOS $variant ($BASE+$variant)"
  local venv=dist/raw/macos-$variant-venv
  UV_PROJECT_ENVIRONMENT=$venv UV_PYTHON=$PY \
    uv sync --frozen --extra "$extra" --no-default-groups --group build --no-install-project
  MACOSX_DEPLOYMENT_TARGET=14.0 SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ELEANOR_NCAS="$BASE+$variant" \
    uv build --wheel --no-build-isolation --python "$venv/bin/python" --out-dir dist/raw/macos-"$variant"
  pin_variant "$variant" "$extra" dist/raw/macos-"$variant"/*.whl
}

rm -rf dist/raw dist/wheels dist/index
mkdir -p dist/raw
while read -r variant image extra archs; do
  [[ -z $variant ]] && continue
  [[ " ${VARIANTS-cpu cu126 cu130} " == *" $variant "* ]] || continue
  build_variant "$variant" "$image" "$extra" "$archs"
done <<< "$VARIANT_SPECS"
for variant in ${MACOS_VARIANTS:-}; do
  case $variant in
    cpu) build_macos cpu cpu ;;
    mps) build_macos mps mps ;;
    *) echo "error: unknown macOS variant $variant"; exit 1 ;;
  esac
done

uvx abi3audit --strict dist/wheels/*.whl
python3 scripts/build_index.py -o dist/index dist/wheels/*.whl

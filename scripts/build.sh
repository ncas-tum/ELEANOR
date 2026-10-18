#!/usr/bin/env bash
# Local equivalent of .github/workflows/wheels.yml:
#   1. build one wheel per hardware variant in Docker   -> dist/raw/<variant>/
#   2. pin torch/torchvision/jax for that variant        -> dist/wheels/
#   3. generate the per-variant package index            -> dist/index/<variant>/
# Pick variants with e.g. VARIANTS="cpu cu126" scripts/build.sh
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

  # The exact torch/torchvision builds this variant was compiled against
  local pins
  pins=$(uv export --frozen --extra "$extra" --no-hashes --no-emit-project \
         | grep -E "^(torch|torchvision)==[^ ]*\+$variant( |$)" | cut -d' ' -f1 | sed 's/^/--pin /')
  # shellcheck disable=SC2086
  python3 scripts/make_variant_wheel.py dist/raw/"$variant"/*.whl \
    --variant "$variant" --extra "$extra" $pins -o dist/wheels
}

rm -rf dist/raw dist/wheels dist/index
while read -r variant image extra archs; do
  [[ -z $variant ]] && continue
  [[ " ${VARIANTS:-cpu cu126 cu130} " == *" $variant "* ]] || continue
  build_variant "$variant" "$image" "$extra" "$archs"
done <<< "$VARIANT_SPECS"

uvx abi3audit --strict dist/wheels/*.whl
python3 scripts/build_index.py -o dist/index dist/wheels/*.whl

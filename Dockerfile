ARG BASE_IMAGE=nvidia/cuda:12.6.3-devel-rockylinux8
FROM ${BASE_IMAGE} AS build

RUN dnf install -y gcc-toolset-12 && dnf clean all
ENV PATH="/opt/rh/gcc-toolset-12/root/usr/bin:${PATH}" \
    CC=gcc \
    CXX=g++ \
    CUDAHOSTCXX=g++

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

WORKDIR /workspace

ARG PYTHON_VERSION=3.14
ARG MAX_JOBS=8
ARG UV_EXTRA=cuda12
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON=${PYTHON_VERSION} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS}

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=LICENSE.txt,target=LICENSE.txt \
    mkdir -p eleanor && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ELEANOR=0.0.0 \
    uv sync --frozen --extra ${UV_EXTRA} --no-install-project

COPY . /workspace

ARG ELEANOR_VERSION
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ELEANOR=${ELEANOR_VERSION} \
    VIRTUAL_ENV=/workspace/.venv \
    PATH="/workspace/.venv/bin:${PATH}"

RUN --mount=type=cache,target=/root/.cache/uv \
    test -n "${ELEANOR_VERSION}" || { echo "ERROR: pass --build-arg ELEANOR_VERSION=..."; exit 1; } && \
    uv pip install --python /workspace/.venv/bin/python \
      scikit-build-core setuptools-scm cmake ninja && \
    uv build --wheel --no-build-isolation \
      --python /workspace/.venv/bin/python \
      --out-dir /dist_raw

# Verify the CUDA module contains every architecture from TORCH_CUDA_ARCH_LIST
RUN set -e; \
    rm -rf /tmp/whl; \
    python -m zipfile -e /dist_raw/*.whl /tmp/whl; \
    so=$(find /tmp/whl -name '_C_cu*.so' | head -n1); \
    if [ -z "$so" ]; then echo "No CUDA module found (CPU build): skipping arch check"; exit 0; fi; \
    echo "Checking $so"; \
    elfs=$(cuobjdump --list-elf "$so"); \
    ptx=$(cuobjdump --list-ptx "$so" 2>/dev/null || true); \
    echo "$elfs"; \
    echo "$ptx"; \
    missing=""; \
    for arch in $(echo "${TORCH_CUDA_ARCH_LIST}" | tr ';' ' '); do \
      sm=$(echo "$arch" | sed 's/+PTX//; s/\.//'); \
      echo "$elfs" | grep -q "sm_${sm}\." || missing="$missing sm_${sm}"; \
      case "$arch" in \
        *+PTX) echo "$ptx" | grep -q "sm_${sm}\." || missing="$missing PTX-${sm}" ;; \
      esac; \
    done; \
    if [ -n "$missing" ]; then echo "ERROR: missing CUDA architectures:$missing"; exit 1; fi; \
    echo "All CUDA architectures present: ${TORCH_CUDA_ARCH_LIST}"; \
    rm -rf /tmp/whl

RUN --mount=type=cache,target=/root/.cache/uv \
    uvx --with patchelf auditwheel repair /dist_raw/*.whl \
      --plat manylinux_2_28_x86_64 \
      -w /dist \
      --exclude 'libtorch*' \
      --exclude 'libc10*' \
      --exclude 'libgomp*' \
      --exclude 'libcudart.so*' \
      --exclude 'libcuda.so*'

FROM scratch AS export
COPY --from=build /dist /
"""Check an installed eleanor-ncas against the hardware variant it came from.

Used by the smoke tests after `pip install eleanor-ncas --extra-index-url .../<variant>`
or `uv add eleanor-ncas --index .../<variant>`. Works without a GPU: it checks
which builds were installed and that the extension loads, not that kernels run.

Usage:
    python scripts/check_install.py cu126
"""

import importlib.metadata as md
import sys

JAX_PLUGIN = {
    "cpu": None,
    "cu126": "jax-cuda12-plugin",
    "cu130": "jax-cuda13-plugin",
    "mps": "jax-mps",
}
# mps uses the CPU extension (the kernels have no Metal implementation)
EXTENSION = {"mps": "cpu"}


def installed(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main():
    variant = sys.argv[1]
    import torch

    import eleanor
    from eleanor.torch import models

    plugins = {p: installed(p) for p in filter(None, JAX_PLUGIN.values())}
    print(
        f"eleanor-ncas {md.version('eleanor-ncas')} | eleanor.__version__ {eleanor.__version__}"
    )
    print(f"torch {torch.__version__} | loaded {models._C.__name__}")
    print(f"jax {installed('jax')} | plugins {plugins}")
    print(f"GPU available: {torch.cuda.is_available()}")

    errors = []
    if (
        md.version("eleanor-ncas") != f"{eleanor.__version__}"
        or eleanor.__version__ == "0.0.dev0"
    ):
        errors.append("eleanor/_version.py is missing or does not match the wheel")
    if not md.version("eleanor-ncas").endswith(f"+{variant}"):
        errors.append(f"eleanor-ncas is not the +{variant} build")
    # torch's macOS builds have no local version (2.14.1, not 2.14.1+cpu)
    torch_local = "" if sys.platform == "darwin" else variant
    if torch.__version__.partition("+")[2] != torch_local:
        errors.append(
            f"torch {torch.__version__} is not the "
            f"{'+' + torch_local if torch_local else 'plain'} build"
        )
    extension = f"_C_{EXTENSION.get(variant, variant)}"
    if models._C.__name__.rsplit(".", 1)[-1] != extension:
        errors.append(f"loaded {models._C.__name__} instead of {extension}")
    for plugin, version in plugins.items():
        if plugin == JAX_PLUGIN[variant] and version is None:
            errors.append(f"{plugin} is missing")
        elif plugin != JAX_PLUGIN[variant] and version is not None:
            errors.append(f"{plugin} should not be installed")
    for op in ("bruno", "heracles"):
        if not hasattr(torch.ops.eleanor, op):
            errors.append(f"torch.ops.eleanor.{op} is not registered")

    if errors:
        sys.exit("FAILED:\n  " + "\n  ".join(errors))
    print("OK")


if __name__ == "__main__":
    main()

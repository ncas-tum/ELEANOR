import importlib
import pkgutil
import re
import sys

import torch


def _available_builds():
    """Suffixes of the compiled extensions shipped in this package, e.g. ["cpu", "cu126"].

    Modules already in sys.modules count too, e.g. the stub tests/conftest.py
    installs when no extension is built.
    """
    shipped = {m.name for m in pkgutil.iter_modules(__path__)}
    loaded = {
        name.rpartition(".")[2]
        for name in sys.modules
        if name.startswith(f"{__name__}._C_")
    }
    return sorted(
        n.removeprefix("_C_") for n in shipped | loaded if n.startswith("_C_")
    )


def _select_build(cuda_version, available):
    """Pick the extension that matches torch's CUDA version.

    A wheel can ship several builds (_C_cpu, _C_cu126, _C_cu130). CUDA keeps
    compatibility within a major version, so a torch built for CUDA 12.8 can
    use the cu126 build. Prefer an exact match, then the newest build of the
    same major version that is not newer than torch's CUDA, then any build of
    the same major version.
    """
    if cuda_version is None:
        return "cpu" if "cpu" in available else None

    major, minor = (int(x) for x in cuda_version.split(".")[:2])
    same_major = []
    for build in available:
        match = re.fullmatch(r"cu(\d+?)(\d)", build)
        if match and int(match.group(1)) == major:
            same_major.append((int(match.group(2)), build))
    if not same_major:
        return None

    exact = [b for m, b in same_major if m == minor]
    older = [b for m, b in sorted(same_major) if m <= minor]
    return (exact or older[-1:] or [max(same_major)[1]])[0]


def _load_extension():
    cuda_version = torch.version.cuda
    available = _available_builds()
    build = _select_build(cuda_version, available)
    if build is None:
        wanted = "CPU" if cuda_version is None else f"CUDA {cuda_version}"
        raise ImportError(
            f"No compiled ELEANOR extension for {wanted} (torch {torch.__version__}). "
            f"Available builds: {available or 'none'}. Install a torch build that "
            f"matches one of them, or rebuild ELEANOR against this torch."
        )
    return importlib.import_module(f"{__name__}._C_{build}")


_C = _load_extension()
from ._bruno import Bruno
from ._felif import FeLIF
from ._heracles import Heracles

__all__ = ["Bruno", "FeLIF", "Heracles"]

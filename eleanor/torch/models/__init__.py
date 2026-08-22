import importlib

import torch


def _load_extension():
    cuda_version = torch.version.cuda
    if cuda_version is None:
        suffix = "cpu"
    else:
        major, minor = cuda_version.split(".")[:2]
        suffix = f"cu{major}{minor}"

    module_name = f"{__name__}._C_{suffix}"
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        available = [n.replace("_C_", "") for n in dir() if n.startswith("_C_")]
        raise ImportError(
            f"No compiled extension found for CUDA {cuda_version} "
            f"(tried {module_name}). Available builds: {available}. "
            f"Rebuild with CUDA_VERSION set to a matching version."
        ) from e


_C = _load_extension()
from ._bruno import Bruno
from ._felif import FeLIF
from ._heracles import Heracles

__all__ = ["Bruno", "FeLIF", "Heracles"]

import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)
from eleanor._cuda_version import cuda_local_scheme

LIBRARY_NAME = "eleanor"

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1" if torch.cuda.is_available() else "0") == "1"
    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None

    extension = CUDAExtension if use_cuda else CppExtension

    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fopenmp",
            "-fdiagnostics-color=always",
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
            '-U', 'Py_LIMITED_API'
        ],
    }

    extra_link_args = []
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.relpath(__file__))
    extensions_dir = os.path.join(this_dir, LIBRARY_NAME, "torch", "models", "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "**/*.cpp"), recursive=True))
    if use_cuda:
        sources += glob.glob(os.path.join(extensions_dir, "**/*.cu"), recursive=True)

    return [
        extension(
            f"{LIBRARY_NAME}.torch.models._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=False,
        )
    ]


setup(
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    options={},
    use_scm_version={"local_scheme": cuda_local_scheme}
)

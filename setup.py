import os
import glob

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    CUDA_HOME,
    CppExtension,
    CUDAExtension,
    BuildExtension,
)

LIBRARY_NAME = "eleanor"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1" if torch.cuda.is_available() else "0") == "1"
    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    print("Compiling")
    if debug_mode:
        print("Compiling in debug mode")

    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fopenmp",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }

    extra_link_args = []
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.relpath(__file__))
    extensions_dir = os.path.join(this_dir, LIBRARY_NAME, "models", "torch", "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "**/*.cpp"), recursive=True))
    if use_cuda:
        sources += glob.glob(os.path.join(extensions_dir, "**/*.cu"), recursive=True)

    return [
        extension(
            f"{LIBRARY_NAME}.models.torch._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

setup(
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)

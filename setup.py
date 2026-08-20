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

LIBRARY_NAME = "eleanor"


def cuda_local_scheme(version):
    cuda = os.environ.get("CUDA_VERSION")
    if cuda and cuda != "cpu":
        return f"+cu{cuda}"
    return ""


def install_requires():
    jax_extra = os.environ.get("JAX_EXTRA", "cpu")
    jax_dep = "jax>=0.10.0" if jax_extra == "cpu" else f"jax[{jax_extra}]>=0.10.0"
    return [
        "boilerplot@git+https://github.com/fehlings/boilerplot.git",
        "datasets>=4.8.5",
        "equinox>=0.13.8",
        "flwr-datasets>=0.6.0",
        "ipykernel>=7.2.0",
        "ipywidgets>=8.1.8",
        "jax-tqdm>=0.4.0",
        "matplotlib>=3.10.9",
        "mpi4py>=4.1.2",
        "optax>=0.2.8",
        "optuna>=4.8.0",
        "optuna-dashboard>=0.20.0",
        "orbax-checkpoint>=0.11.40",
        "pandas>=3.0.3",
        "scikit-learn>=1.8.0",
        "seaborn>=0.13.2",
        "snntorch>=0.9.4",
        "spyx>=0.1.20",
        "tensorboard>=2.20.0",
        "tonic>=1.4.3",
        "torch>=2.12.0",
        "torchvision>=0.27.0",
        "tqdm>=4.67.3",
        "tyro",
        jax_dep,
    ]


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1" if torch.cuda.is_available() else "0") == "1"
    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    cuda_tag = os.environ.get("CUDA_VERSION", "cpu")
    cuda_tag = cuda_tag.replace(".", "") if cuda_tag != "cpu" else "cpu"
    ext_name = f"{LIBRARY_NAME}.torch.models._C_cu{cuda_tag}" if cuda_tag != "cpu" else f"{LIBRARY_NAME}.torch.models._C_cpu"

    extension = CUDAExtension if use_cuda else CppExtension

    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fopenmp",
            "-fdiagnostics-color=always",
        ],
        "nvcc": ["-O3" if not debug_mode else "-O0", "-U", "Py_LIMITED_API"],
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
            ext_name,
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
    use_scm_version={"local_scheme": cuda_local_scheme},
    install_requires=install_requires(),
)

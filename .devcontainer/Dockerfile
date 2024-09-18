# FROM ghcr.io/nvidia/jax:jax-2024-09-10
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04
# FROM ghcr.io/nvidia/jax

RUN apt update && apt install -y git python3 python3-pip python-is-python3 nvtop ffmpeg && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# Generic python libraries
RUN pip install matplotlib seaborn pandas tqdm scikit-learn ipykernel ipywidgets

# JAX
RUN pip install equinox git+https://github.com/google/orbax/#subdirectory=checkpoint aqtp

# SNN related libs
# RUN pip install spyx tonic torch torchvision torchaudio
RUN pip install git+https://github.com/PGI15/snnax

# Dev libraries
RUN pip install pre-commit black black[jupyter] isort flake8-black flake8-isort flake8-bugbear sphinx>=4.0 sphinx-book-theme myst_nb
# flake8-print flake8-bandit flake8-pylint flake8-builtins flake8-spellcheck flake8-pytest-style flake8-fastapi

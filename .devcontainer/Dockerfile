FROM ghcr.io/nvidia/jax:equinox

# RUN apt update && apt install nvtop libegl1 libnss3 libxcomposite1 libxdamage1 libxtst6 libxkbfile1 && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*
RUN apt update && apt install nvtop && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# Generic python libraries
RUN pip install matplotlib seaborn pandas tqdm scikit-learn ipykernel ipywidgets

# JAX
RUN pip install git+https://github.com/google/orbax/#subdirectory=checkpoint aqtp

# SNN related libs
RUN pip install spyx tonic torch torchvision torchaudio

# Dev libraries
RUN pip install pre-commit black black[jupyter] isort flake8-black flake8-isort flake8-print flake8-bandit flake8-pylint flake8-builtins flake8-spellcheck flake8-pytest-style flake8-fastapi

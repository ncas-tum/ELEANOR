FROM ghcr.io/nvidia/jax:equinox

RUN apt update && apt install nvtop && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*
RUN pip install matplotlib seaborn pandas tqdm spyx scikit-learn ipykernel ipywidgets tonic torch torchvision torchaudio
RUN pip install pre-commit black isort flake8-black flake8-isort flake8-print flake8-bandit flake8-pylint flake8-builtins flake8-spellcheck flake8-pytest-style flake8-fastapi

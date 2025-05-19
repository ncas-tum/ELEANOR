FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y git python3 python3-pip python-is-python3 python3-tk nvtop ffmpeg locales locales-all mpich && rm -rf /var/cache/apt/archives /var/lib/apt/lists/* && pip install --no-cache-dir -U pip

# Generic python libraries
RUN pip install --no-cache-dir matplotlib seaborn pandas tqdm scikit-learn ipykernel ipywidgets optuna optuna-dashboard

# PyTorch & JAX
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir jax[cuda12]==0.6.0

# SNN
RUN pip install --no-cache-dir git+https://github.com/jeshraghian/snntorch.git git+https://github.com/PGI15/snnax tonic

# Dev libraries
RUN pip install --no-cache-dir pre-commit black black[jupyter] isort flake8-black flake8-isort flake8-bugbear sphinx>=4.0 sphinx-book-theme myst_nb git+https://github.com/fehlings/boilerplot.git
#RUN wget https://go.dev/dl/go1.24.2.linux-amd64.tar.gz
#RUN rm -rf /usr/local/go && tar -C /usr/local -xzf go1.24.2.linux-amd64.tar.gz
#ENV PATH="$PATH:/root/go/bin:/usr/local/go/bin"
#RUN go install github.com/google/pprof@latest

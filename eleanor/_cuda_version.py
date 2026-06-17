import os

def cuda_local_scheme(version):
    cuda = os.environ.get("CUDA_VERSION")
    return f"+cu{cuda}" if cuda else ""
import os

def cuda_local_scheme(version):
    cuda = os.environ.get("CUDA_VERSION")
    if cuda and cuda != "cpu":
        return f"+cu{cuda}"
    return ""
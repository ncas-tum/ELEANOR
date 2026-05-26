PyTorch backend
===============

The PyTorch implementation of ELEANOR lives under :mod:`eleanor.torch`.
Neurons subclass ``snntorch.SpikingNeuron`` and integrate with standard
PyTorch training loops, autograd, and optional C++/CUDA kernels for Bruno
and Heracles.

.. toctree::
   :maxdepth: 1

   models
   variability
   surrogate

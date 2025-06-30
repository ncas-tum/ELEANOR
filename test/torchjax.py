import unittest

import numpy as np
import torch
import jax.numpy as jnp
import jax.random as jrand
from torch.testing._internal.common_utils import TestCase

from eleanor.models.jax import Bruno as Brunojax
from eleanor.models.jax import FeLIF as FeLIFjax
from eleanor.models.jax import Heracles as Heraclesjax
from eleanor.models.torch import Bruno as BrunoTorch
from eleanor.models.torch import FeLIF as FeLIFTorch
from eleanor.models.torch import Heracles as HeraclesTorch


class TestNeuronBase(TestCase):

    def sample_inputs(self):

        def make_array(*size):
            return np.random.randn(*size).astype(np.float32)

        return [make_array(3), make_array(256)]

    def _test_compare_tensors(self, torch_tensor, jax_array):
        # Check shapes
        self.assertEqual(
            torch_tensor.shape,
            jax_array.shape,
            msg=f"Shape mismatch: torch_tensor shape {torch_tensor.shape} != jax_array shape {jax_array.shape}",
        )

        # Convert JAX array to numpy
        jax_np = np.array(jax_array)

        # Create a PyTorch tensor from JAX array with the same device and dtype as the original PyTorch tensor
        jax_tensor = torch.from_numpy(jax_np).to(device=torch_tensor.device)

        # Compare the tensors
        torch.testing.assert_close(torch_tensor, jax_tensor, check_dtype=False)

    def _test_correctness(self, torch_model, jax_model, device):
        torch_model.to(device)
        samples = self.sample_inputs()

        for sample in samples:
            sample_torch = torch.asarray(sample, device=device, requires_grad=False)
            pol, mem = torch_model.reset_mem()
            spk, pol, mem = torch_model(sample_torch, pol, mem)

            sample_jax = jnp.asarray(sample)
            state = jax_model.init_state(sample_jax.shape, key=jrand.key(0))
            state, output = jax_model(state, sample_jax)

            spk_jax = np.asarray(output[0])
            mem_jax = np.asarray(output[1])
            pol_jax = np.asarray(output[2])

            self._test_compare_tensors(spk, spk_jax)
            self._test_compare_tensors(mem, mem_jax)
            self._test_compare_tensors(pol, pol_jax)


class TestFeLIF(TestNeuronBase):

    def test_correctness_cpu(self):
        device = "cpu"
        params = [np.random.rand() for _ in range(10)]
        torch_model = FeLIFTorch(*params, output=True).to(device)
        jax_model = FeLIFjax(*params, key=jrand.key(0))

        self._test_correctness(torch_model, jax_model, device)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        device = "cuda"
        params = [np.random.rand() for _ in range(10)]
        torch_model = FeLIFTorch(*params, output=True).to(device)
        jax_model = FeLIFjax(*params, key=jrand.key(0))

        self._test_correctness(torch_model, jax_model, device)


class TestHeracles(TestNeuronBase):

    def test_correctness_cpu(self):
        device = "cpu"
        params = [np.random.rand() for _ in range(18)]
        torch_model = HeraclesTorch(*params, output=True).to(device)
        jax_model = Heraclesjax(*params, key=jrand.key(0))

        self._test_correctness(torch_model, jax_model, device)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        device = "cuda"
        params = [np.random.rand() for _ in range(18)]
        torch_model = HeraclesTorch(*params, output=True).to(device)
        jax_model = Heraclesjax(*params, key=jrand.key(0))

        self._test_correctness(torch_model, jax_model, device)


class TestBruno(TestNeuronBase):

    def test_correctness_cpu(self):
        device = "cpu"
        params = {
            # "A": 30e-12 * np.random.rand()
        }
        torch_model = BrunoTorch(**params, output=True).to(device)
        jax_model = Brunojax(**params, key=jrand.key(0))

        self._test_correctness(torch_model, jax_model, device)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        device = "cuda"
        params = {
            # "A": 30e-12 * np.random.rand()
        }
        torch_model = BrunoTorch(**params, output=True).to(device)
        torch_model_cpu = BrunoTorch(**params, output=True)

        samples = self.sample_inputs()

        for sample in samples:
            sample_torch = torch.asarray(sample, device=device, requires_grad=False)
            pol, mem = torch_model.reset_mem()
            spk_cuda, pol_cuda, mem_cuda = torch_model(sample_torch, pol, mem)

            sample_torch_cpu = torch.asarray(sample, device="cpu", requires_grad=False)
            pol, mem = torch_model_cpu.reset_mem()
            spk_cpu, pol_cpu, mem_cpu = torch_model_cpu(sample_torch_cpu, pol, mem)

            torch.testing.assert_close(spk_cuda.cpu(), spk_cpu, check_dtype=False)
            torch.testing.assert_close(pol_cuda.cpu(), pol_cpu, check_dtype=False)
            torch.testing.assert_close(mem_cuda.cpu(), mem_cpu, check_dtype=False)


if __name__ == "__main__":
    unittest.main()

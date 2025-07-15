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
            msg=f"Shape mismatch: torch_tensor shape {torch_tensor.shape} != \
            jax_array shape {jax_array.shape}",
        )

        # Convert JAX array to numpy
        jax_np = np.array(jax_array)

        # Create a PyTorch tensor from JAX array with the same device and
        # dtype as the original PyTorch tensor
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
        params = {
            "A": 25e-12 * (1 + 0.1 * np.random.randn()),
            "t_fe": 9.8e-9 * (1 + 0.1 * np.random.randn()),
            "eps_fe": 70 * (1 + 0.1 * np.random.randn()),
            "eps_depl": 3.6 * (1 + 0.1 * np.random.randn()),
            "q_fix_depl": 945e-4 * (1 + 0.1 * np.random.randn()),
            "n_depl": 1.4e28 * (1 + 0.1 * np.random.randn()),
            "e_off": 2e7 * (1 + 0.1 * np.random.randn()),
            "temp": 294 * (1 + 0.1 * np.random.randn()),
            "w_b": 1.05 * (1 + 0.1 * np.random.randn()),
            "d_e": 7.5e-9 * (1 + 0.1 * np.random.randn()),
            "P_s": 27e-2 * (1 + 0.1 * np.random.randn()),
            "I_0": 1e-4 * (1 + 0.1 * np.random.randn()),
            "V_t": 0.32 * (1 + 0.1 * np.random.randn()),
            "C_par": 15e-15 * (1 + 0.1 * np.random.randn()),
            "I_dsc": 10e-12 * (1 + 0.1 * np.random.randn()),
            "threshold": 3.5 * (1 + 0.1 * np.random.randn()),
            "dt": 1e-3 * (1 + 0.1 * np.random.randn()),
            "paramsScale": 1e12 * (1 + 0.1 * np.random.randn()),
        }
        torch_model = HeraclesTorch(**params, output=True).to(device)
        jax_model = Heraclesjax(**params, key=jrand.key(0))

        self._test_correctness(torch_model, jax_model, device)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        device = "cuda"
        params = {
            "A": 25e-12 * (1 + 0.1 * np.random.randn()),
            "t_fe": 9.8e-9 * (1 + 0.1 * np.random.randn()),
            "eps_fe": 70 * (1 + 0.1 * np.random.randn()),
            "eps_depl": 3.6 * (1 + 0.1 * np.random.randn()),
            "q_fix_depl": 945e-4 * (1 + 0.1 * np.random.randn()),
            "n_depl": 1.4e28 * (1 + 0.1 * np.random.randn()),
            "e_off": 2e7 * (1 + 0.1 * np.random.randn()),
            "temp": 294 * (1 + 0.1 * np.random.randn()),
            "w_b": 1.05 * (1 + 0.1 * np.random.randn()),
            "d_e": 7.5e-9 * (1 + 0.1 * np.random.randn()),
            "P_s": 27e-2 * (1 + 0.1 * np.random.randn()),
            "I_0": 1e-4 * (1 + 0.1 * np.random.randn()),
            "V_t": 0.32 * (1 + 0.1 * np.random.randn()),
            "C_par": 15e-15 * (1 + 0.1 * np.random.randn()),
            "I_dsc": 10e-12 * (1 + 0.1 * np.random.randn()),
            "threshold": 3.5 * (1 + 0.1 * np.random.randn()),
            "dt": 1e-3 * (1 + 0.1 * np.random.randn()),
            "paramsScale": 1e12 * (1 + 0.1 * np.random.randn()),
        }
        torch_model = HeraclesTorch(**params, output=True).to(device)
        jax_model = Heraclesjax(**params, key=jrand.key(0))

        self._test_correctness(torch_model, jax_model, device)


class TestBruno(TestNeuronBase):

    def test_correctness_cpu(self):
        device = "cpu"
        params = {
            "A": 25e-12 * (1 + 0.1 * np.random.randn()),
            "t_hzo": 10e-9 * (1 + 0.1 * np.random.randn()),
            "t_int": 1.375e-9 * (1 + 0.1 * np.random.randn()),
            "eps_hzo": 25.2 * (1 + 0.1 * np.random.randn()),
            "eps_int": 33 * (1 + 0.1 * np.random.randn()),
            "E_a": 12.7e8 * (1 + 0.1 * np.random.randn()),
            "P_s": 22e-2 * (1 + 0.1 * np.random.randn()),
            "tau_0": 1e-13 * (1 + 0.1 * np.random.randn()),
            "I_0": 1e-4 * (1 + 0.1 * np.random.randn()),
            "V_t": 0.32 * (1 + 0.1 * np.random.randn()),
            "C_par": 15e-15 * (1 + 0.1 * np.random.randn()),
            "alpha": 1.3 * (1 + 0.1 * np.random.randn()),
            "soft_E": 5e-6 * (1 + 0.1 * np.random.randn()),
            "I_dsc": 10e-12 * (1 + 0.1 * np.random.randn()),
            "threshold": 2.5 * (1 + 0.1 * np.random.randn()),
            "dt": 1e-3 * (1 + 0.1 * np.random.randn()),
            "paramsScale": 1e12 * (1 + 0.1 * np.random.randn()),
        }
        torch_model = BrunoTorch(**params, output=True).to(device)
        jax_model = Brunojax(**params, key=jrand.key(0))

        self._test_correctness(torch_model, jax_model, device)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        device = "cuda"
        params = {
            "A": 25e-12 * (1 + 0.1 * np.random.randn()),
            "t_hzo": 10e-9 * (1 + 0.1 * np.random.randn()),
            "t_int": 1.375e-9 * (1 + 0.1 * np.random.randn()),
            "eps_hzo": 25.2 * (1 + 0.1 * np.random.randn()),
            "eps_int": 33 * (1 + 0.1 * np.random.randn()),
            "E_a": 12.7e8 * (1 + 0.1 * np.random.randn()),
            "P_s": 22e-2 * (1 + 0.1 * np.random.randn()),
            "tau_0": 1e-13 * (1 + 0.1 * np.random.randn()),
            "I_0": 1e-4 * (1 + 0.1 * np.random.randn()),
            "V_t": 0.32 * (1 + 0.1 * np.random.randn()),
            "C_par": 15e-15 * (1 + 0.1 * np.random.randn()),
            "alpha": 1.3 * (1 + 0.1 * np.random.randn()),
            "soft_E": 5e-6 * (1 + 0.1 * np.random.randn()),
            "I_dsc": 10e-12 * (1 + 0.1 * np.random.randn()),
            "threshold": 2.5 * (1 + 0.1 * np.random.randn()),
            "dt": 1e-3 * (1 + 0.1 * np.random.randn()),
            "paramsScale": 1e12 * (1 + 0.1 * np.random.randn()),
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

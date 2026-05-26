#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace eleanor
{

  std::vector<at::Tensor> bruno_cpu(const at::Tensor &synaptic_input,
                                    const at::Tensor &v, const at::Tensor &p,
                                    const at::Tensor & cap_divider,
                                    const at::Tensor & depol_divider,
                                    const at::Tensor & P_s,
                                    const at::Tensor & A,
                                    const at::Tensor & I_0,
                                    const at::Tensor & E_a,
                                    double V_t, double I_dsc,
                                    double tau_0,
                                    const at::Tensor & C_tot,
                                    double soft_E, double alpha,
                                    double threshold, double dt, int64_t nsteps)
  {
    TORCH_CHECK(v.sizes() == synaptic_input.sizes());
    TORCH_CHECK(v.sizes() == p.sizes());
    TORCH_CHECK(v.sizes() == cap_divider.sizes());
    TORCH_CHECK(v.sizes() == depol_divider.sizes());
    TORCH_CHECK(v.sizes() == P_s.sizes());
    TORCH_CHECK(v.sizes() == A.sizes());
    TORCH_CHECK(v.sizes() == I_0.sizes());
    TORCH_CHECK(v.sizes() == E_a.sizes());
    TORCH_CHECK(v.sizes() == C_tot.sizes());

    TORCH_CHECK(v.dtype() == at::kFloat);
    TORCH_CHECK(p.dtype() == at::kFloat);
    TORCH_CHECK(synaptic_input.dtype() == at::kFloat);
    TORCH_CHECK(cap_divider.dtype() == at::kFloat);
    TORCH_CHECK(depol_divider.dtype() == at::kFloat);
    TORCH_CHECK(P_s.dtype() == at::kFloat);
    TORCH_CHECK(A.dtype() == at::kFloat);
    TORCH_CHECK(I_0.dtype() == at::kFloat);
    TORCH_CHECK(E_a.dtype() == at::kFloat);
    TORCH_CHECK(C_tot.dtype() == at::kFloat);

    TORCH_INTERNAL_ASSERT(v.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(p.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(synaptic_input.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(cap_divider.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(depol_divider.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(P_s.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(A.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(I_0.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(E_a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(C_tot.device().type() == at::DeviceType::CPU);

    at::Tensor v_contig = v.contiguous();
    at::Tensor p_contig = p.contiguous();
    at::Tensor synaptic_input_contig = synaptic_input.contiguous();
    at::Tensor cap_divider_contig = cap_divider.contiguous();
    at::Tensor depol_divider_contig = depol_divider.contiguous();
    at::Tensor P_s_contig = P_s.contiguous();
    at::Tensor A_contig = A.contiguous();
    at::Tensor I_0_contig = I_0.contiguous();
    at::Tensor E_a_contig = E_a.contiguous();
    at::Tensor C_tot_contig = C_tot.contiguous();

    const float * __restrict__ v_ptr               = v_contig.data_ptr<float>();
    const float * __restrict__ p_ptr               = p_contig.data_ptr<float>();
    const float * __restrict__ synaptic_input_ptr  = synaptic_input_contig.data_ptr<float>();
    const float * __restrict__ cap_divider_ptr     = cap_divider_contig.data_ptr<float>();
    const float * __restrict__ depol_divider_ptr   = depol_divider_contig.data_ptr<float>();
    const float * __restrict__ P_s_ptr             = P_s_contig.data_ptr<float>();
    const float * __restrict__ A_ptr               = A_contig.data_ptr<float>();
    const float * __restrict__ I_0_ptr             = I_0_contig.data_ptr<float>();
    const float * __restrict__ E_a_ptr             = E_a_contig.data_ptr<float>();
    const float * __restrict__ C_tot_ptr           = C_tot_contig.data_ptr<float>();

    at::Tensor v_result = torch::empty(v_contig.sizes(), v_contig.options());
    at::Tensor p_result = torch::empty(v_contig.sizes(), v_contig.options());
    float * __restrict__ v_result_ptr = v_result.data_ptr<float>();
    float * __restrict__ p_result_ptr = p_result.data_ptr<float>();

    const float tau_0_f     = static_cast<float>(tau_0);
    const float alpha_f     = static_cast<float>(alpha);
    const float soft_E_f    = static_cast<float>(soft_E);
    const float V_t_f       = static_cast<float>(V_t);
    const float I_dsc_f     = static_cast<float>(I_dsc);
    const float dt_f        = static_cast<float>(dt);
    const float threshold_f = static_cast<float>(threshold);
    const float int_div     = 1.f / static_cast<float>(nsteps);
    const float sub_dt      = int_div * dt_f;
    constexpr float V_CLIP  = 5.f;
    const int64_t numel     = v.numel();

#pragma omp parallel for if(numel > 64) schedule(static)
    for (int64_t neuron = 0; neuron < numel; neuron++)
    {
      // Per-neuron loop invariants — hoisted out of the inner t-loop.
      const float cap_div   = cap_divider_ptr[neuron];
      const float depol_div = depol_divider_ptr[neuron];
      const float P_s_n     = P_s_ptr[neuron];
      const float A_n       = A_ptr[neuron];
      const float I_0_n     = I_0_ptr[neuron];
      const float E_a_n     = E_a_ptr[neuron];
      const float C_tot_n   = C_tot_ptr[neuron];
      const float isyn      = synaptic_input_ptr[neuron];

      float v_tmp = v_ptr[neuron];
      float p_tmp = p_ptr[neuron];

      for (int64_t t = 0; t < nsteps; t++)
      {
        const float E   = v_tmp * cap_div - p_tmp * depol_div;
        const float tau = 1.f / (tau_0_f * std::exp(std::pow(E_a_n / (std::fabs(E) + soft_E_f), alpha_f)));

        const float sign_E = static_cast<float>((E > 0.f) - (E < 0.f));
        const float I_p_new = (sign_E * P_s_n - p_tmp) * A_n * tau;

        const float sign_v = static_cast<float>((v_tmp > 0.f) - (v_tmp < 0.f));
        const float I_leak = (I_0_n * A_n * std::expm1(v_tmp / V_t_f) + I_dsc_f) * sign_v;

        const float dp = I_p_new / A_n;
        const float dv = (isyn - I_leak - I_p_new) / C_tot_n;

        if (v_tmp < threshold_f)
        {
          v_tmp += sub_dt * dv;
          p_tmp += sub_dt * dp;
        }

        v_tmp = std::clamp(v_tmp, -V_CLIP, V_CLIP);
        p_tmp = std::clamp(p_tmp, -P_s_n, P_s_n);
      }

      v_result_ptr[neuron] = v_tmp;
      p_result_ptr[neuron] = p_tmp;
    }

    return std::vector<at::Tensor>{v_result, p_result};
  }

  TORCH_LIBRARY_IMPL(eleanor, CPU, m)
  {
    m.impl("bruno", &bruno_cpu);
  }

}

#include <torch/extension.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace eleanor
{

  __device__ float tau_fn_cuda(float E, float E_a, float tau_0, float soft_E, float alpha)
  {
    return 1 / (tau_0 * exp(pow(E_a / (abs(E) + soft_E), alpha)));
  }

  __global__ void felif_kernel(int numel,
    const float *synaptic_input,
    const float *v, const float *p,
    float *v_res, float *p_res,
    const float *cap_divider,
    const float *depol_divider,
    const float *P_s,
    const float *A,
    const float *I_0,
    const float *E_a,
    float V_t,
    float I_dsc,
    float tau_0,
    const float *C_tot,
    float soft_E,
    float alpha,
    float threshold,
    float dt,
    int64_t nsteps)
  {
    float E, tau, I_p_new, I_leak, dp, dv;
    float int_div = 1/static_cast<float>(nsteps);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
    {
      float v_tmp = v[idx];
      float p_tmp = p[idx];

      for (int64_t t = 0; t < nsteps; t++)
      {
        E = v_tmp * cap_divider[idx] - p_tmp * depol_divider[idx];
        tau = tau_fn_cuda(E, E_a[idx], tau_0, soft_E, alpha);

        I_p_new = (copysign(1.0, E) * P_s[idx] - p_tmp) * A[idx] * tau;
        I_leak = (I_0[idx] * A[idx] * expm1(v_tmp / V_t) + I_dsc) * copysign(1.0, v_tmp);

        dp = I_p_new / A[idx];
        dv = (synaptic_input[idx] - I_leak - I_p_new) / C_tot[idx];

        if (v_tmp <= threshold)
        {
          v_tmp = v_tmp + int_div * dt * dv;
          p_tmp = p_tmp + int_div * dt * dp;
        }

        if (v_tmp > 5)
          v_tmp = 5;
        else if (v_tmp < -5)
          v_tmp = -5;

        if (p_tmp > P_s[idx])
          p_tmp = P_s[idx];
        else if (p_tmp < -P_s[idx])
          p_tmp = -P_s[idx];
      }
      v_res[idx] = v_tmp;
      p_res[idx] = p_tmp;
    }
  }

  std::vector<at::Tensor> bruno_cuda(
    const at::Tensor &synaptic_input,
    const at::Tensor &v,
    const at::Tensor &p,
    const at::Tensor &cap_divider,
    const at::Tensor &depol_divider,
    const at::Tensor &P_s,
    const at::Tensor &A,
    const at::Tensor &I_0,
    const at::Tensor &E_a,
    double V_t,
    double I_dsc,
    double tau_0,
    const at::Tensor &C_tot,
    double soft_E,
    double alpha,
    double threshold,
    double dt,
    int64_t nsteps)
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

    TORCH_INTERNAL_ASSERT(v.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(p.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(synaptic_input.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(cap_divider.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(depol_divider.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(P_s.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(A.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(I_0.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(E_a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(C_tot.device().type() == at::DeviceType::CUDA);

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

    at::Tensor v_result = torch::empty(v_contig.sizes(), v_contig.options());
    at::Tensor p_result = torch::empty(v_contig.sizes(), v_contig.options());

    float *v_ptr = v_contig.data_ptr<float>();
    float *p_ptr = p_contig.data_ptr<float>();
    float *synaptic_input_ptr = synaptic_input_contig.data_ptr<float>();

    float *cap_divider_ptr = cap_divider_contig.data_ptr<float>();
    float *depol_divider_ptr = depol_divider_contig.data_ptr<float>();
    float *P_s_ptr = P_s_contig.data_ptr<float>();
    float *A_ptr = A_contig.data_ptr<float>();
    float *I_0_ptr = I_0_contig.data_ptr<float>();
    float *E_a_ptr = E_a_contig.data_ptr<float>();
    float *C_tot_ptr = C_tot_contig.data_ptr<float>();

    float *v_result_ptr = v_result.data_ptr<float>();
    float *p_result_ptr = p_result.data_ptr<float>();

    int numel = v_contig.numel();
    felif_kernel<<<(numel + 255) / 256, 256>>>(numel, synaptic_input_ptr, v_ptr, p_ptr, v_result_ptr, p_result_ptr, cap_divider_ptr, depol_divider_ptr, P_s_ptr, A_ptr, I_0_ptr, E_a_ptr, V_t, I_dsc, tau_0, C_tot_ptr, soft_E, alpha, threshold, dt, nsteps);

    return std::vector<at::Tensor>{v_result, p_result};
  }

  // Registers CUDA implementations for mymuladd, mymul, myadd_out
  TORCH_LIBRARY_IMPL(eleanor, CUDA, m)
  {
    m.impl("bruno", &bruno_cuda);
  }

}

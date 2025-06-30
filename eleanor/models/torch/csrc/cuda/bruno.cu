#include <torch/extension.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace eleanor {

__device__ float tau_fn_cuda(float E, float E_a, float tau_0, float soft_E, float alpha) {
  return 1 / (tau_0 * exp(pow(E_a / (abs(E) + soft_E),  alpha)));
}

__global__ void felif_kernel(int numel, const float* synaptic_input, const float* v, const float* p, float* v_res, float* p_res, float cap_divider, float depol_divider, float P_s, float A, float I_0, float E_a, float V_t, float I_dsc, float tau_0, float C_tot, float soft_E, float alpha, float threshold, float dt) {
  float E, tau, I_p_new, I_leak, dp, dv;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    float v_tmp = v[idx];
    float p_tmp = p[idx];

    for (int64_t t = 0; t < 1000; t++) {
      E = v_tmp * cap_divider - p_tmp * depol_divider;
      tau = tau_fn_cuda(E, E_a, tau_0, soft_E, alpha);

      I_p_new = (copysign(1.0, E) * P_s - p_tmp) * A * tau;
      I_leak = (I_0 * A * expm1(v_tmp/ V_t) + I_dsc) * copysign(1.0, v_tmp);

      dp = I_p_new / A;
      dv = (synaptic_input[idx] - I_leak - I_p_new) / C_tot;

      if (v_tmp <= threshold){
        v_tmp = v_tmp + 0.001 * dt * dv;
        p_tmp = p_tmp + 0.001 * dt * dp;
      }

      if (v_tmp > 5)
        v_tmp = 5;
      else if (v_tmp < -5)
        v_tmp = -5;

      if (p_tmp > P_s)
        p_tmp = P_s;
      else if (p_tmp < -P_s)
        p_tmp = -P_s;

    }
    v_res[idx] = v_tmp;
    p_res[idx] = p_tmp;
  }
}

std::vector<at::Tensor> bruno_cuda(const at::Tensor& synaptic_input, const at::Tensor& v, const at::Tensor& p, double cap_divider, double depol_divider, double P_s, double A, double I_0, double E_a, double V_t, double I_dsc, double tau_0, double C_tot, double soft_E, double alpha, double threshold, double dt) {
  TORCH_CHECK(v.sizes() == synaptic_input.sizes());
  TORCH_CHECK(v.sizes() == p.sizes());
  TORCH_CHECK(v.dtype() == at::kFloat);
  TORCH_CHECK(p.dtype() == at::kFloat);
  TORCH_CHECK(synaptic_input.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(v.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(p.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(synaptic_input.device().type() == at::DeviceType::CUDA);
  at::Tensor v_contig = v.contiguous();
  at::Tensor p_contig = p.contiguous();
  at::Tensor synaptic_input_contig = synaptic_input.contiguous();
  at::Tensor v_result = torch::empty(v_contig.sizes(), v_contig.options());
  at::Tensor p_result = torch::empty(v_contig.sizes(), v_contig.options());

  const float* v_ptr = v_contig.data_ptr<float>();
  const float* p_ptr = p_contig.data_ptr<float>();
  const float* synaptic_input_ptr = synaptic_input_contig.data_ptr<float>();
  float* v_result_ptr = v_result.data_ptr<float>();
  float* p_result_ptr = p_result.data_ptr<float>();

  int numel = v_contig.numel();
  felif_kernel<<<(numel+255)/256, 256>>>(numel, synaptic_input_ptr, v_ptr, p_ptr, v_result_ptr, p_result_ptr, cap_divider, depol_divider, P_s, A, I_0, E_a, V_t, I_dsc, tau_0, C_tot, soft_E, alpha, threshold, dt);

  return std::vector<at::Tensor>{v_result, p_result};
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(eleanor, CUDA, m) {
  m.impl("bruno", &bruno_cuda);
}

}

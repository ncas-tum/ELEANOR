#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <vector>

#include "heracles.h"

namespace eleanor
{

  __global__ void heracles_kernel(int64_t numel,
                                  const float *__restrict__ synaptic_input,
                                  const float *__restrict__ v, const float *__restrict__ p,
                                  float *__restrict__ v_res, float *__restrict__ p_res,
                                  BroadcastParam A, BroadcastParam t_fe,
                                  BroadcastParam n_depl, BroadcastParam P_s,
                                  HeraclesConsts consts, int64_t nsteps)
  {
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < numel;
         i += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
      float vi = v[i];
      float pi = p[i];
      heracles_integrate(vi, pi, synaptic_input[i], A[i], t_fe[i], n_depl[i], P_s[i], consts, nsteps);
      v_res[i] = vi;
      p_res[i] = pi;
    }
  }

  std::vector<at::Tensor> heracles_cuda(const at::Tensor &synaptic_input,
                                        const at::Tensor &v, const at::Tensor &p,
                                        const at::Tensor &A, const at::Tensor &t_fe, double eps_fe, double eps_depl,
                                        double q_fix_depl, const at::Tensor &n_depl, double e_off,
                                        double temp, double w_b, double d_e, const at::Tensor &P_s,
                                        double I_0, double V_t, double C_par, double C_fe,
                                        double I_dsc, double _eps0, double _q, double _k, double _h,
                                        double threshold, double dt, double paramsScale, int64_t nsteps)
  {
    check_state(v, "v", {{&p, "p"}, {&synaptic_input, "synaptic_input"}}, at::DeviceType::CUDA);
    const HeraclesConsts consts = make_heracles_consts(
        eps_fe, eps_depl, q_fix_depl, e_off, temp, w_b, d_e, I_0, V_t, C_par, C_fe,
        I_dsc, _eps0, _q, _k, _h, threshold, dt, paramsScale, nsteps);
    const c10::cuda::OptionalCUDAGuard device_guard(v.device());

    const at::Tensor v_c = v.contiguous();
    const at::Tensor p_c = p.contiguous();
    const at::Tensor isyn_c = synaptic_input.contiguous();
    const at::Tensor A_c = broadcast_param(A, "A", v);
    const at::Tensor t_fe_c = broadcast_param(t_fe, "t_fe", v);
    const at::Tensor n_depl_c = broadcast_param(n_depl, "n_depl", v);
    const at::Tensor P_s_c = broadcast_param(P_s, "P_s", v);

    at::Tensor v_out = at::empty_like(v_c);
    at::Tensor p_out = at::empty_like(v_c);

    const int64_t numel = v_c.numel();
    if (numel == 0)
      return {v_out, p_out};

    constexpr int threads = 256;
    const int64_t blocks = std::min<int64_t>((numel + threads - 1) / threads, 65535);
    heracles_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        numel, isyn_c.data_ptr<float>(), v_c.data_ptr<float>(), p_c.data_ptr<float>(),
        v_out.data_ptr<float>(), p_out.data_ptr<float>(),
        view(A_c), view(t_fe_c), view(n_depl_c), view(P_s_c), consts, nsteps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {v_out, p_out};
  }

  TORCH_LIBRARY_IMPL(eleanor, CUDA, m)
  {
    m.impl("heracles", &heracles_cuda);
  }

} // namespace eleanor

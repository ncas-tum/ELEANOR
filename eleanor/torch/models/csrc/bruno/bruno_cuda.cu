#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <vector>

#include "bruno.h"

namespace eleanor
{

  __global__ void bruno_kernel(int64_t numel,
                               const float *__restrict__ synaptic_input,
                               const float *__restrict__ v, const float *__restrict__ p,
                               float *__restrict__ v_res, float *__restrict__ p_res,
                               BroadcastParam cap_divider, BroadcastParam depol_divider,
                               BroadcastParam P_s, BroadcastParam A, BroadcastParam I_0,
                               BroadcastParam E_a, BroadcastParam C_tot,
                               BrunoConsts consts, int64_t nsteps)
  {
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < numel;
         i += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
      float vi = v[i];
      float pi = p[i];
      bruno_integrate(vi, pi, synaptic_input[i], cap_divider[i], depol_divider[i], P_s[i], A[i],
                      I_0[i], E_a[i], C_tot[i], consts, nsteps);
      v_res[i] = vi;
      p_res[i] = pi;
    }
  }

  std::vector<at::Tensor> bruno_cuda(const at::Tensor &synaptic_input,
                                     const at::Tensor &v, const at::Tensor &p,
                                     const at::Tensor &cap_divider,
                                     const at::Tensor &depol_divider,
                                     const at::Tensor &P_s,
                                     const at::Tensor &A,
                                     const at::Tensor &I_0,
                                     const at::Tensor &E_a,
                                     double V_t, double I_dsc,
                                     double tau_0,
                                     const at::Tensor &C_tot,
                                     double soft_E, double alpha,
                                     double threshold, double dt, int64_t nsteps)
  {
    check_state(v, "v", {{&p, "p"}, {&synaptic_input, "synaptic_input"}}, at::DeviceType::CUDA);
    const BrunoConsts consts = make_bruno_consts(V_t, I_dsc, tau_0, soft_E, alpha, threshold, dt, nsteps);
    const c10::cuda::OptionalCUDAGuard device_guard(v.device());

    const at::Tensor v_c = v.contiguous();
    const at::Tensor p_c = p.contiguous();
    const at::Tensor isyn_c = synaptic_input.contiguous();
    const at::Tensor cap_c = broadcast_param(cap_divider, "cap_divider", v);
    const at::Tensor depol_c = broadcast_param(depol_divider, "depol_divider", v);
    const at::Tensor P_s_c = broadcast_param(P_s, "P_s", v);
    const at::Tensor A_c = broadcast_param(A, "A", v);
    const at::Tensor I_0_c = broadcast_param(I_0, "I_0", v);
    const at::Tensor E_a_c = broadcast_param(E_a, "E_a", v);
    const at::Tensor C_tot_c = broadcast_param(C_tot, "C_tot", v);

    at::Tensor v_out = at::empty_like(v_c);
    at::Tensor p_out = at::empty_like(v_c);

    const int64_t numel = v_c.numel();
    if (numel == 0)
      return {v_out, p_out};

    constexpr int threads = 256;
    const int64_t blocks = std::min<int64_t>((numel + threads - 1) / threads, 65535);
    bruno_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        numel, isyn_c.data_ptr<float>(), v_c.data_ptr<float>(), p_c.data_ptr<float>(),
        v_out.data_ptr<float>(), p_out.data_ptr<float>(),
        view(cap_c), view(depol_c), view(P_s_c), view(A_c), view(I_0_c), view(E_a_c), view(C_tot_c),
        consts, nsteps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {v_out, p_out};
  }

  TORCH_LIBRARY_IMPL(eleanor, CUDA, m)
  {
    m.impl("bruno", &bruno_cuda);
  }

} // namespace eleanor

#include <ATen/Parallel.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

#include "bruno.h"

namespace eleanor
{

  std::vector<at::Tensor> bruno_cpu(const at::Tensor &synaptic_input,
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
    check_state(v, "v", {{&p, "p"}, {&synaptic_input, "synaptic_input"}}, at::DeviceType::CPU);
    const BrunoConsts consts = make_bruno_consts(V_t, I_dsc, tau_0, soft_E, alpha, threshold, dt, nsteps);

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

    const float *v_in = v_c.data_ptr<float>();
    const float *p_in = p_c.data_ptr<float>();
    const float *isyn = isyn_c.data_ptr<float>();
    const BroadcastParam cap = view(cap_c), depol = view(depol_c), Ps = view(P_s_c),
                         Ap = view(A_c), I0 = view(I_0_c), Ea = view(E_a_c), Ctot = view(C_tot_c);
    float *v_res = v_out.data_ptr<float>();
    float *p_res = p_out.data_ptr<float>();

    // Each element runs `nsteps` sub-steps, so even small chunks are worth a thread.
    at::parallel_for(0, v_c.numel(), 16, [&](int64_t begin, int64_t end)
    {
      for (int64_t i = begin; i < end; i++)
      {
        float vi = v_in[i];
        float pi = p_in[i];
        bruno_integrate(vi, pi, isyn[i], cap[i], depol[i], Ps[i], Ap[i], I0[i], Ea[i], Ctot[i], consts, nsteps);
        v_res[i] = vi;
        p_res[i] = pi;
      }
    });

    return {v_out, p_out};
  }

  TORCH_LIBRARY_IMPL(eleanor, CPU, m)
  {
    m.impl("bruno", &bruno_cpu);
  }

} // namespace eleanor

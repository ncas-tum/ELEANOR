#include <ATen/Parallel.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

#include "heracles.h"

namespace eleanor
{

  std::vector<at::Tensor> heracles_cpu(const at::Tensor &synaptic_input,
                                       const at::Tensor &v, const at::Tensor &p,
                                       const at::Tensor &A, const at::Tensor &t_fe, double eps_fe, double eps_depl,
                                       double q_fix_depl, const at::Tensor &n_depl, double e_off,
                                       double temp, double w_b, double d_e, const at::Tensor &P_s,
                                       double I_0, double V_t, double C_par, double C_fe,
                                       double I_dsc, double _eps0, double _q, double _k, double _h,
                                       double threshold, double dt, double paramsScale, int64_t nsteps)
  {
    check_state(v, "v", {{&p, "p"}, {&synaptic_input, "synaptic_input"}}, at::DeviceType::CPU);
    const HeraclesConsts consts = make_heracles_consts(
        eps_fe, eps_depl, q_fix_depl, e_off, temp, w_b, d_e, I_0, V_t, C_par, C_fe,
        I_dsc, _eps0, _q, _k, _h, threshold, dt, paramsScale, nsteps);

    const at::Tensor v_c = v.contiguous();
    const at::Tensor p_c = p.contiguous();
    const at::Tensor isyn_c = synaptic_input.contiguous();
    const at::Tensor A_c = broadcast_param(A, "A", v);
    const at::Tensor t_fe_c = broadcast_param(t_fe, "t_fe", v);
    const at::Tensor n_depl_c = broadcast_param(n_depl, "n_depl", v);
    const at::Tensor P_s_c = broadcast_param(P_s, "P_s", v);

    at::Tensor v_out = at::empty_like(v_c);
    at::Tensor p_out = at::empty_like(v_c);

    const float *v_in = v_c.data_ptr<float>();
    const float *p_in = p_c.data_ptr<float>();
    const float *isyn = isyn_c.data_ptr<float>();
    const BroadcastParam Ap = view(A_c), tfe = view(t_fe_c), nd = view(n_depl_c), Ps = view(P_s_c);
    float *v_res = v_out.data_ptr<float>();
    float *p_res = p_out.data_ptr<float>();

    // Each element runs `nsteps` sub-steps, so even small chunks are worth a thread.
    at::parallel_for(0, v_c.numel(), 16, [&](int64_t begin, int64_t end)
    {
      for (int64_t i = begin; i < end; i++)
      {
        float vi = v_in[i];
        float pi = p_in[i];
        heracles_integrate(vi, pi, isyn[i], Ap[i], tfe[i], nd[i], Ps[i], consts, nsteps);
        v_res[i] = vi;
        p_res[i] = pi;
      }
    });

    return {v_out, p_out};
  }

  TORCH_LIBRARY_IMPL(eleanor, CPU, m)
  {
    m.impl("heracles", &heracles_cpu);
  }

} // namespace eleanor

#pragma once

// Heracles model update, shared by the CPU loop and the CUDA kernel so both backends
// integrate exactly the same equations (reference: eleanor/jax/models/_heracles.py).

#include "../common.h"

#include <cmath>

namespace eleanor
{

  // Loop-invariant combinations of the scalar parameters, folded in double on the host.
  struct HeraclesConsts
  {
    float eps0_eps_fe;       // _eps0 * eps_fe
    float w_scale;           // paramsScale / _q   (divided by n_depl per neuron)
    float q_fix_depl;
    float eps_fe, eps_depl;
    float inv_eps0;          // 1 / _eps0
    float inv_C_fe_par;      // 1 / (C_fe + C_par)
    float inv_eps0_eps_depl; // 1 / (_eps0 * eps_depl)   (divided by A per neuron)
    float e_off, d_e, w_b;
    float beta;              // _q / (_k * temp)
    float attempt;           // _k * temp / _h
    float I_0, inv_V_t, I_dsc;
    float threshold, sub_dt;
  };

  inline HeraclesConsts make_heracles_consts(double eps_fe, double eps_depl, double q_fix_depl,
                                             double e_off, double temp, double w_b, double d_e,
                                             double I_0, double V_t, double C_par, double C_fe,
                                             double I_dsc, double _eps0, double _q, double _k,
                                             double _h, double threshold, double dt,
                                             double paramsScale, int64_t nsteps)
  {
    TORCH_CHECK(nsteps > 0, "nsteps must be positive, got ", nsteps);
    auto f = [](double x) { return static_cast<float>(x); };
    return {f(_eps0 * eps_fe), f(paramsScale / _q), f(q_fix_depl),
            f(eps_fe), f(eps_depl),
            f(1.0 / _eps0), f(1.0 / (C_fe + C_par)), f(1.0 / (_eps0 * eps_depl)),
            f(e_off), f(d_e), f(w_b),
            f(_q / (_k * temp)), f(_k * temp / _h),
            f(I_0), f(1.0 / V_t), f(I_dsc),
            f(threshold), f(dt / static_cast<double>(nsteps))};
  }

  // Integrates one neuron over `nsteps` sub-steps of size c.sub_dt, in place.
  ELEANOR_HD void heracles_integrate(float &v, float &p, float isyn, float A, float t_fe,
                                     float n_depl, float P_s, const HeraclesConsts &c,
                                     int64_t nsteps)
  {
    // Depletion width is w = a*v + b (down) and |a*v - b| (up).
    const float w_scale = c.w_scale / n_depl;
    const float a = c.eps0_eps_fe / t_fe * w_scale;
    const float b = c.q_fix_depl * w_scale;
    const float inv_2P_s = 0.5f / P_s;
    const float two_P_s = 2.f * P_s;
    const float inv_eps0_eps_depl_A = c.inv_eps0_eps_depl / A;
    const float t_fe_eps_depl = t_fe * c.eps_depl;
    const float I_0A = c.I_0 * A;

    for (int64_t t = 0; t < nsteps; t++)
    {
      const float prob = p * inv_2P_s + 0.5f;

      const float w_depl_d = a * v + b;
      const float w_depl_u = fabsf(a * v - b);
      const float w_depl = w_depl_d * w_depl_u / (prob * w_depl_u + (1.f - prob) * w_depl_d);

      // 1/C_tot = 1/(C_fe + C_par) + w_depl / (_eps0 * eps_depl * A)
      const float inv_C_tot = c.inv_C_fe_par + w_depl * inv_eps0_eps_depl_A;
      // E = v * cap_divider - p * depol_divider, sharing the common denominator.
      const float inv_denom = 1.f / (t_fe_eps_depl + w_depl * c.eps_fe);
      const float E = (v * c.eps_depl - p * w_depl * c.inv_eps0) * inv_denom;

      const float w_e = (E - c.e_off) * c.d_e;
      const float k_down = c.attempt * expf(-reluf(c.w_b - w_e) * c.beta);
      const float k_up = c.attempt * expf(-reluf(c.w_b + w_e) * c.beta);

      const float dp = two_P_s * (k_down * (1.f - prob) - k_up * prob);
      const float I_p = dp * A;
      const float I_leak = (I_0A * expm1f(v * c.inv_V_t) + c.I_dsc) * sgn(v);
      const float dv = (isyn - I_leak - I_p) * inv_C_tot;

      // A neuron that crossed threshold holds its state for the rest of the step.
      if (v < c.threshold)
      {
        v += c.sub_dt * dv;
        p += c.sub_dt * dp;
      }

      v = clampf(v, -5.f, 5.f);
      p = clampf(p, -P_s, P_s);
    }
  }

} // namespace eleanor

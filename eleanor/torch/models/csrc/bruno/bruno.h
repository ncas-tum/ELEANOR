#pragma once

// Bruno model update, shared by the CPU loop and the CUDA kernel so both backends
// integrate exactly the same equations (reference: eleanor/jax/models/_bruno.py).

#include "../common.h"

#include <cmath>

namespace eleanor
{

  struct BrunoConsts
  {
    float inv_V_t, I_dsc, inv_tau_0, soft_E, alpha, threshold, sub_dt;
  };

  inline BrunoConsts make_bruno_consts(double V_t, double I_dsc, double tau_0, double soft_E,
                                       double alpha, double threshold, double dt, int64_t nsteps)
  {
    TORCH_CHECK(nsteps > 0, "nsteps must be positive, got ", nsteps);
    return {static_cast<float>(1.0 / V_t), static_cast<float>(I_dsc),
            static_cast<float>(1.0 / tau_0), static_cast<float>(soft_E),
            static_cast<float>(alpha), static_cast<float>(threshold),
            static_cast<float>(dt / static_cast<double>(nsteps))};
  }

  // Integrates one neuron over `nsteps` sub-steps of size c.sub_dt, in place.
  ELEANOR_HD void bruno_integrate(float &v, float &p, float isyn, float cap_div, float depol_div,
                                  float P_s, float A, float I_0, float E_a, float C_tot,
                                  const BrunoConsts &c, int64_t nsteps)
  {
    const float inv_C_tot = 1.f / C_tot;
    const float I_0A = I_0 * A;

    for (int64_t t = 0; t < nsteps; t++)
    {
      const float E = v * cap_div - p * depol_div;
      const float tau = c.inv_tau_0 * expf(-powf(E_a / (fabsf(E) + c.soft_E), c.alpha));

      const float dp = (sgn(E) * P_s - p) * tau;
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

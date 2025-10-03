#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <omp.h>

#include <vector>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace eleanor
{
TORCH_LIBRARY(eleanor, m) {
  m.def("bruno(Tensor synaptic_input, Tensor v, Tensor p, Tensor cap_divider, Tensor depol_divider, Tensor P_s, Tensor A, Tensor I_0, Tensor E_a, float V_t, float I_dsc, float tau_0, Tensor C_tot, float soft_E, float alpha, float threshold, float dt, int nsteps) -> Tensor[]");
  m.def("heracles(Tensor synaptic_input, Tensor v, Tensor p, Tensor A, Tensor t_fe, float eps_fe, float eps_depl, float q_fix_depl, Tensor n_depl, float e_off, float temp, float w_b, float d_e, Tensor P_s, float I_0, float V_t, float C_par, float C_fe, float I_dsc, float _eps0, float _q, float _k, float _h, float threshold, float dt, float paramsScale, int nsteps) -> Tensor[]");
}

} // namespace name

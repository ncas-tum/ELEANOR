#include <Python.h>
#include <torch/library.h>

#ifndef TORCH_EXTENSION_NAME
#error "TORCH_EXTENSION_NAME must be defined (torch.utils.cpp_extension sets it)"
#endif

#define ELEANOR_CONCAT_(a, b) a##b
#define ELEANOR_CONCAT(a, b) ELEANOR_CONCAT_(a, b)
#define ELEANOR_STR_(x) #x
#define ELEANOR_STR(x) ELEANOR_STR_(x)

/* Creates an empty module that can be imported from Python. Importing it loads the
   .so so that the TORCH_LIBRARY static initializers below run. The init symbol must
   match the module name, which setup.py suffixes with the CUDA flavor (_C_cpu, _C_cu126). */
PyMODINIT_FUNC ELEANOR_CONCAT(PyInit_, TORCH_EXTENSION_NAME)(void)
{
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      ELEANOR_STR(TORCH_EXTENSION_NAME), /* name of module */
      NULL,                              /* module documentation */
      -1,                                /* module keeps state in global variables */
      NULL,                              /* methods */
  };
  return PyModule_Create(&module_def);
}

namespace eleanor
{
  TORCH_LIBRARY(eleanor, m)
  {
    m.def("bruno(Tensor synaptic_input, Tensor v, Tensor p, Tensor cap_divider, Tensor depol_divider, Tensor P_s, Tensor A, Tensor I_0, Tensor E_a, float V_t, float I_dsc, float tau_0, Tensor C_tot, float soft_E, float alpha, float threshold, float dt, int nsteps) -> Tensor[]");
    m.def("heracles(Tensor synaptic_input, Tensor v, Tensor p, Tensor A, Tensor t_fe, float eps_fe, float eps_depl, float q_fix_depl, Tensor n_depl, float e_off, float temp, float w_b, float d_e, Tensor P_s, float I_0, float V_t, float C_par, float C_fe, float I_dsc, float _eps0, float _q, float _k, float _h, float threshold, float dt, float paramsScale, int nsteps) -> Tensor[]");
  }
} // namespace eleanor

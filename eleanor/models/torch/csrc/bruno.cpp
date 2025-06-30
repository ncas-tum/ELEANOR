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

namespace eleanor {

std::vector<at::Tensor> bruno_cpu(const at::Tensor& synaptic_input, const at::Tensor& v, const at::Tensor& p, double cap_divider, double depol_divider, double P_s, double A, double I_0, double E_a, double V_t, double I_dsc, double tau_0, double C_tot, double soft_E, double alpha, double threshold, double dt) {
  at::Tensor v_contig = v.contiguous();
  at::Tensor p_contig = p.contiguous();
  at::Tensor synaptic_input_contig = synaptic_input.contiguous();
  float* v_ptr = v_contig.data_ptr<float>();
  float* p_ptr = p_contig.data_ptr<float>();
  float* synaptic_input_ptr = synaptic_input_contig.data_ptr<float>();

  at::Tensor v_result = torch::empty(v_contig.sizes(), v_contig.options());
  at::Tensor p_result = torch::empty(v_contig.sizes(), v_contig.options());
  float* v_result_ptr = v_result.data_ptr<float>();
  float* p_result_ptr = p_result.data_ptr<float>();

  // omp_set_num_threads(omp_get_max_threads());

  #pragma omp parallel for
  for (int64_t neuron=0; neuron<v.numel(); neuron++) {
    double E, tau, I_p_new, I_leak, dp, dv;
    float v_tmp = v_ptr[neuron];
    float p_tmp = p_ptr[neuron];
    // float v_new, p_new;

    for (int64_t t = 0; t < 1000; t++) {
      E = v_tmp * cap_divider - p_tmp * depol_divider;
      // tau = 1 / (tau_0 * std::exp(std::pow(E_a / (std::abs(E) + soft_E),  alpha)));
      tau = 1 / (tau_0 * std::exp(std::pow(E_a / (std::abs(E) + soft_E),  alpha)));

      // I_p_new = (static_cast<float>(std::copysign(1.0, E) * P_s) - p_tmp) * static_cast<float>(A * tau);
      I_p_new = (std::copysign(1.0, E) * P_s - p_tmp) * A * tau;
      I_leak = (I_0 * A * std::expm1(v_tmp/ V_t) + I_dsc) * std::copysign(1.0, v_tmp);

      dp = I_p_new / A;
      dv = (synaptic_input_ptr[neuron] - I_leak - I_p_new) / C_tot;

      if (v_tmp <= threshold){
        v_tmp = v_tmp + static_cast<float>(0.001 * dt * dv);
        p_tmp = p_tmp + static_cast<float>(0.001 * dt * dp);
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

    v_result_ptr[neuron] = v_tmp;
    p_result_ptr[neuron] = p_tmp;
  }

  return std::vector<at::Tensor>{v_result, p_result};
}

// Defines the operators
TORCH_LIBRARY(eleanor, m) {
  m.def("bruno(Tensor synaptic_input, Tensor v, Tensor p, float cap_divider, float depol_divider, float P_s, float A, float I_0, float E_a, float V_t, float I_dsc, float tau_0, float C_tot, float soft_E, float alpha, float threshold, float dt) -> Tensor[]");
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(eleanor, CPU, m) {
  m.impl("bruno", &bruno_cpu);
}

}

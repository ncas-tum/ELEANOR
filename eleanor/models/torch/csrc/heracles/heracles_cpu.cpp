#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <omp.h>

#include <vector>

namespace eleanor
{
    std::vector<at::Tensor> heracles_cpu(const at::Tensor &synaptic_input,
                                         const at::Tensor &v, const at::Tensor &p,
                                         const at::Tensor & A, const at::Tensor & t_fe, double eps_fe, double eps_depl,
                                         double q_fix_depl, const at::Tensor & n_depl, double e_off,
                                         double temp, double w_b, double d_e, const at::Tensor & P_s,
                                         double I_0, double V_t, double C_par, double C_fe,
                                         double C_tot_init, double I_dsc, double _eps0, double _q, double _k, double _h,
                                         double threshold, double dt, double paramsScale)
    {
        TORCH_CHECK(v.sizes() == synaptic_input.sizes());
        TORCH_CHECK(v.sizes() == p.sizes());
        TORCH_CHECK(v.dtype() == at::kFloat);
        TORCH_CHECK(p.dtype() == at::kFloat);
        TORCH_CHECK(synaptic_input.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(v.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(p.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(synaptic_input.device().type() == at::DeviceType::CPU);

        at::Tensor v_contig = v.contiguous();
        at::Tensor p_contig = p.contiguous();
        at::Tensor synaptic_input_contig = synaptic_input.contiguous();

        at::Tensor A_contig = A.contiguous();
        at::Tensor t_fe_contig = t_fe.contiguous();
        at::Tensor n_depl_contig = n_depl.contiguous();
        at::Tensor P_s_contig = P_s.contiguous();

        float *v_ptr = v_contig.data_ptr<float>();
        float *p_ptr = p_contig.data_ptr<float>();
        float *synaptic_input_ptr = synaptic_input_contig.data_ptr<float>();

        float *A_ptr = A_contig.data_ptr<float>();
        float *t_fe_ptr = t_fe_contig.data_ptr<float>();
        float *n_depl_ptr = n_depl_contig.data_ptr<float>();
        float *P_s_ptr = P_s_contig.data_ptr<float>();

        at::Tensor v_result = torch::empty(v_contig.sizes(), v_contig.options());
        at::Tensor p_result = torch::empty(v_contig.sizes(), v_contig.options());
        float *v_result_ptr = v_result.data_ptr<float>();
        float *p_result_ptr = p_result.data_ptr<float>();

        // omp_set_num_threads(omp_get_max_threads());

#pragma omp parallel for
        for (int64_t neuron = 0; neuron < v.numel(); neuron++)
        {
            float E, I_p_new, I_leak, dp, dv;
            float prob, e_dummy, w_depl_d, w_depl_u, w_depl, C_tot, cap_divider, depol_divider, w_e, w_exp_down, w_exp_up, k_down, k_up;
            float v_tmp = v_ptr[neuron];
            float p_tmp = p_ptr[neuron];
            // float v_new, p_new;

            float A = A_ptr[neuron];
            float t_fe = t_fe_ptr[neuron];
            float n_depl = n_depl_ptr[neuron];
            float P_s = P_s_ptr[neuron];

            for (int64_t t = 0; t < 1000; t++)
            {
                // Calculate cap and depol dividers
                prob = p_tmp / 2 / P_s + 0.5;
                e_dummy = v_tmp / t_fe;
                w_depl_d = ((_eps0 * eps_fe * e_dummy + q_fix_depl) * paramsScale / _q / n_depl);
                w_depl_u = std::abs((_eps0 * eps_fe * e_dummy - q_fix_depl) * paramsScale / _q / n_depl);
                w_depl = w_depl_d * w_depl_u / (prob * w_depl_u + (1 - prob) * w_depl_d);

                C_tot = 1 / (1 / (C_fe + C_par) + 1 / (_eps0 * eps_depl / w_depl * A));
                cap_divider = eps_depl / (t_fe * eps_depl + w_depl * eps_fe);
                depol_divider = 1 / _eps0 * w_depl / (t_fe * eps_depl + w_depl * eps_fe);

                // FeLIF equation
                E = v_tmp * cap_divider - p_tmp * depol_divider;
                w_e = (E - e_off) * d_e;
                w_exp_down = std::exp(-(w_b - w_e) * _q / _k / temp);
                k_down = _k * temp / _h * w_exp_down;
                w_exp_up = std::exp(-(w_b + w_e) * _q / _k / temp);
                k_up = _k * temp / _h * w_exp_up;

                dp = 2 * P_s * (k_down * (1 - prob) - k_up * prob);

                I_p_new = dp * A;
                I_leak = (I_0 * A * std::expm1(v_tmp / V_t) + I_dsc) * std::copysign(1.0, v_tmp);

                dv = (synaptic_input_ptr[neuron] - I_leak - I_p_new) / C_tot;

                if (v_tmp <= threshold)
                {
                    v_tmp = v_tmp + 0.001 * dt * dv;
                    p_tmp = p_tmp + 0.001 * dt * dp;
                }

                if (v_tmp > 4)
                    v_tmp = 4;
                else if (v_tmp < -1)
                    v_tmp = -1;

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

    // Registers CUDA implementations for mymuladd, mymul, myadd_out
    TORCH_LIBRARY_IMPL(eleanor, CPU, m)
    {
        m.impl("heracles", &heracles_cpu);
    }

}

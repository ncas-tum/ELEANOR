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
                                         double I_dsc, double _eps0, double _q, double _k, double _h,
                                         double threshold, double dt, double paramsScale, int64_t nsteps)
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

        const float * __restrict__ v_ptr                = v_contig.data_ptr<float>();
        const float * __restrict__ p_ptr                = p_contig.data_ptr<float>();
        const float * __restrict__ synaptic_input_ptr   = synaptic_input_contig.data_ptr<float>();

        const float * __restrict__ A_ptr                = A_contig.data_ptr<float>();
        const float * __restrict__ t_fe_ptr             = t_fe_contig.data_ptr<float>();
        const float * __restrict__ n_depl_ptr           = n_depl_contig.data_ptr<float>();
        const float * __restrict__ P_s_ptr              = P_s_contig.data_ptr<float>();

        at::Tensor v_result = torch::empty(v_contig.sizes(), v_contig.options());
        at::Tensor p_result = torch::empty(v_contig.sizes(), v_contig.options());
        float * __restrict__ v_result_ptr = v_result.data_ptr<float>();
        float * __restrict__ p_result_ptr = p_result.data_ptr<float>();

        // const float eps_fe_f     = static_cast<float>(eps_fe);
        // const float eps_depl_f     = static_cast<float>(eps_depl);
        // const float q_fix_depl_f    = static_cast<float>(q_fix_depl);
        // const float e_off_f       = static_cast<float>(e_off);
        // const float temp_f     = static_cast<float>(temp);
        // const float w_b_f        = static_cast<float>(w_b);
        // const float d_e_f = static_cast<float>(d_e);
        // const float I_0_f = static_cast<float>(I_0);
        // const float V_t_f = static_cast<float>(V_t);
        // const float C_par_f = static_cast<float>(C_par);
        // const float C_fe_f = static_cast<float>(C_fe);
        // const float I_dsc_f = static_cast<float>(I_dsc);
        // const float _eps0_f = static_cast<float>(_eps0);
        // const float _q_f = static_cast<float>(_q);
        // const float _k_f = static_cast<float>(_k);
        // const float _h_f = static_cast<float>(_h);
        // const float threshold_f = static_cast<float>(threshold);
        const float dt_f = static_cast<float>(dt);
        // const float paramsScale_f = static_cast<float>(paramsScale);

        const float int_div     = 1.f / static_cast<float>(nsteps);
        const float sub_dt      = int_div * dt_f;
        constexpr float V_CLIP  = 5.f;
        const int64_t numel     = v.numel();

        // omp_set_num_threads(omp_get_max_threads());

#pragma omp parallel for if(numel > 64) schedule(static)
        for (int64_t neuron = 0; neuron < v.numel(); neuron++)
        {
            const float  A = A_ptr[neuron];
            const float  t_fe = t_fe_ptr[neuron];
            const float  n_depl = n_depl_ptr[neuron];
            const float  P_s = P_s_ptr[neuron];

            float v_tmp = v_ptr[neuron];
            float p_tmp = p_ptr[neuron];

            for (int64_t t = 0; t < nsteps; t++)
            {
                // Calculate cap and depol dividers
                const float prob = p_tmp / 2 / P_s + 0.5;
                const float e_dummy = v_tmp / t_fe;

                const float w_depl_d = ((_eps0 * eps_fe * e_dummy + q_fix_depl) * paramsScale / _q / n_depl);
                const float w_depl_u_signed = (_eps0 * eps_fe * e_dummy - q_fix_depl) * paramsScale / _q / n_depl;
                const float w_depl_u = std::abs(w_depl_u_signed);

                const float w_depl_denom = prob * w_depl_u + (1 - prob) * w_depl_d;
                const float w_depl = w_depl_d * w_depl_u / w_depl_denom;

                const float C_tot = 1 / (1 / (C_fe + C_par) + 1 / (_eps0 * eps_depl / w_depl * A));
                const float cap_divider = eps_depl / (t_fe * eps_depl + w_depl * eps_fe);
                const float depol_divider = 1 / _eps0 * w_depl / (t_fe * eps_depl + w_depl * eps_fe);

                // FeLIF equation
                const float E = v_tmp * cap_divider - p_tmp * depol_divider;
                const float w_e = (E - e_off) * d_e;
                const float w_exp_down = std::exp(-(w_b - w_e) * _q / _k / temp);
                const float k_down = _k * temp / _h * w_exp_down;
                const float w_exp_up = std::exp(-(w_b + w_e) * _q / _k / temp);
                const float k_up = _k * temp / _h * w_exp_up;

                const float dp = 2 * P_s * (k_down * (1 - prob) - k_up * prob);

                const float sign_v = static_cast<float>((v_tmp > 0.f) - (v_tmp < 0.f));
                const float I_p_new = dp * A;
                const float I_leak = (I_0 * A * std::expm1(v_tmp / V_t) + I_dsc) * sign_v;

                const float dv = (synaptic_input_ptr[neuron] - I_leak - I_p_new) / C_tot;

                if (v_tmp < threshold)
                {
                    v_tmp = v_tmp + sub_dt * dv;
                    p_tmp = p_tmp + sub_dt * dp;
                }

                v_tmp = std::clamp(v_tmp, -V_CLIP, V_CLIP);
                p_tmp = std::clamp(p_tmp, -P_s, P_s);
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

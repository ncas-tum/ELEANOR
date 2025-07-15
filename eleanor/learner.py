import torch
import torch.nn as nn
from snntorch.functional import probe


class TwoPhasePlasticity(nn.Module):
    # Initialization parameters
    ica_0: float = 17e-12
    v_h0: float = 900e-3
    z_0: float = 0.0
    dt: float = 1e-3

    # Calcium
    tau_DPI: float = 4.88e-3  # For dt = 1e-3
    delta_capre: float = 60e-12
    delta_capost: float = 16.55e-12
    I_TH: float = 10e-12
    I_INDC: float = 25e-12
    I_TAU: float = 20e-12

    # Early-phase
    C: float = 1.2215e-12
    I_THPOT: float = 62e-12
    I_THDEP: float = 39.5e-12
    I_TAILP: float = 50e-12
    I_TAILP_low: float = 1.2e-15
    I_TAILD: float = 10e-12
    I_TAILD_low: float = 0.8e-15
    i_hrn: float = 2.5e-15
    i_hrp: float = 2.5e-15

    # Late-phase
    tau_z_c: float = 5
    z_max: float = 1
    z_min: float = -0.5
    theta_tag_c: float = 0.0151226  # 1226
    alpha: float = 1.0
    theta_pro_circuit: float = 0.45
    # h_0: float = 4.20075e-3

    # Total weight
    # v_H0: float = 0.9
    beta: float = 4.6675e-3

    def __init__(
        self,
        synapse: nn.Linear,
        sn,
    ):
        super().__init__()
        self.synapse = synapse
        self.in_spike_monitor = probe.InputMonitor(synapse)
        self.out_spike_monitor = probe.OutputMonitor(sn)

        self.register_buffer("i_ca", torch.empty(0), False)
        self.register_buffer("v_h", torch.empty(0), False)
        self.register_buffer("z_ji", torch.empty(0), False)
        self.register_buffer("p_i", torch.empty(0), False)

        torch.nn.init.constant_(self.synapse.weight.data, self.beta * self.v_h0)

    def reset(self):
        super(TwoPhasePlasticity, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def forward(self, on_grad: bool = True, scale: float = 1.0):
        self.step(on_grad, scale)

    def step(self, on_grad: bool = True, scale: float = 1.0):
        length = self.in_spike_monitor.records.__len__()

        for _ in range(length):
            in_spike = self.in_spike_monitor.records.pop(0)  # [batch_size, N_in]
            out_spike = self.out_spike_monitor.records.pop(0)  # [batch_size, N_out]
            if isinstance(out_spike, tuple):
                out_spike = out_spike[0]

            trace_shape = out_spike.shape + (in_spike.shape[1],)
            if not self.i_ca.shape == trace_shape:
                self.i_ca = (
                    torch.zeros(trace_shape, device=self.i_ca.device) + self.ica_0
                )

            if not self.v_h.shape == trace_shape:
                self.v_h = torch.zeros(trace_shape, device=self.v_h.device) + self.v_h0

            if not self.z_ji.shape == trace_shape:
                self.z_ji = torch.zeros(trace_shape, device=self.z_ji.device) + self.z_0

            if not self.p_i.shape == out_spike.shape:
                self.p_i = torch.zeros(out_spike.shape, device=self.p_i.device)

            # Calcium part ---------------------------------------------------------
            dica_dt = (1 / self.tau_DPI) * (
                -self.i_ca + self.I_TH * self.I_INDC / self.I_TAU
            )
            self.i_ca = self.i_ca + dica_dt * self.dt

            ica_pre_increase = in_spike * self.delta_capre
            ica_post_increase = out_spike * self.delta_capost
            self.i_ca = (
                self.i_ca
                + ica_pre_increase.unsqueeze(1)
                + ica_post_increase.unsqueeze(2)
            )

            # Early-phase part -----------------------------------------------------
            recovery_condition = self.v_h > self.v_h0
            potentiation_condition = self.i_ca > self.I_THPOT
            depression_condition = self.i_ca > self.I_THDEP

            i_recovery = torch.where(recovery_condition, -self.i_hrn, self.i_hrp)
            i_potentiation = torch.where(
                potentiation_condition, self.I_TAILP, self.I_TAILP_low
            )
            i_depression = torch.where(
                depression_condition, -self.I_TAILD, self.I_TAILD_low
            )

            self.v_h = (
                self.v_h
                + (1 / self.C) * (i_recovery + i_potentiation + i_depression) * self.dt
            )

            # Ensure v_h_new is not less than 0
            self.v_h = torch.clamp(self.v_h, 0, 5)
            # self.v_h = torch.clamp_min(self.v_h, 0)

            # Late-phase part ------------------------------------------------------
            epsilon_hi = torch.sum(torch.abs(self.v_h - self.v_h0), dim=2)
            condition = epsilon_hi > self.theta_pro_circuit
            self.p_i = torch.where(condition, self.alpha, self.p_i)
            pot_term = (
                self.p_i.unsqueeze(2)
                * (self.z_max - self.z_ji)
                * ((self.v_h - self.v_h0 - self.theta_tag_c) > 0)
            )
            dep_term = (
                self.p_i.unsqueeze(2)
                * (self.z_ji - self.z_min)
                * ((self.v_h0 - self.v_h - self.theta_tag_c) > 0)
            )
            dz_dt = (1 / self.tau_z_c) * (pot_term - dep_term)
            self.z_ji = self.z_ji + dz_dt * self.dt

            # Ensure z_ji_new is not less than 0
            self.z_ji = torch.clamp_min(self.z_ji, 0)

            # Total synaptic weight ------------------------------------------------
            self.synapse.weight.data = torch.mean(
                self.beta * self.v_h + self.beta * self.v_h0 * self.z_ji, dim=0
            )

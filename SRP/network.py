import os
import sys

import jax
import jax.numpy as jnp
from jax import vmap
from spyx.axn import tanh

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pizzo_neuron_jax import FeLIF  # noqa: E402
import time

# Network definition
def network_builder(
    # Network params
    # nb_inputs,
    nb_hidden,
    nb_outputs,
    nb_steps,
    # Encoding
    # enc_fan_out,
    alpha,
    beta,
    # FeLIF
    A,
    I_dsc,
    V_thr,
    P_s,
    spike_fn=tanh(k=10),  # noqa: B008
):
    
    felif_step, felif_reset = FeLIF(
            dt=1e-3,
            innerStep=1000,
            A=A,
            I_dsc=I_dsc,
            V_thr=V_thr,
            P_s=P_s,
            spike_fn=spike_fn,
            paramsScale=1e12,
        )
    
    @jax.jit
    def encoder_step(state, input_):
        enc, input_spk = state

        # Compute encoder activity
        new_enc = (beta * enc + (1.0 - beta) * input_) * (
            1.0 - jax.lax.stop_gradient(input_spk)
        )
        new_input_spk = spike_fn(enc - 1.0)

        return (new_enc, new_input_spk), new_input_spk

    @jax.jit
    def hidden_step(state, input_):
        syn, mem, out = state

        # Compute hidden layer activity
        # h1 = jnp.dot(input_, w1) #+ jnp.dot(out, v1)
        mthr = mem - 1.0
        new_out = spike_fn(mthr)
        rst = jax.lax.stop_gradient(
            new_out
        )  # We do not want to backprop through the reset

        new_syn = alpha * syn + input_
        new_mem = (beta * mem + (1.0 - beta) * syn) * (1.0 - rst)

        return (new_syn, new_mem, new_out), new_out

    @jax.jit
    def output_step(state, input_):
        syn, mem = state

        new_syn = alpha * syn + input_
        new_mem = beta * mem + (1.0 - beta) * syn
        return (new_syn, new_mem), new_mem

    @jax.jit
    def predict(params, input_):
        h1 = jnp.dot(input_, params[0])
        syn = jnp.zeros((nb_hidden,))
        mem = jnp.zeros((nb_hidden,))
        out = jnp.zeros((nb_hidden,))
        _, spk_rec = jax.lax.scan(hidden_step, (syn, mem, out), h1, nb_steps, unroll=1)

        h2 = jnp.dot(spk_rec, params[1])
        syn2 = jnp.zeros((nb_outputs,))
        mem2 = jnp.zeros((nb_outputs,))
        _, out_rec = jax.lax.scan(output_step, (syn2, mem2), h2, nb_steps, unroll=1)
        return out_rec, h2, spk_rec, h1
        _, (out_rec, charge, V_rec, P_Rec) = jax.lax.scan(
            felif_step,
            felif_reset(nb_outputs),
            h2,
            nb_steps,
            unroll=1,
        )
        # print(out_rec_felif.shape)
        # print(out_rec.shape)
    
        return out_rec, charge, V_rec, P_Rec, h2, spk_rec, h1

    batched_predict = vmap(predict, in_axes=(None, 0))

    return batched_predict

def network_builder_FELIF(
    # Network params
    # nb_inputs,
    nb_hidden,
    nb_outputs,
    nb_steps,
    # Encoding
    # enc_fan_out,
    alpha,
    beta,
    # FeLIF
    A,
    I_dsc,
    V_thr,
    P_s,
    spike_fn=tanh(k=10),  # noqa: B008
):
    
    felif_step, felif_reset = FeLIF(
            dt=1e-3,
            innerStep=1000,
            A=A,
            I_dsc=I_dsc,
            V_thr=V_thr,
            P_s=P_s,
            spike_fn=spike_fn,
            paramsScale=1e12,
        )
    
    @jax.jit
    def encoder_step(state, input_):
        enc, input_spk = state

        # Compute encoder activity
        new_enc = (beta * enc + (1.0 - beta) * input_) * (
            1.0 - jax.lax.stop_gradient(input_spk)
        )
        new_input_spk = spike_fn(enc - 1.0)

        return (new_enc, new_input_spk), new_input_spk

    @jax.jit
    def hidden_step(state, input_):
        syn, mem, out = state

        # Compute hidden layer activity
        # h1 = jnp.dot(input_, w1) #+ jnp.dot(out, v1)
        mthr = mem - 1.0
        new_out = spike_fn(mthr)
        rst = jax.lax.stop_gradient(
            new_out
        )  # We do not want to backprop through the reset

        new_syn = alpha * syn + input_
        new_mem = (beta * mem + (1.0 - beta) * syn) * (1.0 - rst)

        return (new_syn, new_mem, new_out), new_out

    @jax.jit
    def output_step(state, input_):
        syn, mem = state

        new_syn = alpha * syn + input_
        new_mem = beta * mem + (1.0 - beta) * syn
        return (new_syn, new_mem), new_mem

    @jax.jit
    def predict(params, input_):
        h1 = jnp.dot(input_, params[0])
        syn = jnp.zeros((nb_hidden,))
        mem = jnp.zeros((nb_hidden,))
        out = jnp.zeros((nb_hidden,))
        _, spk_rec = jax.lax.scan(hidden_step, (syn, mem, out), h1, nb_steps, unroll=1)

        h2 = jnp.dot(spk_rec, params[1])
        syn2 = jnp.zeros((nb_outputs,))
        mem2 = jnp.zeros((nb_outputs,))
        # _, out_rec = jax.lax.scan(output_step, (syn2, mem2), h2, nb_steps, unroll=1)
        # return out_rec, h2, spk_rec, h1
        _, (out_rec, charge, V_rec, P_Rec) = jax.lax.scan(
            felif_step,
            felif_reset(nb_outputs),
            h2,
            nb_steps,
            unroll=1,
        )
        # print(out_rec_felif.shape)
        # print(out_rec.shape)
    
        return out_rec, charge, V_rec, P_Rec, h2, spk_rec, h1

    batched_predict = vmap(predict, in_axes=(None, 0))

    return batched_predict

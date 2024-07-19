import os
import sys

import jax
import jax.numpy as jnp
from jax import vmap
from spyx.axn import custom, heaviside, tanh

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from pizzo_neuron_jax import FeLIF  # noqa: E402


def fast_sigm(k=1):
    def fast_sigmoid(x):
        kx = k * jnp.abs(x)
        return x / 1 + kx

    return custom(fast_sigmoid, heaviside)


# Network definition
def network_builder(
    # Network params
    # nb_inputs,
    nb_hidden,
    nb_outputs,
    nb_steps,
    # Encoding
    # enc_fan_out,
    beta,
    # FeLIF
    A,
    I_dsc,
    V_thr,
    P_s,
    spike_fn=fast_sigm(k=100),  # noqa: B008
):

    felif_step, felif_reset = FeLIF(
        dt=14e-3,
        innerStep=1000,
        A=A,
        I_dsc=I_dsc,
        V_thr=V_thr,
        P_s=P_s,
        spike_fn=spike_fn,
        paramsScale=1e12,
    )

    @jax.jit
    def hidden_step(state, input_):
        mem, syn = state

        # Compute hidden layer activity
        # h1 = jnp.dot(input_, w1) #+ jnp.dot(out, v1)
        mthr = mem - 1.0
        new_out = spike_fn(mthr)
        rst = jax.lax.stop_gradient(
            new_out
        )  # We do not want to backprop through the reset

        new_syn = 0.135 * syn + input_
        new_mem = (beta * mem + syn) * (1.0 - rst)

        return (new_mem, new_syn), new_out

    @jax.jit
    def output_step(state, input_):
        mem, syn = state

        new_syn = 0.135 * syn + input_
        new_mem = beta * mem + syn
        return (new_mem, new_syn), new_mem

    @jax.jit
    def predict(params, input_):
        h1 = jnp.dot(input_, params[0])
        # mem = jnp.zeros((nb_hidden,))
        # syn = jnp.zeros((nb_hidden,))

        # _, spk_rec = jax.lax.scan(hidden_step, (mem,syn), h1, nb_steps, unroll=1)
        _, (spk_rec, charge, V_rec, P_Rec) = jax.lax.scan(
            felif_step,
            felif_reset(nb_hidden),
            h1,
            nb_steps,
            unroll=1,
        )
        # print(out_rec_felif.shape)
        # print(out_rec.shape)

        h2 = jnp.dot(spk_rec, params[1])
        mem2 = jnp.zeros((nb_outputs,))
        syn2 = jnp.zeros((nb_outputs,))
        _, out_rec = jax.lax.scan(output_step, (mem2, syn2), h2, nb_steps, unroll=1)
        # return out_rec, h2, spk_rec, h1
        return out_rec, charge, V_rec, P_Rec, h2, spk_rec, h1

    batched_predict = vmap(predict, in_axes=(None, 0))

    return batched_predict

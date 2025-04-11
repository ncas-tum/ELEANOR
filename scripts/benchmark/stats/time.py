import time
from typing import Union, Callable, Optional, Sequence

import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrand
from chex import Array, PRNGKey

import snnax.snn as snn
from eleanor.models import FeLIF, NoBruno, Heracles
from snnax.snn.composed import StateShape
from snnax.snn.layers.stateful import StatefulOutput, default_init_fn
from snnax.functional.surrogate import SpikeFn, superspike_surrogate

jax.config.update("jax_platform_name", "cpu")


class Tanh(eqx.Module):

    def __call__(self, x, *, key=None):
        return jnp.tanh(x)


_spikefn = superspike_surrogate(10.0)


class RLIF(snn.LIF):
    recurrent: eqx.Module

    def __init__(
        self,
        n_features,
        decay_constants: Union[Sequence[float], Array],
        spike_fn: SpikeFn = _spikefn,
        threshold: Array = 1.0,
        stop_reset_grad: bool = True,
        reset_val: Optional[Array] = None,
        init_fn: Optional[Callable] = default_init_fn,
        shape: Optional[StateShape] = None,
        use_bias: Optional[bool] = False,
        key: Optional[PRNGKey] = None,
    ) -> None:

        key, krec = jrand.split(key, 2)
        super().__init__(
            decay_constants,
            spike_fn,
            threshold,
            stop_reset_grad,
            reset_val,
            init_fn,
            shape,
            key,
        )
        self.recurrent = eqx.nn.Linear(
            n_features, n_features, use_bias=use_bias, key=krec
        )

    def __call__(
        self,
        state: Sequence[Array],
        synaptic_input: Array,
        *,
        key: Optional[PRNGKey] = None,
    ) -> StatefulOutput:
        mem_pot, syn_curr, spike_output = state

        if self.reset_val is None:
            reset_pot = mem_pot * spike_output
        else:
            reset_pot = (mem_pot - self.reset_val) * spike_output

        # Optionally stop gradient propagation through refectory potential
        refectory_potential = (
            jax.lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        )
        mem_pot = mem_pot - refectory_potential

        alpha = jax.lax.clamp(0.5, self.decay_constants[0], 1.0)
        beta = jax.lax.clamp(0.5, self.decay_constants[1], 1.0)

        mem_pot = alpha * mem_pot + (1.0 - alpha) * (syn_curr)
        syn_curr = beta * syn_curr + (1.0 - beta) * (
            synaptic_input + self.recurrent(spike_output, key=key)
        )

        spike_output = self.spike_fn(mem_pot - self.threshold)

        state = [mem_pot, syn_curr, spike_output]
        return [state, spike_output]


def setup(key, hidden_size, seq_len, input_features, model_name):
    input_key, key_model, key_linear = jax.random.split(key, 3)

    if model_name == "LIF":
        outputLayer = snn.LIF([0.9], key=key_model)
    elif model_name == "RLIF":
        outputLayer = RLIF(hidden_size, [0.9], key=key_model)
    elif model_name == "Bruno":
        outputLayer = FeLIF(
            dt=1e-3, V_thr=2.5, P_s=0.22, paramsScale=1e9, key=key_model
        )
    elif model_name == "FeLIF":
        outputLayer = NoBruno(
            dt=1e-6, V_thr=2.5, P_s=0.22, paramsScale=1e9, key=key_model
        )
    elif model_name == "Heracles":
        outputLayer = Heracles(
            dt=1e-3, V_thr=2.5, P_s=0.22, paramsScale=1e9, key=key_model
        )
    else:
        raise Exception("Model not supported")

    # Initialize model
    model = snn.Sequential(
        eqx.nn.Linear(input_features, hidden_size, use_bias=False, key=key_linear),
        Tanh(),
        outputLayer,
    )

    dummy_input = jax.random.normal(input_key, (seq_len, input_features))

    return model, dummy_input


def time_forward(hidden_size, seq_len, input_features, model_name):
    key = jrand.key(0)
    key_setup, key = jrand.split(key)

    model, dummy_input = setup(
        key_setup, hidden_size, seq_len, input_features, model_name
    )

    init_key, run_key = jax.random.split(key)
    init_state = model.init_state((input_features,), key=init_key)
    _, output = model(init_state, dummy_input, key=run_key)
    output = jax.tree.map(lambda x: x.block_until_ready(), output)

    # Timing loop
    times = []
    for _ in range(100):
        start = time.process_time_ns()
        _, output = model(init_state, dummy_input, key=run_key)
        output = jax.tree.map(lambda x: x.block_until_ready(), output)
        end = time.process_time_ns()
        avg_time = end - start  # Convert to milliseconds
        times.append(avg_time)

    return times


@eqx.filter_jit
def loss_lif(output):
    return jnp.sum(output)


@eqx.filter_jit
def loss_felif(output):
    return jnp.sum(output[0])


@eqx.filter_grad
def loss_fn(model, input_, loss_neuron, *, key):
    # Initialize parameters and run forward pass
    init_key, run_key = jax.random.split(key)
    init_state = model.init_state((input_features,), key=init_key)
    _, output = model(init_state, input_, key=run_key)

    loss = loss_neuron(output[-1])
    return loss


def time_backward(hidden_size, seq_len, input_features, model_name):
    key = jrand.key(0)
    key_setup, key = jrand.split(key)

    model, dummy_input = setup(
        key_setup, hidden_size, seq_len, input_features, model_name
    )
    loss_neuron = (
        loss_felif if model_name == "FeLIF" or model_name == "Bruno" else loss_lif
    )
    grads = loss_fn(model, dummy_input, loss_neuron, key=key)
    grads = jax.tree.map(lambda x: x.block_until_ready(), grads)

    # Timing loop
    times = []
    for _ in range(100):
        start = time.process_time_ns()
        grads = loss_fn(model, dummy_input, loss_neuron, key=key)
        grads = jax.tree.map(lambda x: x.block_until_ready(), grads)
        end = time.process_time_ns()
        avg_time = end - start  # Convert to milliseconds
        times.append(avg_time)

    return times


if __name__ == "__main__":
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", choices=["FeLIF", "LIF", "Bruno", "RLIF"], default="FeLIF"
    )
    parser.add_argument("--method", choices=["backward", "forward"], default="forward")
    parser.add_argument("--hidden", type=int, choices=[64, 128, 256])
    parser.add_argument("--seq_len", type=int)
    args = parser.parse_args()

    input_features = 32

    # Process tasks sequentially to avoid GPU memory conflicts
    if args.method == "forward":
        measure_memory = time_forward
    else:
        measure_memory = time_backward

    eqx.clear_caches()
    jax.clear_caches()
    total_time = measure_memory(args.hidden, args.seq_len, input_features, args.model)
    results = (args.hidden, args.seq_len, args.model, total_time)

    # Organize results for plotting
    hs, sl, mdl, t = results
    time_data = {
        "Hidden Size": [hs] * len(t),
        "Sequence Length": [sl] * len(t),
        # "Peak Memory Usage": [],
        "Time": t,
        "Model": [mdl] * len(t),
    }

    df = pd.DataFrame(time_data)
    df.to_csv(
        f"results/time-{args.method}-{args.hidden}-{args.seq_len}-{args.model}.csv",
        index=False,
    )

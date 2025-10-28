import time

import jax
import equinox as eqx
import jax.numpy as jnp

from eleanor.models.jax import Bruno, Checkpoint


class Sequential(eqx.Module):
    model: eqx.Module
    linear: eqx.nn.Linear
    hidden: int = eqx.field(static=True)

    def __init__(self, mode, hidden, key, *, checkpoints=None):
        k1, k2 = jax.random.split(key)
        if mode == "Bruno":
            self.model = Bruno(paramsScale=1e9, dt=1e-3, key=k1)
        elif mode == "Checkpoints":
            self.model = Checkpoint(
                paramsScale=1e9, dt=1e-6, checkpoints=checkpoints, key=k1
            )
        else:
            self.model = Checkpoint(paramsScale=1e9, dt=1e-6, checkpoints=None, key=k1)
        self.linear = eqx.nn.Linear(512, hidden, key=k2)
        self.hidden = hidden

    def outputs(self, x, scan_fn, key=None):
        def step(carry, x):
            carry, [s, v, p] = self.model(carry, self.linear(x))
            return carry, (s, v, p)

        state = self.model.init_state((self.hidden,), key=jax.random.key(0))
        _, (s, v, p) = scan_fn(step, state, x)
        return (s, v, p)

    def __call__(self, x, scan_fn, key=None):
        def step(carry, x):
            carry, [s, v, p] = self.model(carry, self.linear(x))
            return carry, s

        state = self.model.init_state((self.hidden,), key=jax.random.key(0))
        _, y = scan_fn(step, state, x)
        return y


# Loss function that uses scan
def loss_fn(params, xs):
    scan_fn = jax.lax.scan
    ys = params(xs, scan_fn=scan_fn)
    return jnp.sum(ys**2)


def measure_memory(mode, seq_len, hidden, checkpoints):
    """Measure memory for a single configuration"""
    # Create problem
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    params = Sequential(mode, hidden, key=k1, checkpoints=checkpoints)
    xs = jax.random.normal(k2, (seq_len, 512))

    # Get baseline
    baseline = jax.devices()[0].memory_stats()["bytes_in_use"]

    # Compile and run
    grad_fn = eqx.filter_jit(eqx.filter_grad(lambda p: loss_fn(p, xs)))
    grads = grad_fn(params)
    jax.block_until_ready(grads)

    # Measure peak during this specific execution
    peak = jax.devices()[0].memory_stats()["peak_bytes_in_use"]
    current = jax.devices()[0].memory_stats()["bytes_in_use"]

    start_time = time.process_time()
    for _ in range(10):
        grads = grad_fn(params)
        jax.block_until_ready(grads)
    end_time = time.process_time()
    backward_time = end_time - start_time

    loss = loss_fn(params, xs)
    jax.block_until_ready(loss)
    start_time = time.process_time()
    for _ in range(10):
        loss = loss_fn(params, xs)
        jax.block_until_ready(loss)
    end_time = time.process_time()
    forward_time = end_time - start_time

    return baseline, current, peak, forward_time / 10, backward_time / 10


if __name__ == "__main__":
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )
    parser.add_argument(
        "-m", "--model", choices=["Bruno", "Vanilla", "Checkpoints"], default="Bruno"
    )
    parser.add_argument("-s", "--sequence", type=int, default=10)
    parser.add_argument("-hs", "--hidden", type=int, default=64)
    parser.add_argument("-ch", "--checkpoints", type=int, default=31)
    args = parser.parse_args()

    baseline, current, peak, forward_time, backward_time = measure_memory(
        args.model, args.sequence, args.hidden, checkpoints=args.checkpoints
    )
    data = pd.DataFrame(
        {
            "Method": args.model,
            "Hidden size": args.hidden,
            "Sequence length": args.sequence,
            "Baseline": baseline,
            "Current": current,
            "Peak": peak,
            "Used": current - baseline,
            "Inference time": forward_time,
            "Backward time": backward_time,
        },
        index=[0],
    )
    data.to_csv(f"metrics/{time.time()}")

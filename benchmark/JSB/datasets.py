import pickle
from typing import List, Tuple

import numpy as np
import jax.numpy as jnp
import jax.random as jrand
from chex import PRNGKey
from jaxtyping import Array


def pad_sequences(
    sequences: List[Array], padding_value: float = 0.0
) -> Tuple[Array, Array]:
    """
    Pad sequences to the same length.

    Returns:
        padded: (batch_size, max_seq_len, features)
        mask: (batch_size, max_seq_len) - True for real data, False for padding
    """
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    batch_size = len(sequences)
    feature_dim = sequences[0].shape[-1]

    # Create padded array
    padded = np.full((batch_size, max_len, feature_dim), padding_value)
    # mask = np.zeros((batch_size, max_len), dtype=bool)
    mask = np.zeros((batch_size,))

    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        padded[i, :length] = seq
        # mask[i, :length] = True
        mask[i] = length

    return jnp.array(padded), jnp.array(mask)


class JSBChorales:
    def __init__(
        self,
        path: str,
        shuffle: bool = False,
        key: PRNGKey | None = None,
    ):
        with open(path + ".pkl", "rb") as f:
            self.data = pickle.load(f)
        self.num_samples = len(self.data)
        self.indices = jnp.arange(self.num_samples)
        self.shuffle = shuffle
        self.key = key if key is not None else jrand.key(0)

    def __iter__(self):
        if self.shuffle:
            self.key, key = jrand.split(self.key)
            indices = jrand.permutation(key, self.indices)
        else:
            indices = self.indices

        for idx in range(self.num_samples):
            batch_indices = indices[idx]
            sample = self.data[batch_indices][:, 22:76]
            x = sample[:-1]
            y = sample[1:]
            yield x, y

        # for start_idx in range(0, self.num_samples, self.batch_size):
        #     end_idx = min(start_idx + self.batch_size, self.num_samples)
        #     batch_indices = indices[start_idx:end_idx]

        #     batch_sequences = [self.data[i] for i in batch_indices]
        #     padded, mask = pad_sequences(batch_sequences)
        #     yield padded, mask

    def __len__(self):
        # return (self.num_samples + self.batch_size - 1) // self.batch_size
        return self.num_samples


if __name__ == "__main__":
    import os

    dataset = JSBChorales(
        os.path.join(os.path.dirname(__file__), "data/JSB_test"),
        shuffle=False,
    )
    print(next(iter(dataset)))

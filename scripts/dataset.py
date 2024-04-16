import pickle

import jax
import jax.numpy as jnp
import numpy as np


def shuffle(dataset, shuffle_rng, batch_size):
    x, y = dataset

    cutoff = y.shape[0] % batch_size

    obs = jax.random.permutation(shuffle_rng, x, axis=0)[:-cutoff]
    labels = jax.random.permutation(shuffle_rng, y, axis=0)[:-cutoff]

    obs = jnp.reshape(obs, (-1, batch_size) + obs.shape[1:])
    labels = jnp.reshape(labels, (-1, batch_size))  # should make batch size a global

    return (obs, labels)


def loadNMNIST():
    pass


def loadBraille(nb_upsample, nb_repetitions):
    file_name = "/home/p306945/data/reading_braille_data/data/data_braille_letters_raw"
    with open(file_name, "rb") as infile:
        data_dict = pickle.load(infile)

    letter_written = [
        "Space",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]

    # Extract data
    data = []
    labels = []
    for i, _ in enumerate(letter_written):
        for repetition in np.arange(nb_repetitions):
            idx = i * nb_repetitions + repetition
            dat = 1.0 - data_dict[idx]["taxel_data"][:] / 255
            data.append(dat)
            labels.append(i)

    # Crop to same length
    data_steps = l = np.min([len(d) for d in data])
    data = jnp.array([d[:l] for d in data])
    labels = jnp.array(labels)

    # Select nonzero inputs
    nzid = [1, 2, 6, 10]
    data = data[:, :, nzid]

    # Standardize data
    rshp = data.reshape((-1, data.shape[2]))
    data = (data - rshp.mean(0)) / (rshp.std(0) + 1e-3)

    # Upsample
    def upsample(data, n=2):
        shp = data.shape
        tmp = jnp.tile(data, (1, 1, 1, n))
        return tmp.reshape((shp[0], n * shp[1], shp[2]))

    data = upsample(data, n=nb_upsample)

    data = jax.random.permutation(jax.random.PRNGKey(0), data, axis=0)
    labels = jax.random.permutation(jax.random.PRNGKey(0), labels, axis=0)

    a = int(0.8 * len(labels))
    x_train, x_test = data[:a], data[a:]
    y_train, y_test = labels[:a], labels[a:]

    nb_channels = len(nzid)
    time_step = (
        2e-3 / nb_upsample
    )  # TODO needs to be updated to reflect the correct time scale
    nb_steps = (
        nb_upsample * data_steps
    )  # TODO We should change this and upsample the input data
    nb_outputs = len(np.unique(labels)) + 1

    return (
        (x_train, y_train),
        (x_test, y_test),
        nb_outputs,
        nb_channels,
        nb_steps,
        time_step,
    )

import pickle
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from urllib.request import urlopen

import numpy as np
from scipy import signal
from datasets import Array2D, Dataset, Features, ClassLabel


def create_dataset(nb_upsample=2, nb_repetitions=200, shuffle=False, seed=42):
    file_name = Path("/tmp/braille/data/data_braille_letters_raw")
    if not file_name.exists():
        resp = urlopen(
            "https://zenodo.org/records/6556273/files/reading_braille_data.zip"
        )
        with ZipFile(BytesIO(resp.read())) as zObject:
            zObject.extract("data/data_braille_letters_raw", path="/tmp/braille")

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
    for repetition in np.arange(nb_repetitions):
        for i, _ in enumerate(letter_written):
            idx = i * nb_repetitions + repetition
            dat = 1.0 - data_dict[idx]["taxel_data"][:] / 255
            data.append(dat)
            labels.append(i)

    # Crop to same length
    data_steps = l = int(np.min([len(d) for d in data]))
    data = np.array([d[:l] for d in data])
    labels = np.array(labels)

    # Select nonzero inputs
    nzid = [1, 2, 6, 10]
    data = data[:, :, nzid]

    # Standardize data
    rshp = data.reshape((-1, data.shape[2]))
    data = (data - rshp.mean(0)) / (rshp.std(0) + 1e-3)

    # Upsample
    def upsample(data, n):
        data_dummy = signal.resample(data, int(data_steps * n))  # upsample

        return data_dummy

    upsampled_data = [upsample(d, nb_upsample) for d in data]
    data = np.stack(upsampled_data)
    # data = upsample(data, nb_upsample)

    # if shuffle:
    #     data = jax.random.permutation(key, data, axis=0)
    #     labels = jax.random.permutation(key, labels, axis=0)

    nb_channels = len(nzid)
    time_step = (
        2e-3 / nb_upsample
    )  # TODO needs to be updated to reflect the correct time scale
    nb_steps = (
        nb_upsample * data_steps
    )  # TODO We should change this and upsample the input data
    nb_outputs = len(np.unique(labels))  # + 1

    clslabel = ClassLabel(nb_outputs, names=letter_written)
    features = Features(
        {
            "data": Array2D(shape=(nb_steps, nb_channels), dtype="float32"),
            "label": clslabel,
        }
    )

    ds = Dataset.from_dict({"data": data, "label": labels}, features=features)

    a = int(0.8 * len(labels))
    if shuffle:
        ds = ds.train_test_split(
            train_size=a, shuffle=shuffle, stratify_by_column="label", seed=seed
        )
    else:
        ds = ds.train_test_split(train_size=a, shuffle=shuffle)
    # ds = ds.with_format("jax")

    return (
        ds,
        nb_outputs,
        nb_channels,
        nb_steps,
        time_step,
    )

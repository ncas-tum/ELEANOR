import glob

import pandas as pd

dataframes = []
for file in glob.glob("results/time-forward-*.csv"):
    # _, method, hs, seq, model = file.split('.')[0].split('-')
    dataframes.append(pd.read_csv(file))
df = pd.concat(dataframes, ignore_index=True)
df.to_csv("results/time-forward.csv", index=False)


dataframes = []
for file in glob.glob("results/time-backward-*.csv"):
    # _, method, hs, seq, model = file.split('.')[0].split('-')
    dataframes.append(pd.read_csv(file))
df = pd.concat(dataframes, ignore_index=True)
df.to_csv("results/time-backward.csv", index=False)

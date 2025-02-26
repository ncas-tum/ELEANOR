#!/bin/bash

models=("FeLIF" "LIF" "RLIF" "Heracles")
qvalues=("FP" 8 4 3)

for model in "${models[@]}"
do
    for qval in "${qvalues[@]}"
    do
        # echo $model $qval
        # echo "${qval}bit $model"
        optuna create-study --storage sqlite:///bruno.db --study-name "${qval}bit $model" --skip-if-exists --direction maximize
        sbatch send.sh --model $model --quantization $qval
    done
done

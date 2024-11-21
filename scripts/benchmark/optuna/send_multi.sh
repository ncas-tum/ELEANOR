#!/bin/bash

models=("LIF" "FeLIF" "Heracles" "RLIF")
qvalues=("FP" 4 8 3)

for model in "${models[@]}"
do
    for qval in "${qvalues[@]}"
    do
        # echo $model $qval
        # echo "${qval}bit $model"
        optuna create-study --storage sqlite:///bruno_results.db --study-name "${qval}bit $model" --skip-if-exists --direction maximize
        sbatch send.sh --model $model --quantization $qval
    done
done

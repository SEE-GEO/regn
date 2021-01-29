#!/usr/bin/env bash
#SBATCH -A C3SE508-19-3 -p chair
#SBATCH -t 0-24:00:00
#SBATCH --gres=gpu:1        # allocates 1 GPU of either type

TRAINING_DATA=${HOME}/src/regn/data/gprof_gmi_data_00.nc
VALIDATION_DATA=${HOME}/src/regn/data/gprof_gmi_data_80.nc
MODEL_PATH=${HOME}/src/regn/models/

cd ${HOME}/src/regn/scripts
#source ${HOME}/src/regn/scripts/setup_vera.sh

python train_qrnn.py  ${TRAINING_DATA} ${VALIDATION_DATA} ${MODEL_PATH} --n_neurons=256 --n_layers=12 --sensor=gmi --batch_norm

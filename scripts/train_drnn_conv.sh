#!/usr/bin/env bash
#SBATCH -A C3SE508-19-3 -p chair
#SBATCH -t 0-32:00:00
#SBATCH --gres=gpu:1        # allocates 1 GPU of either type

TRAINING_DATA="sftp://129.16.35.202/mnt/array1/share/MLDatasets/gprof/conv/training_data"
VALIDATION_DATA="sftp://129.16.35.202/mnt/array1/share/MLDatasets/gprof/conv/validation_data"
MODEL_PATH=${HOME}/src/regn/models/

cd ${HOME}/src/regn/scripts
source ${HOME}/src/regn/scripts/setup_vera.sh

python train_drnn_conv.py  ${TRAINING_DATA} ${VALIDATION_DATA} ${MODEL_PATH} 

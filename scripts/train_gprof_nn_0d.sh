#!/usr/bin/env bash
#SBATCH -A C3SE508-19-3 -p chair
#SBATCH -t 0-48:00:00
#SBATCH --gres=gpu:1        # allocates 1 GPU of either type
#SBATCH --job-name=qrnn-4-256

TRAINING_DATA=/gdata/simon/gprof_0d/era5/gmi/training_data_small
#TRAINING_DATA=/home/simonpf/src/regn/data/training_data_small
VALIDATION_DATA=/gdata/simon/gprof_0d/era5/gmi/validation_data_small
#VALIDATION_DATA=/home/simonpf/src/regn/data/validation_data_small

MODEL_PATH=${HOME}/src/regn/models/
TARGETS="surface_precip convective_precip rain_water_path ice_water_path cloud_water_path cloud_water_content snow_water_content rain_water_content latent_heat"

export QUANTNN_LOG_LEVEL=INFO
export OMP_NUM_THREADS=4

cd ${HOME}/src/regn/scripts


<<<<<<< Updated upstream
python train_gprof_nn_0d.py  ${TRAINING_DATA} ${VALIDATION_DATA} ${MODEL_PATH} --n_neurons 256 --n_layers_body 8 --n_layers_head 1 --device cuda:0 --targets ${TARGETS} --type qrnn --batch_size 512
=======
python train_gprof_nn_0d.py  ${TRAINING_DATA} ${VALIDATION_DATA} ${MODEL_PATH} --n_neurons 256 --n_layers_body 6 --n_layers_head 3 --device cuda:1 --targets ${TARGETS} --type qrnn --batch_size 1024
>>>>>>> Stashed changes

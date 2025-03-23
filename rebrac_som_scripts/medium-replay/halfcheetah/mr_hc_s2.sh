#!/bin/bash -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/Bellman-Diffusion-Offline-RL/
conda init bash
conda activate offline_2

echo "hello"

python setup.py install




/home/liam/Desktop/offline_rl/OfflineRL-Kit/pyrallis_scripts/run_rebrac_som.py --config=/home/liam/Desktop/offline_rl/OfflineRL-Kit/pyrallis_scripts/configs/offline/rebrac/halfcheetah/medium-replay_v2.yaml --train_seed=2
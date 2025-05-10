#!/bin/bash -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/offline/Bellman-Diffusion-Offline-RL/
# conda init bash
conda activate offline

echo "hello"

#python setup.py install



cd /common/home/lbs105/Desktop/offline/Bellman-Diffusion-Offline-RL/tuning/optuna_sequential_rebrac_som_no_q_scripts/medium-replay/walker2d

sbatch -G 2 mr_walk_s0.sh
sleep 30s

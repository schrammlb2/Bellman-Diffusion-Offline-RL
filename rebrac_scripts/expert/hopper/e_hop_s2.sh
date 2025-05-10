#!/bin/bash -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/offline/Bellman-Diffusion-Offline-RL/
# conda init bash
conda activate offline

echo "hello"

#python setup.py install




python /common/home/lbs105/Desktop/offline/Bellman-Diffusion-Offline-RL/pyrallis_scripts/run_rebrac.py --config=/common/home/lbs105/Desktop/offline/Bellman-Diffusion-Offline-RL/pyrallis_scripts/configs/offline/rebrac/hopper/expert_v2.yaml --train_seed=2
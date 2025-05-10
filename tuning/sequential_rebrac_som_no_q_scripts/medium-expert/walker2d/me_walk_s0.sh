#!/bin/bash -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/offline/Bellman-Diffusion-Offline-RL/
# conda init bash
conda activate offline

echo "hello"

#python setup.py install




python /common/home/lbs105/Desktop/offline/Bellman-Diffusion-Offline-RL/pyrallis_scripts/seq_som_grid_search.py --config=/common/home/lbs105/Desktop/offline/Bellman-Diffusion-Offline-RL/pyrallis_scripts/configs/offline/rebrac/walker2d/medium-expert_v2.yaml --train_seed=0
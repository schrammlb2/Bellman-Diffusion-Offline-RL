#!/bin/bash -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/Bellman-Diffusion-Offline-RL/
conda init bash
conda activate offline_2

echo "hello"

python setup.py install




python pyrallis_scripts/run_rebrac_som.py --config=pyrallis_scripts/configs/offline/rebrac/halfcheetah/medium-expert_v2.yaml --train-seed=3
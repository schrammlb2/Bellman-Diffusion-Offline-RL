#!/bin/bash -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/Bellman-Diffusion-Offline-RL/
conda init bash
conda activate offline_2

echo "hello"

python setup.py install




python pyrallis_scripts/run_rebrac.py --config=pyrallis_scripts/configs/offline/rebrac/walker2d/medium-replay_v2.yaml --train-seed=2
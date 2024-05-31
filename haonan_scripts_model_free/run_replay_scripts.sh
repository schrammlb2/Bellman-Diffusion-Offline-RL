#!/bin/bash/ -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/Bellman-Diffusion-Offline-RL/haonan_scripts/replay/
conda init bash
conda activate offline_2
echo "hello"

bash halfcheetah_replay.sh
sleep 30s
bash walker2d_replay.sh
sleep 30s
bash hopper_replay.sh

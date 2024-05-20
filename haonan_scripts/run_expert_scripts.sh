#!/bin/bash/ -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/Bellman-Diffusion-Offline-RL/haonan_scripts/expert/
conda init bash
conda activate offline_2
echo "hello"

sbatch -G 1 halfcheetah_expert.sh
sleep 30s
sbatch -G 1 walker2d_expert.sh
sleep 30s
sbatch -G 1 hopper_expert.sh

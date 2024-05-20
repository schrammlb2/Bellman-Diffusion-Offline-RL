#!/bin/bash/ -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/Bellman-Diffusion-Offline-RL/haonan_scripts/medium/
conda init bash
conda activate offline_2
echo "hello"

sbatch -G 1 halfcheetah.sh
sleep 30s
sbatch -G 1 walker2d.sh
sleep 30s
sbatch -G 1 hopper.sh

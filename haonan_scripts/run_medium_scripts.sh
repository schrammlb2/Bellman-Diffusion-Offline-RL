#!/bin/bash/ -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/Bellman-Diffusion-Offline-RL/haonan_scripts/medium/
conda init bash
conda activate offline_2
echo "hello"

bash halfcheetah.sh
sleep 30s
bash walker2d.sh
sleep 30s
bash hopper.sh

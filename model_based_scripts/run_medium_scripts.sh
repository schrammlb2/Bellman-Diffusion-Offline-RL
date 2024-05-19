#!/bin/bash/ -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd /common/home/lbs105/Desktop/Bellman-Diffusion-Offline-RL
conda init bash
conda activate offline_2
echo "hello"

sbatch -G 1 halfcheetah.sh
sleep 30s
sbatch -G 1 walker2d.sh
sleep 30s
sbatch -G 1 hopper.sh

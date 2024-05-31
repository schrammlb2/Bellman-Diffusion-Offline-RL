#!/bin/bash/ -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/Bellman-Diffusion-Offline-RL/haonan_scripts/expert/walker
conda init bash
conda activate offline_2
echo "hello"

sbatch -G 1 walk_e_s1.sh
sleep 30s
sbatch -G 1 walk_e_s2.sh
sleep 30s
sbatch -G 1 walk_e_s3.sh
sleep 30s
sbatch -G 1 walk_e_s4.sh

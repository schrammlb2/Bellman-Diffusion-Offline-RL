#!/bin/bash -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/Bellman-Diffusion-Offline-RL/
conda init bash
conda activate offline_2

echo "hello"

python setup.py install



cd ~/Desktop/Bellman-Diffusion-Offline-RL//home/liam/Desktop/offline_rl/OfflineRL-Kit/rebrac_som_scripts/expert/walker2d

sbatch -G 1 e_walk_s0.sh
sleep 30s

sbatch -G 1 e_walk_s1.sh
sleep 30s

sbatch -G 1 e_walk_s2.sh
sleep 30s

sbatch -G 1 e_walk_s3.sh
sleep 30s

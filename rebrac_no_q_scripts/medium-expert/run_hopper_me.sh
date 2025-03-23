#!/bin/bash -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/Bellman-Diffusion-Offline-RL/
conda init bash
conda activate offline_2

echo "hello"

python setup.py install



cd ~/Desktop/Bellman-Diffusion-Offline-RL//home/liam/Desktop/offline_rl/OfflineRL-Kit/rebrac_no_q_scripts/medium-expert/hopper

sbatch -G 1 me_hop_s0.sh
sleep 30s

sbatch -G 1 me_hop_s1.sh
sleep 30s

sbatch -G 1 me_hop_s2.sh
sleep 30s

sbatch -G 1 me_hop_s3.sh
sleep 30s

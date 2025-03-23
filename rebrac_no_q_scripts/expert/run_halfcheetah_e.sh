#!/bin/bash -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/Bellman-Diffusion-Offline-RL/
conda init bash
conda activate offline_2

echo "hello"

python setup.py install



cd ~/Desktop/Bellman-Diffusion-Offline-RL//home/liam/Desktop/offline_rl/OfflineRL-Kit/rebrac_no_q_scripts/expert/halfcheetah

sbatch -G 1 e_hc_s0.sh
sleep 30s

sbatch -G 1 e_hc_s1.sh
sleep 30s

sbatch -G 1 e_hc_s2.sh
sleep 30s

sbatch -G 1 e_hc_s3.sh
sleep 30s

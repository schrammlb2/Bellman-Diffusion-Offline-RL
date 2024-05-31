#!/bin/bash -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/Bellman-Diffusion-Offline-RL/
conda init bash
conda activate offline_2

echo "hello"

python setup.py install



cd ~/Desktop/Bellman-Diffusion-Offline-RL/rebrac_som_no_q_scripts/medium-replay/hopper

sbatch -G 1 mr_hop_s0.sh
sleep 30s

sbatch -G 1 mr_hop_s1.sh
sleep 30s

sbatch -G 1 mr_hop_s2.sh
sleep 30s

sbatch -G 1 mr_hop_s3.sh
sleep 30s

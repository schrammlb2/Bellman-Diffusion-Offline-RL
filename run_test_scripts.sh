#!/bin/bash/ -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd /common/home/lbs105/Desktop/Bellman-Diffusion-Offline-RL
conda init bash
conda activate offline_2
echo "hello"

python setup.py install
sbatch -G 1 test_scripts/halfcheetah.sh
sbatch -G 1 test_scripts/walker2d.sh
sbatch -G 1 test_scripts/hopper.sh

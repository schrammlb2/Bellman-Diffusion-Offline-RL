#!/bin/bash -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/Bellman-Diffusion-Offline-RL
conda init bash
conda activate offline_2

echo "hello"

python setup.py install

#for method in "run_example/run_slow_test_kl_reg.py";
python run_example/run_diffusion_combo.py --task=walker2d-medium-v2 --rollout-length=1 --cql-weight=0.5 --seed=2 --batch-size=1000

#!/bin/bash -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd /common/home/lbs105/Desktop/Bellman-Diffusion-Offline-RL
conda init bash
conda activate offline_2

echo "hello"

python setup.py install

#for method in "run_example/run_slow_test_kl_reg.py";
python run_example/run_semi_mb_combo.py --task=hopper-medium-expert-v2 --rollout-length=5 --cql-weight=5.0 --seed=1 --batch-size=1000

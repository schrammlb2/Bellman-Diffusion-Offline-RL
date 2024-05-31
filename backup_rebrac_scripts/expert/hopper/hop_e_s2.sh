#!/bin/bash -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd ~/Desktop/Bellman-Diffusion-Offline-RL/
conda init bash
conda activate offline_2

echo "hello"

python setup.py install

#for method in "run_example/run_slow_test_kl_reg.py";
python pyrallis_scripts/run_rebrac_no_q.py --config=pyrallis_scripts/configs/offline/rebrac/hopper/medium_expert_v2.yaml --train_seed=2

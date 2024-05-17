#!/bin/bash/ -l
#SBATCH --output=logfile

epoch=1000
source ~/.bashrc
cd /common/home/lbs105/Desktop/Bellman-Diffusion-Offline-RL
conda init bash
conda activate offline_2
echo "hello"

python setup.py install
for i in {1..5};
do
	echo $i
	for settings in "--task=halfcheetah-medium-v2 --rollout-length=5 --cql-weight=0.5 --epoch=$epoch --seed=$i" \
				"--task=hopper-medium-v2 --rollout-length=5 --cql-weight=5.0 --epoch=$epoch --seed=$i" \
				"--task=walker2d-medium-v2 --rollout-length=1 --cql-weight=5.0 --epoch=$epoch --seed=$i" \
				"--task=halfcheetah-medium-replay-v2 --rollout-length=5 --cql-weight=0.5 --epoch=$epoch --seed=$i" \
				"--task=hopper-medium-replay-v2 --rollout-length=5 --cql-weight=0.5 --epoch=$epoch --seed=$i" \
				"--task=walker2d-medium-replay-v2 --rollout-length=1 --cql-weight=0.5 --epoch=$epoch --seed=$i" \
				"--task=halfcheetah-medium-expert-v2 --rollout-length=5 --cql-weight=5.0 --epoch=$epoch --seed=$i" \
				"--task=hopper-medium-expert-v2 --rollout-length=5 --cql-weight=5.0 --epoch=$epoch --seed=$i" \
				"--task=walker2d-medium-expert-v2 --rollout-length=1 --cql-weight=5.0 --epoch=$epoch --seed=$i" \
	do
	    sbatch -G 1 semi_mb_combo_script.sh $settings
	    # bash run_all_methods.sh "--task=$task --epoch=$epoch --seed=$i"
	done

done

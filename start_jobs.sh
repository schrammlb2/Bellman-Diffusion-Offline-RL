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
	for settings in "--task=halfcheetah-medium-v2 --cql-weight=0.5 --epoch=$epoch --seed=$i" \
				"--task=hopper-medium-v2 --cql-weight=5.0 --epoch=$epoch --seed=$i" \
				"--task=walker2d-medium-v2 --cql-weight=5.0 --epoch=$epoch --seed=$i" \
				"--task=halfcheetah-medium-replay-v2 --cql-weight=0.5 --epoch=$epoch --seed=$i" \
				"--task=hopper-medium-replay-v2 --cql-weight=0.5 --epoch=$epoch --seed=$i" \
				"--task=walker2d-medium-replay-v2 --cql-weight=0.5 --epoch=$epoch --seed=$i" \
				"--task=halfcheetah-medium-expert-v2 --cql-weight=0.5 --epoch=$epoch --seed=$i" \
				"--task=hopper-medium-expert-v2 --cql-weight=0.5 --epoch=$epoch --seed=$i" \
				"--task=walker2d-medium-expert-v2 --cql-weight=0.5 --epoch=$epoch --seed=$i" \
walker2d-medium-v2: rollout-length=1, cql-weight=5.0
halfcheetah-medium-replay-v2: rollout-length=5, cql-weight=0.5
hopper-medium-replay-v2: rollout-length=5, cql-weight=0.5
walker2d-medium-replay-v2: rollout-length=1, cql-weight=0.5
halfcheetah-medium-expert-v2: rollout-length=5, cql-weight=5.0
hopper-medium-expert-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-expert-v2: rollout-length=1, cql-weight=5.0
	"" "halfcheetah-medium-replay-v2"   "walker2d-medium-replay-v2" \
	    "hopper-medium-expert-v2" "halfcheetah-medium-expert-v2"   "walker2d-medium-expert-v2";
	#   "hopper-medium-v2" "halfcheetah-medium-v2"   "walker2d-medium-v2";
	do
	    sbatch -G 1 run_slow.sh $settings
	    # bash run_all_methods.sh "--task=$task --epoch=$epoch --seed=$i"
	done

done

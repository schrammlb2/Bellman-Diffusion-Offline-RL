epoch=400
# for task in "hopper-medium-v2";# "hopper-medium-expert-v2";
python setup.py install
for i in 1;
do
	echo $i
	# for task in "hopper-medium-expert-v2" "halfcheetah-expert-v2" "maze2d-umaze-dense-v1" "maze2d-umaze-v1" ;
	# for task in "door-human-v1" "maze2d-umaze-dense-v1" "antmaze-umaze-dense-v1"	\
	# 	"hopper-medium-v2"  "hopper-medium-expert-v2" "hopper-expert-v2" \
	# 	"halfcheetah-medium-v2" "halfcheetah-medium-expert-v2"  "halfcheetah-expert-v2" \
	# 	"walker2d-medium-v2" "walker2d-medium-expert-v2" "walker2d-expert-v2";
	# for task in "hopper-medium-v2" "hopper-medium-expert-v2" "hopper-expert-v2" "hopper-full-replay-v2" \
	# 	"walker2d-medium-v2" "walker2d-medium-expert-v2" "walker2d-expert-v2" "walker2d-full-replay-v2" \
	# 	"ant-medium-v2" "ant-medium-expert-v2" "ant-expert-v2" "ant-full-replay-v2";
	# for task in "hopper-medium-v2"  "walker2d-full-replay-v2" \
	# 	"ant-medium-v2" "ant-medium-expert-v2" "ant-expert-v2" "ant-full-replay-v2";
	for method in "run_example/run_test_kl_reg.py" "run_example/run_td3bc.py" "run_example/run_cql.py";
	do
		# for task in "hopper-medium-v2" "halfcheetah-medium-v2"   "walker2d-medium-v2" \
		# 	 "hopper-medium-replay-v2" "halfcheetah-medium-replay-v2"   "walker2d-medium-replay-v2" \
		# 	 "hopper-medium-expert-v2" "halfcheetah-medium-expert-v2"  "walker2d-medium-expert-v2";
		for task in "hopper-medium-replay-v2" "halfcheetah-medium-replay-v2"   "walker2d-medium-replay-v2" 
		do 
			echo $task
			settings="--task=$task --epoch=$epoch --seed=$i"
			python $method $settings
		done
	done
done

epoch=300
# for task in "hopper-medium-v2";# "hopper-medium-expert-v2";
python setup.py install
for i in {1..5};
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
	for task in "walker2d-full-replay-v2" "ant-full-replay-v2" "hopper-full-replay-v2" ;
	do 
		echo $task
		settings="--task=$task --epoch=$epoch --seed=$i" # --hidden-dims 512 512 --batch-size=10000"
		# python run_example/run_som_reg_only.py --task=$task --epoch=$epoch --seed=$i
		# python run_example/run_som_regularized_sac.py $settings
		# python run_example/run_tweaked_som_regularized_sac.py $settings
		python run_example/run_renyi_reg_sac.py $settings

		# python run_example/run_som_regularized_sac_original.py $settings
		# python run_example/run_tweaked_som_reg_only.py $settings
		# python run_example/run_rebrac.py $settings
		# python run_example/run_state_action_reg.py --task=$task --epoch=$epoch --seed=$i
		python run_example/run_cql.py $settings
		# # python run_example/run_edac.py --task=$task --epoch=$epoch --seed=$i
		# python run_example/run_iql.py $settings
	done
done

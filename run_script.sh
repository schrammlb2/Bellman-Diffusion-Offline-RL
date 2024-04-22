epoch=600
# for task in "hopper-medium-v2";# "hopper-medium-expert-v2";
for i in {1..5};
do
	echo $i
	# for task in "hopper-medium-expert-v2" "halfcheetah-expert-v2" "maze2d-umaze-dense-v1" "maze2d-umaze-v1" ;
	for task in "door-human-v1" "maze2d-umaze-dense-v1" "antmaze-umaze-dense-v1"	\
		"hopper-medium-v2"  "hopper-medium-expert-v2" "hopper-expert-v2" \
		"halfcheetah-medium-v2" "halfcheetah-medium-expert-v2"  "halfcheetah-expert-v2" \
		"walker2d-medium-v2" "walker2d-medium-expert-v2" "walker2d-expert-v2";
	do 
		echo $task
		settings="--task=$task --epoch=$epoch --seed=$i" # --hidden-dims 512 512 --batch-size=10000"
		echo $i
		# python run_example/run_som_reg_only.py $settings
		python run_example/run_som_regularized_sac.py $settings
		# python run_example/run_som_regularized_sac.py --task=$task --epoch=250 --seed=$i
		# python run_example/run_cql.py --task=$task --epoch=250 --seed=$i
		# python run_example/run_td3bc.py --task=$task --epoch=250 --seed=$i
		python run_example/run_iql.py $settings
	done
done

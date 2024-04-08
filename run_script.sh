for task in "hopper-medium-v2";# "hopper-medium-expert-v2";
do 
	echo $task
	for i in {1..5};
	do
		echo $i
		python run_example/run_som_reg_only.py --task=$task --epoch=250 --seed=$i
		python run_example/run_som_regularized_sac.py --task=$task --epoch=250 --seed=$i
		python run_example/run_cql.py --task=$task --epoch=250 --seed=$i
		python run_example/run_td3bc.py --task=$task --epoch=250 --seed=$i
		python run_example/run_iql.py --task=$task --epoch=250 --seed=$i
	done
done

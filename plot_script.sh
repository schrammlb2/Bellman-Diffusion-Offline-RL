# rootdir=oril_log
# rootdir=SOMBC_log
# rootdir=sequential_log_merged
# rootdir=seq_bc_log
rootdir=sequential_log_new
reward=false

for task in "hopper-medium-v2" "halfcheetah-medium-v2"  "walker2d-medium-v2"\
 	"hopper-medium-expert-v2" "halfcheetah-medium-expert-v2"  "walker2d-medium-expert-v2"\
 	"hopper-medium-replay-v2" "halfcheetah-medium-replay-v2"  "walker2d-medium-replay-v2"\
 	"hopper-expert-v2" "halfcheetah-expert-v2"  "walker2d-expert-v2";
do 
	#python run_example/plotter.py --task=$task --algos slow_test_kl_reg test_kl_reg td3bc cql --root-dir=ilab_log
	echo $task
	# python run_example/plotter.py --task=$task --algos rebracno_q rebrac_som_no_q --root-dir=bc_log_5
	# python run_example/plotter.py --task=$task --algos rebrac rebrac_som --root-dir=bc_log_5
	# python run_example/plotter.py --task=$task --algos rebrac rebrac_som_no_q --root-dir=bc_log_5
	# python run_example/plotter.py --task=$task --algos rebrac rebrac_som rebracno_q rebrac_som_no_q --root-dir=bc_log_5
	# python run_example/plotter.py --task=$task --algos rebrac rebrac_som rebracno_q rebrac_som_no_q smodice --root-dir=SMODICE_log --title=$task
	# python run_example/plotter.py --task=$task --algos rebracno_q rebrac_som_no_q smodice --root-dir=SMODICE_log --title=$task
	# python run_example/plotter.py --task=$task --algos rebrac rebrac_som_no_q --root-dir=SMODICE_log --title=$task

	#Offline RL 
	if [ "$reward" = true ] ; then
		python run_example/plotter.py --task=$task --algos rebrac rebrac_som rebrac_sequential_som --root-dir=$rootdir --title=$task
		# python run_example/plotter.py --task=$task --algos rebrac_som --root-dir=$rootdir --title=$task
	fi
	#Imitation learning 
	# python run_example/plotter.py --task=$task --algos rebracno_q rebrac_som_no_q smodice smodice_layernorm oril --root-dir=$rootdir --title=$task

	if [ "$reward" = false ] ; then
		# python run_example/plotter.py --task=$task --algos rebracno_q rebrac_som_no_q smodice oril --root-dir=$rootdir --title=$task
		# python run_example/plotter.py --task=$task --algos rebrac rebrac_som rebracno_q rebrac_seq_som rebrac_seq_som_no_q smodice_layernorm --root-dir=$rootdir --title=$task
		python run_example/plotter.py --task=$task --algos rebracno_q smodice_layernorm rebrac_sequential_som_no_q --root-dir=$rootdir --title=$task
		# python run_example/plotter.py --task=$task --algos rebrac_seq_som_no_q --root-dir=$rootdir --title=$task
	fi
done

if [ "$reward" = false ] ; then
	# python run_example/plotter.py --task="walker2d-medium-expert-v2" --algos rebrac rebrac_som rebracno_q rebrac_seq_som_no_q --root-dir=$rootdir --title=$task
	python run_example/plotter.py --task="walker2d-medium-expert-v2" --algos rebracno_q rebrac_sequential_som_no_q --root-dir=$rootdir --title=$task
fi
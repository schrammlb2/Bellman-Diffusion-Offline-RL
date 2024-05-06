for task in "hopper-medium-v2" "halfcheetah-medium-v2"   "walker2d-medium-v2" \
	 "hopper-medium-replay-v2" "halfcheetah-medium-replay-v2"   "walker2d-medium-replay-v2" \
	 "hopper-medium-expert-v2" "halfcheetah-medium-expert-v2"  "walker2d-medium-expert-v2";
do 
	python run_example/plotter.py --task=$task
done
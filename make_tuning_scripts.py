import shutil
import os

def env_map(env):
	if env == "halfcheetah":
		return "hc"
	elif env == "hopper":
		return "hop"
	elif env == "walker2d":
		return "walk"
	import ipdb
	ipdb.set_trace()

def data_map(env):
	if env == "medium":
		return "m"
	elif env == "medium-replay":
		return "mr"
	elif env == "medium-expert":
		return "me"
	elif env == "expert":
		return "e"
	import ipdb
	ipdb.set_trace()

pwd=os.getcwd()

template = "sbatch_template.sh"
alg_list = ["sequential_rebrac_som_no_q"]
for alg in alg_list:
	base_dir = f"{pwd}/tuning/{alg}_scripts"        
	# script_dir = "/common/home/lbs105/Desktop/Bellman-Diffusion-Offline-RL/pyrallis_scripts"
	script_dir = f"{pwd}/pyrallis_scripts"
	method = f"{script_dir}/seq_som_grid_search.py --config={script_dir}/configs/offline/rebrac/"
	os.mkdir(base_dir)
	datasets = ["medium", "medium-replay", "medium-expert", "expert"]
	envs = [ "hopper", "halfcheetah", "walker2d"]
	for dataset in datasets:
		new_dataset = base_dir + "/" + dataset
		os.mkdir(new_dataset)
		run_dataset_loc = base_dir +  f"/run_{dataset}_scripts.sh"
		with open(run_dataset_loc, "a") as file:
			file.write(f"cd {base_dir}/{dataset}/")
			for env in envs:
				filename = f"run_{env}_{data_map(dataset)}.sh"
				run_env_loc = new_dataset + "/" + filename
				file.write("\n")
				file.write(f"bash {filename}\n")
				file.write(f"sleep 30s\n")
				file.write("\n")

		for env in envs:
			new_env = new_dataset + "/" + env
			os.mkdir(new_env)
			filename = f"run_{env}_{data_map(dataset)}.sh"
			run_env_loc = new_dataset + "/" + filename
			shutil.copyfile(template, run_env_loc)
			with open(run_env_loc, "a") as file:
				file.write("\n")
				file.write(f"cd {base_dir}/{dataset}/{env}")
				file.write(f"\n")

			for seed in range(1):
				filename = f"{data_map(dataset)}_{env_map(env)}_s{seed}.sh"
				loc = new_env + "/" + filename
				shutil.copyfile(template, loc)
				with open(loc, "a") as file:
					file.write("\n\n")
					# file.write(method + f"{env}/{dataset}_v2.yaml --train_seed={seed}")
					file.write(f"python {method}{env}/{dataset}_v2.yaml --train_seed={seed}")

				with open(run_env_loc, "a") as file:
					file.write("\n")
					file.write(f"sbatch -G 1 {filename}\n")
					file.write(f"sleep 30s\n")

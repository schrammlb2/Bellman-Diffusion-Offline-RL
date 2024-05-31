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
	import ipdb
	ipdb.set_trace()

template = "sbatch_template.sh"
for alg in ["rebrac", "rebrac_no_q", "rebrac_som", "rebrac_som_no_q"]:
	base_dir = f"{alg}_scripts"
	method = f"python pyrallis_scripts/run_{alg}.py --config=pyrallis_scripts/configs/offline/rebrac/"
	os.mkdir(base_dir)
	datasets = ["medium", "medium-replay", "medium-expert"]
	envs = [ "hopper", "halfcheetah", "walker2d"]
	for dataset in datasets:
		new_dataset = base_dir + "/" + dataset
		os.mkdir(new_dataset)
		run_dataset_loc = base_dir +  f"/run_{dataset}_scripts.sh"
		with open(run_dataset_loc, "a") as file:
			file.write(f"cd ~/Desktop/Bellman-Diffusion-Offline-RL/{base_dir}/{dataset}/")
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
				file.write(f"cd ~/Desktop/Bellman-Diffusion-Offline-RL/{base_dir}/{dataset}/{env}")
				file.write(f"\n")

			for seed in range(4):
				filename = f"{data_map(dataset)}_{env_map(env)}_s{seed}.sh"
				loc = new_env + "/" + filename
				shutil.copyfile(template, loc)
				with open(loc, "a") as file:
					file.write("\n\n")
					file.write(method + f"{env}/{dataset}_v2.yaml --train-seed={seed}")

				with open(run_env_loc, "a") as file:
					file.write("\n")
					file.write(f"sbatch -G 1 {filename}\n")
					file.write(f"sleep 30s\n")

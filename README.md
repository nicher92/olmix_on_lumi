

# Purpose 
This is a repo using code from Olmix (https://arxiv.org/abs/2602.12237, https://github.com/allenai/olmix)  
The pipeline looks like this:  
First, input paths to our nemotron .bin files, calculate tokens, and generate suggested training mixtures using olmix  
Second, use the training mixtures to generate slurm scripts for small scale training runs  
Thirdly, evaluate the different mixtures using BPB  
Finally train regression models on the mixtures and get a final suggested mix  

# Steps for computing optimal mixture  
Add all datasets that we are interested in, in the config.yaml file  
training data used as input can be found here: /scratch/project_465002530/preprocessed/oellm-v1-256k/catalogue/  

# Command to run to generate mixtures directly  
./generate_mixes.sh -- this will read the config.yaml and use olmix to generate suggested mixes, it will also create the launch_all_swarms.sh script  
the suggested mixes will be put into the /mixes directory  
Per the Olmix paper we generate unconstrained swarms to explore the mixture space, and later constrain the specific mixture to a realistic mix that takes data constraints into account

# Add keys before running swarms
Add variables for wandb and huggingface token in a file here ~/.hpc_secrets - these variables are expected in /scripts/train-0.05B.sh  
export WANDB_API_KEY=""  
export HF_TOKEN=""  

# Command to start run
./launch_all_swarms.sh - starts an array job of all mixes

# Convering final checkpoints to HF models
./scripts/convert_olmix_models_to_hf.sh - converts the final checkpoints (ie the ones ending with iter_0022889) to hf checkpoints

# Running evals with BPB
uv tool install -p 3.12 --force git+https://github.com/nicher92/oellm-eval.git@bpb-metrics
export HF_HOME=/scratch/project_465002530/cache/huggingface
oellm-eval schedule --models "<path to model>" --task_groups "bpb-core"

# TODO  
test the fitting portion of a run based on metrics.csv, ratios.csv files - or run elsewhere

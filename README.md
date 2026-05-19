

# Purpose 
This is a repo using code from Olmix (https://arxiv.org/abs/2602.12237, https://github.com/allenai/olmix)  
The pipeline looks like this:  
First, input paths to our nemotron .bin files, calculate tokens, and generate suggested training mixtures using olmix  
Second, use the training mixtures to generate slurm scripts for small scale training runs  
Thirdly, evaluate the different mixtures using BPB  
Finally train regression models on the mixtures and get a final suggested mix  

# Steps for computing optimal mixture  
add all datasets that we are interested in, in the config.yaml file  
training data used as input can be found here: /scratch/project_465002530/preprocessed/oellm-v1-256k/catalogue/  

# Command to run to generate mixtures directly  
run.sh -- this will read the config.yaml and use olmix to generate suggested mixes  
the suggested mixes will be put into the /mixes directory  


# TODO  
see if we can optimize speed of small training runs, might be reasonable to retokenize a small part of the data using a smaller tokenizer to make the models smaller  
add evaluation using BPB, base it off previous script that uses the oellm-cli generated bash script  

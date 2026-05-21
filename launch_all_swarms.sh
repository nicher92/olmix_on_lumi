#!/bin/bash

# Submits an array of 70 jobs. 
# This tells Slurm to run the script 70 times, setting $SLURM_ARRAY_TASK_ID 
# to a value from 0 to 69 for each respective run.
sbatch --array=0-2 scripts/train-0.05B-ne_test.sh

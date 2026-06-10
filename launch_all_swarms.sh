#!/bin/bash

sbatch --array=0-77%10 scripts/train-0.05B.sh

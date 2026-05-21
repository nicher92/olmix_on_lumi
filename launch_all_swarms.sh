#!/bin/bash

sbatch --array=0-2 scripts/train-0.05B.sh

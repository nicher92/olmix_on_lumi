#!/bin/bash

sbatch --array=0-17 scripts/train-0.05B.sh

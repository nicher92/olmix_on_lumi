#!/bin/bash

sbatch --array=0-77 scripts/train-0.05B.sh

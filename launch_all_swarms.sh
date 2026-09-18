#!/bin/bash

sbatch --array=0-14 scripts/train-0.05B.sh

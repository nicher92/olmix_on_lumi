#!/bin/bash

sbatch --array=0-8 --export=ALL,MIX_PREFIX=stage2_mix_20261006_1519 scripts/train-0.05B.sh

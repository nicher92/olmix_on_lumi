#!/bin/bash

sbatch --array=0-74 scripts/train-0.05B-ne_test.sh

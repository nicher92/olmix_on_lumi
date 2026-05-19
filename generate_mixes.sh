#!/bin/bash
set -e

# Run the unified pipeline
/pfs/lustrep4/scratch/project_462000963/users/niclhert/olmix_tool/.venv/bin/python ./scripts/generate_variants.py

echo "Ready to launch!"

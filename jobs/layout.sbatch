#!/bin/bash
# Top-down maps of the annotated rooms for every recording, and a test on the recorded paths (gray/layout.py). No GPU work.
#
#   sbatch -p apudev -t 15 jobs/layout.sbatch --limit 6 --out /ptmp/enazaren/gray/layout-test    # a test
#   sbatch jobs/layout.sbatch
#
# Log: /ptmp/enazaren/gray/runs/gray-layout-<job id>.log     Output: /ptmp/enazaren/gray/layout
#
#SBATCH -J gray-layout
#SBATCH -t 02:00:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
/u/enazaren/envs/torch/bin/python -u -m gray.layout "$@"

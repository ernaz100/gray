#!/bin/bash
# Which description of an image varies least between two views of the same place (gray/spaces.py). No GPU work.
#
#   sbatch -p apudev -t 15 jobs/spaces.sbatch
#
# Log: /ptmp/enazaren/gray/runs/gray-spaces-<job id>.log     Output: /ptmp/enazaren/gray/runs/spaces
#
#SBATCH -J gray-spaces
#SBATCH -t 00:15:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
/u/enazaren/envs/torch/bin/python -u -m gray.spaces "$@"

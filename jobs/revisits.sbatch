#!/bin/bash
# For every frame a later frame of the same recording from nearly the same pose (gray/revisits.py). No GPU work.
#
#   sbatch -p apudev -t 15 jobs/revisits.sbatch
#
# Log: /ptmp/enazaren/gray/runs/gray-revisits-<job id>.log     Output: /ptmp/enazaren/gray/revisits
#
#SBATCH -J gray-revisits
#SBATCH -t 00:15:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
/u/enazaren/envs/torch/bin/python -u -m gray.revisits "$@"

#!/bin/bash
# Free and blocked space per location from all recorded paths and measured points (gray/ground.py). No GPU work; finished locations are skipped.
#
#   sbatch -p apudev -t 15 jobs/ground.sbatch --part 0 --parts 3
#
# Log: /ptmp/enazaren/gray/runs/gray-ground-<job id>.log     Output: /ptmp/enazaren/gray/ground
#
#SBATCH -J gray-ground
#SBATCH -t 02:00:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
/u/enazaren/envs/torch/bin/python -u -m gray.ground "$@"

#!/bin/bash
# The room maps against the points measured by Aria's SLAM (gray/cloud.py). No GPU work.
#
#   sbatch -p apudev -t 15 jobs/cloud.sbatch maps
#   sbatch -p apudev -t 15 jobs/cloud.sbatch depth --names depth-base depth-large
#
# Log: /ptmp/enazaren/gray/runs/gray-cloud-<job id>.log     Output: /ptmp/enazaren/gray/cloud
#
#SBATCH -J gray-cloud
#SBATCH -t 00:15:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
/u/enazaren/envs/torch/bin/python -u -m gray.cloud "$@"

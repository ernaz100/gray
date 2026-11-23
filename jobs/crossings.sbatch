#!/bin/bash
# What kind of wall the recorded paths cross in the room maps (gray/crossings.py).
#SBATCH -J gray-crossings
#SBATCH -t 00:15:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
/u/enazaren/envs/torch/bin/python -u -m gray.crossings "$@"

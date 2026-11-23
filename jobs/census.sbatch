#!/bin/bash
# How much walking is in the cached windows and what the text says about it (gray/census.py). Reads the cache only, no GPU work.
#
#   sbatch -p apudev -t 15 jobs/census.sbatch
#
# Log: /ptmp/enazaren/gray/runs/gray-census-<job id>.log     Output: /ptmp/enazaren/gray/runs/census
#
#SBATCH -J gray-census
#SBATCH -t 00:15:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
/u/enazaren/envs/torch/bin/python -u -m gray.census --out /ptmp/enazaren/gray/runs/census "$@"

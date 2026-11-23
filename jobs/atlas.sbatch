#!/bin/bash
# Pictures of every location with all recorded head paths (gray/atlas.py). No GPU work.
#
#   sbatch -p apudev -t 15 jobs/atlas.sbatch
#
# Log: /ptmp/enazaren/gray/runs/gray-atlas-<job id>.log     Output: /ptmp/enazaren/gray/atlas
#
#SBATCH -J gray-atlas
#SBATCH -t 00:15:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
/u/enazaren/envs/torch/bin/python -u -m gray.atlas "$@"

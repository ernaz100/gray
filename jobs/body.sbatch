#!/bin/bash
# The body motion of every recording as one track (gray/body.py). Skips what is done: submit again until all are.
#
#   sbatch -p apudev -t 15 jobs/body.sbatch
#
# Log: /ptmp/enazaren/gray/runs/gray-body-<job id>.log     Output: /ptmp/enazaren/gray/body
#
#SBATCH -J gray-body
#SBATCH -t 00:15:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
/u/enazaren/envs/torch/bin/python -u -m gray.body "$@"

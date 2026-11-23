#!/bin/bash
# Does the joint model know when its feature prediction is uncertain? (gray/spread.py). Trains nothing.
#
#   sbatch jobs/spread.sbatch joint-v1
#
# Log: /ptmp/enazaren/gray/runs/gray-spread-<job id>.log     Output: <run>/spread.json
#
#SBATCH -J gray-spread
#SBATCH -t 00:30:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
NAME=joint-v1
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then NAME="$1"; shift; fi
/u/enazaren/envs/torch/bin/python -u -m gray.spread --run "/ptmp/enazaren/gray/runs/$NAME" "$@"

#!/bin/bash
# Experiment 2, second look: several head paths per window and the effect of the text (gray/paths.py). Trains nothing.
#
#   sbatch -p apudev -t 15 jobs/paths.sbatch                 # run joint-v1
#   sbatch -p apudev -t 15 jobs/paths.sbatch <run name> [paths.py arguments]
#
# Log: /ptmp/enazaren/gray/runs/gray-paths-<job id>.log     Output: <run>/paths.json
#
#SBATCH -J gray-paths
#SBATCH -t 00:15:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
NAME=joint-v1
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then NAME="$1"; shift; fi
/u/enazaren/envs/torch/bin/python -u -m gray.paths --run "/ptmp/enazaren/gray/runs/$NAME" "$@"

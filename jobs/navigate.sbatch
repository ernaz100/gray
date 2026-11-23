#!/bin/bash
# Score generated head paths against the room maps and the text (gray/navigate.py). Trains nothing.
#
#   sbatch jobs/navigate.sbatch head-v1 [navigate.py arguments]
#
# Log: /ptmp/enazaren/gray/runs/gray-navigate-<job id>.log     Output: <run>/navigate.json
#
#SBATCH -J gray-navigate
#SBATCH -t 01:30:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
NAME=head-v1
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then NAME="$1"; shift; fi
/u/enazaren/envs/torch/bin/python -u -m gray.navigate --run "/ptmp/enazaren/gray/runs/$NAME" "$@"

#!/bin/bash
# Score a trained head-path model again, for example with text guidance (gray/score.py). Trains nothing.
#
#   sbatch jobs/score.sbatch head-v3 --guidance 2
#
# Log: /ptmp/enazaren/gray/runs/gray-score-<job id>.log     Output: <run>/score.json or score-guidance-<value>.json
#
#SBATCH -J gray-score
#SBATCH -t 00:30:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
NAME=head-v3
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then NAME="$1"; shift; fi
/u/enazaren/envs/torch/bin/python -u -m gray.score --run "/ptmp/enazaren/gray/runs/$NAME" "$@"

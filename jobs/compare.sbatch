#!/bin/bash
# Train the comparator on moments (gray/compare.py): the image feature along the path, with its uncertainty.
#
#   jobs/chain_train.sh 4 jobs/compare.sbatch compare-v1 [compare.py arguments]
#
# Log: /ptmp/enazaren/gray/runs/gray-compare-<job id>.log     Output: /ptmp/enazaren/gray/runs/<name>
#
#SBATCH -J gray-compare
#SBATCH -t 01:00:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
NAME=compare-v1
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then NAME="$1"; shift; fi
/u/enazaren/envs/torch/bin/python -u -m gray.compare --amp --resume --save-dir "/ptmp/enazaren/gray/runs/$NAME" "$@"

#!/bin/bash
# Pictures of the paths a head-path model draws (gray/show.py). Trains nothing.
#
#   sbatch -p apudev -t 15 jobs/show.sbatch head-v5
#
# Log: /ptmp/enazaren/gray/runs/gray-show-<job id>.log     Output: <run>/show/paths-<n>.png
#
#SBATCH -J gray-show
#SBATCH -t 00:30:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
NAME=head-v5
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then NAME="$1"; shift; fi
/u/enazaren/envs/torch/bin/python -u -m gray.show --run "/ptmp/enazaren/gray/runs/$NAME" --out "/ptmp/enazaren/gray/runs/$NAME/show" "$@"

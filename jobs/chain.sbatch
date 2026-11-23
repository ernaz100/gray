#!/bin/bash
# Checks of the body model: sampling settings, the chain with the head-path model, rolling forward (gray/chain.py).
#
#   sbatch -p apudev -t 15 jobs/chain.sbatch motion-v1
#
# Log: /ptmp/enazaren/gray/runs/gray-chain-<job id>.log     Output: <run>/chain.json
#
#SBATCH -J gray-chain
#SBATCH -t 00:30:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
NAME=motion-v1
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then NAME="$1"; shift; fi
/u/enazaren/envs/torch/bin/python -u -m gray.chain --body-run "/ptmp/enazaren/gray/runs/$NAME" --head-run /ptmp/enazaren/gray/runs/head-v5 "$@"

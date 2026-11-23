#!/bin/bash
# More homes from Ropedia Xperience 2 for reading a room (gray/ropedia.py). The download (fetch) runs on the login node.
#
#   sbatch -p apudev -t 15 jobs/ropedia.sbatch set
#   sbatch -p apudev -t 15 jobs/probe.sbatch features --set /ptmp/enazaren/gray/ropeset --name dino-s-7
#   sbatch -p apudev -t 15 jobs/ropedia.sbatch fit
#
# Log: /ptmp/enazaren/gray/runs/gray-ropedia-<job id>.log     Output: /ptmp/enazaren/gray/ropeset
#
#
#SBATCH -J gray-ropedia
#SBATCH -t 01:00:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0 HF_HUB_OFFLINE=1
/u/enazaren/envs/torch/bin/python -u -m gray.ropedia "$@"

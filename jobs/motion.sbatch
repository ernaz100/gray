#!/bin/bash
# Train the body model on moments (gray/motion.py): the head path of the next 5 s -> the body along it.
#
#   sbatch jobs/motion.sbatch motion-v1 [motion.py arguments]
#
# Log: /ptmp/enazaren/gray/runs/gray-motion-<job id>.log     Output: /ptmp/enazaren/gray/runs/<name>
#
#SBATCH -J gray-motion
#SBATCH -t 08:00:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
NAME=motion-v1
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then NAME="$1"; shift; fi
/u/enazaren/envs/torch/bin/python -u -m gray.motion --amp --resume --save-dir "/ptmp/enazaren/gray/runs/$NAME" "$@"

#!/bin/bash
# Does the comparator notice a changed world? (gray/alarm.py). Trains nothing.
#
#   sbatch -p apudev -t 15 jobs/alarm.sbatch compare-v2
#
# Log: /ptmp/enazaren/gray/runs/gray-alarm-<job id>.log     Output: <run>/alarm.json
#
#SBATCH -J gray-alarm
#SBATCH -t 00:30:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
NAME=compare-v2
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then NAME="$1"; shift; fi
/u/enazaren/envs/torch/bin/python -u -m gray.alarm --run "/ptmp/enazaren/gray/runs/$NAME" "$@"

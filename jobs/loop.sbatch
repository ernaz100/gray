#!/bin/bash
# Experiment 3: the replanning loop along recordings, triggers compared (gray/loop.py). Trains nothing.
#
#   sbatch jobs/loop.sbatch                          # run joint-v1, two recordings of each held-out location
#   sbatch -p apudev -t 15 jobs/loop.sbatch joint-v1 --per-location 1 20230703_s0_tommy_ayala_act0_c07rbi
#   sbatch jobs/loop.sbatch <run name> [loop.py arguments]
#
# Log: /ptmp/enazaren/gray/runs/gray-loop-<job id>.log     Output: <run>/loop/
#
#SBATCH -J gray-loop
#SBATCH -t 01:30:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
NAME=joint-v1
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then NAME="$1"; shift; fi
/u/enazaren/envs/torch/bin/python -u -m gray.loop --run "/ptmp/enazaren/gray/runs/$NAME" "$@"

#!/bin/bash
# Experiment 1 on Viper: train the feature predictor with and without the head path (see gray/features.py).
#
#   sbatch jobs/features.sbatch                                   # both runs, 20,000 steps each
#   sbatch -p apudev -t 15 jobs/features.sbatch smoke --steps 300 --eval-every 150 --val-windows 512
#
# The first argument, if it is not an option, names the run (default features-v1); the rest goes to features.py.
#
# Log: /ptmp/enazaren/gray/runs/gray-features-<job id>.log (the folder must exist before submitting)
# Results: /ptmp/enazaren/gray/runs/<run name>/{full,none}/results.json
#
#SBATCH -J gray-features
#SBATCH -t 03:00:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0   # ROCm: rocBLAS, as in STEER's env.sh
PY=/u/enazaren/envs/torch/bin/python
RUNS=/ptmp/enazaren/gray/runs
NAME=features-v1
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then NAME="$1"; shift; fi
for pose in full none; do
    "$PY" -u -m gray.features --pose "$pose" --save-dir "$RUNS/$NAME/$pose" "$@"
done

#!/bin/bash
# Experiment 2 on Viper: train the joint model of head path, image features and body motion (see gray/joint.py).
# Resumable: submit the same command again to continue from last.pt.
#
#   sbatch jobs/joint.sbatch                                      # run joint-v1, 100,000 steps
#   sbatch -p apudev -t 15 jobs/joint.sbatch smoke --steps 300 --eval-every 150 --val-windows 256
#
# The first argument, if it is not an option, names the run (default joint-v1); the rest goes to joint.py.
#
# Log: /ptmp/enazaren/gray/runs/gray-joint-<job id>.log
# Results: /ptmp/enazaren/gray/runs/<run name>/{config.json,log.jsonl,last.pt}
#
#SBATCH -J gray-joint
#SBATCH -t 12:00:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0   # ROCm: rocBLAS, as in STEER's env.sh
PY=/u/enazaren/envs/torch/bin/python
RUNS=/ptmp/enazaren/gray/runs
NAME=joint-v1
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then NAME="$1"; shift; fi
"$PY" -u -m gray.joint --amp --resume --save-dir "$RUNS/$NAME" "$@"

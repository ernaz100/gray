#!/bin/bash
# Train the head-path model on moments (gray/head.py), then score it against the room maps and the text (gray/score.py).
#
#   sbatch jobs/head.sbatch head-v3 [head.py arguments]
#   sbatch -p apudev -t 15 jobs/head.sbatch head-test --steps 300 --eval-every 150      # a test
#
# Log: /ptmp/enazaren/gray/runs/gray-head-<job id>.log     Output: /ptmp/enazaren/gray/runs/<name>
#
#SBATCH -J gray-head
#SBATCH -t 08:00:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0   # ROCm: rocBLAS, as in STEER's env.sh
PY=/u/enazaren/envs/torch/bin/python
NAME=head-v3
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then NAME="$1"; shift; fi
"$PY" -u -m gray.head --amp --resume --save-dir "/ptmp/enazaren/gray/runs/$NAME" "$@"
[ -n "$NO_SCORE" ] || "$PY" -u -m gray.score --run "/ptmp/enazaren/gray/runs/$NAME"

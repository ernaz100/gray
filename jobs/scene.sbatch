#!/bin/bash
# A few prompts in Loc_08 through the chain, without the viewer (gray/scene_worker.py --demo).
#
#   sbatch -p apudev -t 15 jobs/scene.sbatch
#
# Log: /ptmp/enazaren/gray/runs/gray-scene-<job id>.log     Output: /ptmp/enazaren/gray/runs/scene-demo
#
#SBATCH -J gray-scene
#SBATCH -t 00:30:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
/u/enazaren/envs/torch/bin/python -u -m gray.scene_worker --demo 6 --out /ptmp/enazaren/gray/runs/scene-demo "$@"

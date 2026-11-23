#!/bin/bash
# Pinhole image features for every recording of the cache (gray/pinhole.py). Finished recordings are skipped on a rerun.
#
#   sbatch -p apudev -t 15 -a 0 jobs/pinhole.sbatch --limit 1500 <sequence id> ...   # a test
#   sbatch jobs/pinhole.sbatch                                                       # everything, 16 tasks
#
# Log: /ptmp/enazaren/gray/runs/gray-pinhole-<job id>_<task>.log     Output: /ptmp/enazaren/gray/pinhole/<id>/dino.npy
#
#SBATCH -J gray-pinhole
#SBATCH -t 12:00:00
#SBATCH -a 0-15
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%A_%a.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
module load ffmpeg/7.1
/u/enazaren/envs/torch/bin/python -u -m gray.pinhole --part "$SLURM_ARRAY_TASK_ID" --parts 16 "$@"

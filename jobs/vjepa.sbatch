#!/bin/bash
# V-JEPA 2 clip features of the pinhole view for every frame that can be now (gray/vjepa.py). Finished recordings are skipped on a rerun.
#
#   sbatch -p apudev -t 15 -a 0 jobs/vjepa.sbatch --limit 40 <sequence id>    # a test
#   sbatch jobs/vjepa.sbatch                                                 # everything, 16 tasks of 1 hour (rerun until done)
#
# Log: /ptmp/enazaren/gray/runs/gray-vjepa-<job id>_<task>.log     Output: /ptmp/enazaren/gray/vjepa/<id>/
#
#SBATCH -J gray-vjepa
#SBATCH -t 1:00:00
#SBATCH -a 0-15
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%A_%a.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0 HF_HUB_OFFLINE=1
type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
module load ffmpeg/7.1
/u/enazaren/envs/torch/bin/python -u -m gray.vjepa --weights /ptmp/enazaren/gray/weights/vjepa2-vitl-fpc64-256 --part "$SLURM_ARRAY_TASK_ID" --parts 16 "$@"

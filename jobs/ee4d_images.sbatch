#!/bin/bash
# DINOv3 class tokens and 7 x 7 grids for UniEgoMotion's data from the Ego-Exo4D videos (gray/ee4d_images.py). Finished takes are skipped on a rerun.
#
#   sbatch -p apudev -t 15 -a 0 jobs/ee4d_images.sbatch --limit 900 <take> ...   # a test
#   sbatch jobs/ee4d_images.sbatch                                               # everything, 16 tasks
#
# Log: /ptmp/enazaren/gray/runs/gray-ee4d-images-<job id>_<task>.log     Output: /ptmp/enazaren/gray/ee4d/gray_<split>_dino3.npy, gray_<split>_grid.npy
#
#SBATCH -J gray-ee4d-images
#SBATCH -t 06:00:00
#SBATCH -a 0-15
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%A_%a.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
module load ffmpeg/7.1
flock /ptmp/enazaren/gray/ee4d/images.lock /u/enazaren/envs/torch/bin/python -m gray.ee4d_images prepare --ee4d /ptmp/enazaren/gray/ee4d   # the first task makes the files
/u/enazaren/envs/torch/bin/python -u -m gray.ee4d_images encode --ee4d /ptmp/enazaren/gray/ee4d --part "$SLURM_ARRAY_TASK_ID" --parts 16 "$@"

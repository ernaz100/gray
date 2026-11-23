#!/bin/bash
# Whole-recording head tracks (gray/tracks.py), then the list of moments (gray/moments.py). No GPU work.
#
#   sbatch -p apudev -t 15 jobs/moments.sbatch
#
# Log: /ptmp/enazaren/gray/runs/gray-moments-<job id>.log     Output: /ptmp/enazaren/gray/tracks, /ptmp/enazaren/gray/moments
#
#SBATCH -J gray-moments
#SBATCH -t 00:15:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
/u/enazaren/envs/torch/bin/python -u -m gray.tracks
/u/enazaren/envs/torch/bin/python -u -m gray.moments

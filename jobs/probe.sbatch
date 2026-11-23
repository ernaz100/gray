#!/bin/bash
# The test bed for image features (gray/probe.py): build the frame set, extract a candidate's features, or fit the probes.
#
#   sbatch -p apudev -t 15 jobs/probe.sbatch set --part 0 --parts 2
#   sbatch -p apudev -t 15 jobs/probe.sbatch features --name dino-s-14
#   sbatch -p apudev -t 15 jobs/probe.sbatch fit --names token dino-s-7 dino-s-14
#
# Log: /ptmp/enazaren/gray/runs/gray-probe-<job id>.log     Output: /ptmp/enazaren/gray/probeset
#
#SBATCH -J gray-probe
#SBATCH -t 01:00:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0 HF_HUB_OFFLINE=1
/u/enazaren/envs/torch/bin/python -u -m gray.probe "$@"

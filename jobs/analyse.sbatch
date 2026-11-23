#!/bin/bash
# Experiment 1, analysis of a trained run: scores in known and new locations (gray/evaluate.py) and the
# mismatch along a few recordings with pictures of its peaks (gray/trace.py). Trains nothing.
#
#   sbatch -p apudev -t 15 jobs/analyse.sbatch              # run features-v1
#   sbatch -p apudev -t 15 jobs/analyse.sbatch <run name>
#
# Log: /ptmp/enazaren/gray/runs/gray-analyse-<job id>.log
# Output: <run>/evaluate.json and <run>/trace/
#
#SBATCH -J gray-analyse
#SBATCH -t 00:15:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
export TORCH_BLAS_PREFER_HIPBLASLT=0
PY=/u/enazaren/envs/torch/bin/python
RUN="/ptmp/enazaren/gray/runs/${1:-features-v1}"
"$PY" -u -m gray.evaluate --run "$RUN"
# Three recordings in held-out locations, then one the model never saw in Loc_08 (a training location).
"$PY" -u -m gray.trace --run "$RUN" --out "$RUN/trace" \
    20230703_s0_tommy_ayala_act0_c07rbi \
    20230828_s0_kaylee_johnson_act2_2ew9cq \
    20231110_s1_jerry_shaffer_act4_gy8vgr \
    20230621_s1_kevin_davis_act3_bw8qui

#!/bin/bash
# Pictures of fisheye frames next to their pinhole cut-outs (gray/rectify.py). A few frames, no GPU work.
#
#   sbatch -p apudev -t 15 jobs/rectify.sbatch [sequence ids and rectify.py arguments]
#
# Log: /ptmp/enazaren/gray/runs/gray-rectify-<job id>.log     Output: /ptmp/enazaren/gray/runs/rectify
#
#SBATCH -J gray-rectify
#SBATCH -t 00:15:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
if [ $# -eq 0 ]; then set -- 20230621_s1_kevin_davis_act3_bw8qui 20230703_s0_tommy_ayala_act0_c07rbi; fi
/u/enazaren/envs/torch/bin/python -u -m gray.rectify --out /ptmp/enazaren/gray/runs/rectify "$@"

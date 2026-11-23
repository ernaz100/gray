#!/bin/bash
# Pictures comparing the Aria fisheye with robot camera views (gray/views.py). A few frames, no GPU work.
#
#   sbatch -p apudev -t 15 jobs/views.sbatch [sequence ids and rectify.py arguments]
#
# Log: /ptmp/enazaren/gray/runs/gray-views-<job id>.log     Output: /ptmp/enazaren/gray/runs/rectify
#
#SBATCH -J gray-views
#SBATCH -t 00:15:00
#SBATCH --constraint=apu --gres=gpu:1 -c 24 --mem=110000
#SBATCH -D /u/enazaren/projects/gray
#SBATCH -o /ptmp/enazaren/gray/runs/%x-%j.log
set -eo pipefail
if [ $# -eq 0 ]; then set -- 20230621_s1_kevin_davis_act3_bw8qui 20230703_s0_tommy_ayala_act0_c07rbi; fi
/u/enazaren/envs/torch/bin/python -u -m gray.views --out /ptmp/enazaren/gray/runs/rectify "$@"

#!/bin/bash
# Submit a training as a chain of 1-hour jobs that resume one another (run on the login node).
#
#   jobs/chain_train.sh <count> <job file> <run name> [arguments of the training script]
#   jobs/chain_train.sh 4 jobs/motion.sbatch motion-v2 --patches /ptmp/enazaren/gray/grids
#
# Why: on Viper's GPU partition short jobs are fitted into gaps between long ones, and a job that asks for
# 8 hours rarely fits one (24 Nov 2026: 97 of 274 running jobs had a 1-hour limit; ours with 8 hours waited over a
# day). The training scripts save their state at every evaluation and are started with --resume, so a later job
# continues where the one before stopped; a job that finds the training complete ends at once.
# Pick <count> as the expected hours plus one. Set --eval-every so that a state is saved every few minutes.
# Prints the last job id, for a test that should follow:  sbatch --dependency=afterany:<id> ...
set -eo pipefail
count="$1"; file="$2"; shift 2
previous=$(sbatch --parsable -t 01:00:00 "$file" "$@")
echo "first $previous"
for _ in $(seq 2 "$count"); do
  previous=$(sbatch --parsable -t 01:00:00 --dependency=afterany:"$previous" "$file" "$@")
done
echo "last $previous"

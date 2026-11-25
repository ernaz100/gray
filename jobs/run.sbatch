#!/bin/bash
# One hour of a training, on any cluster (jobs/clusters/<name>.sh says where things are). Submitted by jobs/remote.sh,
# which jobs/submit.sh calls from the workstation; not meant to be submitted by hand.
#
# Runs  python -m gray.<module> --amp --resume --save-dir <data>/runs/<name> --workers <n> [arguments]
# with "@data" in the arguments replaced by the cluster's data folder. The training scripts save their state at every
# evaluation and resume from it, so a chain of these jobs is one training.
#SBATCH -t 01:00:00
set -eo pipefail
source "jobs/clusters/$GRAY_CLUSTER.sh"
[ -n "$GRAY_ENV" ] && export $GRAY_ENV
args=()
for a in "$@"; do args+=("${a//@data/$GRAY_DATA}"); done
exec "$GRAY_PYTHON" -u -m "gray.$GRAY_MODULE" --amp --resume --save-dir "$GRAY_DATA/runs/$GRAY_NAME" --workers "$GRAY_WORKERS" "${args[@]}"

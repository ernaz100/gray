#!/bin/bash
# One run of a script of the repository on a GPU, on any cluster (jobs/clusters/<name>.sh): an evaluation, a check, a conversion.
#
#   sbatch [-p <test partition> -t 15] --export=ALL,GRAY_CLUSTER=viper -D <repository> -o <log> jobs/once.sbatch <module> [arguments]
#
# Runs  python -m gray.<module> [arguments]  with "@data" replaced by the cluster's data folder.
#SBATCH -J gray-once
#SBATCH -t 00:30:00
set -eo pipefail
source "jobs/clusters/$GRAY_CLUSTER.sh"
[ -n "$GRAY_ENV" ] && export $GRAY_ENV
module="$1"; shift
args=()
for a in "$@"; do args+=("${a//@data/$GRAY_DATA}"); done
exec "$GRAY_PYTHON" -u -m "gray.$module" "${args[@]}"

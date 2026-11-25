#!/bin/bash
# The cluster's side of jobs/submit.sh. Run in the repository on a cluster:
#
#   jobs/remote.sh <cluster> when   <hours> <module> <name> [arguments]   # prints "<start, seconds since 1970> <partition>" or "missing <path>"
#   jobs/remote.sh <cluster> submit <hours> <module> <name> [arguments]   # submits <hours> jobs of 1 h, each after the one before
#
# GRAY_PARTITION in the environment fixes the partition for submit (as "when" printed it).
set -eo pipefail
cluster="$1"; mode="$2"; hours="$3"; module="$4"; name="$5"; shift 5
source "jobs/clusters/$cluster.sh"
mkdir -p "$GRAY_DATA/runs"
for a in "$@"; do
  case "$a" in @data/*) [ -e "${a//@data/$GRAY_DATA}" ] || { echo "missing ${a//@data/$GRAY_DATA}"; exit 0; } ;; esac
done
[ -x "$GRAY_PYTHON" ] || { echo "missing $GRAY_PYTHON"; exit 0; }
common=(-J "gray-$name" -D "$GRAY_CODE" -o "$GRAY_DATA/runs/%x-%j.log" -t 01:00:00 $GRAY_SBATCH "--export=ALL,GRAY_CLUSTER=$cluster,GRAY_MODULE=$module,GRAY_NAME=$name")
partition() { [ "$1" = default ] || echo "-p $1"; }
if [ "$mode" = when ]; then
  best=""
  for p in $GRAY_PARTITIONS; do
    told=$(sbatch --test-only $(partition "$p") "${common[@]}" jobs/run.sbatch "$@" 2>&1 || true)
    at=$(echo "$told" | grep -o "start at [0-9T:-]*" | cut -d" " -f3)
    [ -n "$at" ] || continue
    seconds=$(date -d "$at" +%s)
    if [ -z "$best" ] || [ "$seconds" -lt "${best%% *}" ]; then best="$seconds $p"; fi
  done
  echo "${best:-unknown}"
else
  p="${GRAY_PARTITION:-$(echo $GRAY_PARTITIONS | cut -d" " -f1)}"
  previous=$(sbatch --parsable $(partition "$p") "${common[@]}" jobs/run.sbatch "$@")
  echo "first $previous"
  for _ in $(seq 2 "$hours"); do
    previous=$(sbatch --parsable $(partition "$p") --dependency=afterany:"$previous" "${common[@]}" jobs/run.sbatch "$@")
  done
  echo "last $previous"
fi

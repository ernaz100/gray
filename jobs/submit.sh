#!/bin/bash
# Submit a training from the workstation to a cluster, or to whichever cluster would start it soonest.
#
#   jobs/submit.sh <cluster|fastest|ask> <hours> <module> <run name> [arguments of the training script]
#
#   jobs/submit.sh fastest 4 motion ee-motion-v1 --ee4d @data/ee4d
#   jobs/submit.sh ask     4 motion ee-motion-v1 --ee4d @data/ee4d     # only says when each cluster would start it
#   jobs/submit.sh viper   4 head   ee-head-v1   --ee4d @data/ee4d
#
# The training runs as <hours> jobs of 1 h that resume one another (jobs/run.sbatch). "@data" stands for the cluster's
# data folder, so the same command works everywhere. A cluster is asked only if it is connected (hpc status), has the
# repository, and has every "@data/..." path the command names. The repository on the cluster is updated from GitHub first.
# Each cluster is described by one file in jobs/clusters/; see the README there for adding one.
set -eo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
target="$1"; hours="$2"; module="$3"; name="$4"; shift 4
quoted=$(printf "%q " "$@")
status=$(~/bin/hpc status 2>&1)
best=""; table=""
for file in "$here"/clusters/*.sh; do
  cluster=$(basename "$file" .sh)
  [ "$target" = fastest ] || [ "$target" = ask ] || [ "$target" = "$cluster" ] || continue
  if ! echo "$status" | grep -q "^$cluster: connected"; then table+="$cluster: not connected (hpc connect $cluster)\n"; continue; fi
  code=$(grep "^GRAY_CODE=" "$file" | cut -d= -f2)
  told=$(~/bin/hpc run "$cluster" "cd $code 2>/dev/null && git pull -q 2>/dev/null; cd $code 2>/dev/null && jobs/remote.sh $cluster when $hours $module $name $quoted || echo no-repository" 2>/dev/null | tail -1)
  case "$told" in
    no-repository|"") table+="$cluster: the repository is not there (see jobs/clusters/README.md)\n" ;;
    missing*) table+="$cluster: $told\n" ;;
    unknown) table+="$cluster: the scheduler gave no start time\n" ;;
    *) seconds="${told%% *}"; partition="${told##* }"
       table+="$cluster: would start $(date -d "@$seconds" "+%a %H:%M")$([ "$partition" = default ] || echo " on $partition")\n"
       if [ -z "$best" ] || [ "$seconds" -lt "${best%% *}" ]; then best="$seconds $cluster $partition"; fi ;;
  esac
done
printf "$table"
[ "$target" = ask ] && exit 0
[ -n "$best" ] || { echo "no cluster can take this job"; exit 1; }
read -r _seconds cluster partition <<< "$best"
code=$(grep "^GRAY_CODE=" "$here/clusters/$cluster.sh" | cut -d= -f2)
echo "submitting to $cluster"
~/bin/hpc run "$cluster" "cd $code && GRAY_PARTITION=$partition jobs/remote.sh $cluster submit $hours $module $name $quoted"

#!/usr/bin/env bash
# List every compute process on every GPU: owner, runtime, docker container, command.
# Usage: bash tools/gpu_procs.sh [cmd_width]   (default command width 120 chars)
WIDTH=${1:-120}

declare -A IDX CNAME
while IFS=', ' read -r idx uuid; do IDX[$uuid]=$idx; done \
    < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
if command -v docker >/dev/null 2>&1; then
    while read -r id name; do CNAME[$id]=$name; done < <(docker ps --no-trunc --format '{{.ID}} {{.Names}}' 2>/dev/null)
fi

echo "== GPU summary"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader \
    | awk -F', ' '{printf "GPU %s  mem %9s / %-9s  util %s\n", $1, $2, $3, $4}'

echo
echo "== Processes"
printf '%-4s %-8s %-10s %-10s %-11s %-20s %s\n' GPU PID MEM USER ELAPSED CONTAINER COMMAND
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits \
    | while IFS=', ' read -r uuid pid mem; do
        user=$(ps -o user= -p "$pid" 2>/dev/null | xargs)
        etime=$(ps -o etime= -p "$pid" 2>/dev/null | xargs)
        cid=$(grep -oE '[0-9a-f]{64}' "/proc/$pid/cgroup" 2>/dev/null | head -1)
        container=${CNAME[$cid]:-${cid:+${cid:0:12}}}
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-"$WIDTH")
        printf '%-4s %-8s %-10s %-10s %-11s %-20s %s\n' \
            "${IDX[$uuid]}" "$pid" "${mem}MiB" "${user:-?}" "${etime:-?}" "${container:-host}" "${cmd:-?}"
    done | sort -n -k1,1

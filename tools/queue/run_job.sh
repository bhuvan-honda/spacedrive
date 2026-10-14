#!/usr/bin/env bash
# One queued experiment: train -> test (format-only, explicit checkpoint) -> planning eval -> results.md row.
# Runs on the HOST; every python step runs in the spacedrive_flash container (repo mounted at /workspace).
#
#   tools/queue/run_job.sh NAME CONFIG SEED GPUS PORT [cfg-options ...]
#   e.g. tools/queue/run_job.sh occ_l28 projects/configs/spacedrive/sd2/sd2_occ.py 888 0,1,2,3 29610 model.bev_occ.layer=28
#
# Work dir: results/spacedrive_sd2_<NAME>/   Logs: logs/queue/<NAME>_{train,inf,eval}.txt
# Re-running a job resumes training from latest.pth and skips finished stages
# (markers .train_done / .test_done in the work dir). Exit code != 0 on failure.
set -uo pipefail

NAME=$1; CONFIG=$2; SEED=$3; GPUS=$4; PORT=$5; shift 5
OPTS=("$@")

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WD_REL=results/spacedrive_sd2_${NAME}
WD=/workspace/$WD_REL
HWD=$ROOT/$WD_REL
LOGD=$ROOT/logs/queue
mkdir -p "$LOGD" "$HWD"
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DX=(docker exec -w /workspace -e CUDA_VISIBLE_DEVICES="$GPUS" -e PYTHONPATH=/workspace spacedrive_flash)
CFG_OPTS=(model.save_path=$WD/_results_planning_only/ "${OPTS[@]}")

log() { echo "[$(date '+%a %F %T')] [$NAME] $*" | tee -a "$LOGD/${NAME}_job.txt"; }

MAX_ITERS=$("${DX[@]}" python -c "from mmcv import Config; print(Config.fromfile('$CONFIG').runner.max_iters)" 2>/dev/null | tail -1)
for o in "${OPTS[@]}"; do [[ $o == runner.max_iters=* ]] && MAX_ITERS=${o#*=}; done
if ! [[ "$MAX_ITERS" =~ ^[0-9]+$ ]]; then log "cannot read runner.max_iters from $CONFIG (${OPTS[*]:-})"; exit 2; fi
log "config=$CONFIG seed=$SEED gpus=$GPUS port=$PORT max_iters=$MAX_ITERS opts=${OPTS[*]:-}"

# ---------------- train (one automatic resume on failure) ----------------
if [[ ! -f $HWD/.train_done ]]; then
    for attempt in 1 2; do
        RESUME=()
        [[ -f $HWD/latest.pth ]] && RESUME=(--resume-from $WD/latest.pth)
        log "train attempt $attempt ${RESUME[*]:-}"
        "${DX[@]}" python -m torch.distributed.launch --use_env --nproc_per_node=$NGPU --master_port=$PORT \
            scripts/train/train.py "$CONFIG" --seed "$SEED" --launcher pytorch --work-dir $WD \
            "${RESUME[@]}" --cfg-options "${CFG_OPTS[@]}" >> "$LOGD/${NAME}_train.txt" 2>&1
        rc=$?
        if [[ $rc -eq 0 && -f $HWD/iter_${MAX_ITERS}.pth ]]; then
            touch "$HWD/.train_done"; break
        fi
        log "train exited rc=$rc, iter_${MAX_ITERS}.pth $( [[ -f $HWD/iter_${MAX_ITERS}.pth ]] && echo present || echo missing)"
        sleep 60
    done
    [[ -f $HWD/.train_done ]] || { log "TRAIN FAILED"; exit 1; }
    log "train done"
fi

# ---------------- test (format-only) ----------------
CKPT=$WD/iter_${MAX_ITERS}.pth
if [[ ! -f $HWD/.test_done ]]; then
    log "test $CKPT"
    echo "CHECKPOINT: $CKPT" >> "$LOGD/${NAME}_inf.txt"
    "${DX[@]}" python -m torch.distributed.launch --use_env --nproc_per_node=$NGPU --master_port=$((PORT + 1)) \
        scripts/test/test.py "$CONFIG" "$CKPT" --seed 888 --launcher pytorch --format-only \
        --cfg-options "${CFG_OPTS[@]}" >> "$LOGD/${NAME}_inf.txt" 2>&1
    rc=$?
    n=$(ls "$HWD/_results_planning_only" 2>/dev/null | wc -l)
    log "test rc=$rc, $n prediction files"
    if [[ $rc -ne 0 || $n -lt 6019 ]]; then log "TEST FAILED"; exit 1; fi
    touch "$HWD/.test_done"
fi

# ---------------- eval + results.md ----------------
log "eval"
"${DX[@]}" python scripts/evaluation/eval_planning.py --pred_path $WD_REL/_results_planning_only/ \
    --base_path data/nuscenes/ > "$LOGD/${NAME}_eval.txt" 2>&1 || { log "EVAL FAILED"; exit 1; }
grep -E "l2_|plan_obj_box_col|plan_boundary" "$LOGD/${NAME}_eval.txt" | tail -9 | tee -a "$LOGD/${NAME}_job.txt"
"${DX[@]}" python tools/results_md.py add --name "sd2_${NAME}" --eval-log logs/queue/${NAME}_eval.txt \
    --inf-log logs/queue/${NAME}_inf.txt --work-dir $WD_REL --ckpt-iter "$MAX_ITERS" \
    --notes "queue: $(basename "$CONFIG") seed $SEED ${OPTS[*]:-}" >> "$LOGD/${NAME}_job.txt" 2>&1 \
    || log "results.md update failed (eval itself is fine)"
log "DONE"

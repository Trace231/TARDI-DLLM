#!/usr/bin/env bash
cd /data/llada_eval
GD=results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/matched_grid_v1
# 等 ours_100 评测完成(orphan eval 进程会写出 raw)
for i in $(seq 1 60); do [ -f "$GD/raw_ours_100.json" ] && break; sleep 20; done
echo "[$(date)] ours_100 raw ready, 续跑 vanilla_100 + rslora_600" >> "$GD"/logs/driver.log
bash scripts/run_matched_grid.sh >> "$GD"/logs/driver.log 2>&1

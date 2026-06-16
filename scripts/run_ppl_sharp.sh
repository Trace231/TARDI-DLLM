#!/usr/bin/env bash
set -uo pipefail
cd /data/llada_eval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# proper mechanism + SHARP PPL sampling (temp 0.2, eps 0.03) so the bandit ACTUALLY reallocates noise
python3 scripts/train_llada_choice_noise_lora.py --model /data/hf/models/GSAI-ML/LLaDA-8B-Instruct --train-jsonl results/domain_shift/task_aware/lora_opt_v1/train/domain_mix_9task_balanced_exclude_eval_seed101.jsonl --peft-variant lora   --mode choice_noise --batch-size 1 --grad-accum 8 --lr 1e-4 --lora-r 16 --lora-alpha 32 --mask-prompt   --denoise-weight 0.15 --consistency-weight 0.05 --noise-ratios 0.15,0.35,0.65,0.85   --lr-scheduler cosine --warmup-ratio 0.1 --max-steps 600 --val-fraction 0.15 --val-every 40 --weight-decay 0.05   --adaptive-noise reducible_loss --adaptive-where sampling --adaptive-signal denoise_ppl   --adaptive-eps 0.03 --adaptive-temp 0.2 --adaptive-ema 0.95 --adaptive-fast-ema 0.7   --out results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/ppl_sharp/best_ppl_sharp > results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/ppl_sharp/logs/train.log 2>&1
echo "[$(date)] trained. scorecard tilt:"
python3 -c "import json;d=json.load(open('results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/ppl_sharp/best_ppl_sharp/adaptive_noise_ema.json'));t=[0]*4
for k in d['ema_seen']:
 for i,v in enumerate(d['ema_seen'][k]): t[i]+=v
T=sum(t);print('  global share:',[f'{100*x/T:.0f}%' for x in t],'(uniform=25%; sharp should deviate MORE)')"
python3 scripts/eval_domain_shift.py --backend llada --model /data/hf/models/GSAI-ML/LLaDA-8B-Instruct --adapter results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/ppl_sharp/best_ppl_sharp/final_adapter   --tasks mmlu_pro,pubmedqa,ceval_computer_network,sciq,winogrande,commonsenseqa,arc_challenge,hellaswag,boolq --limit 50 --seed 23 --steps 32 --gen-length 32 --block-length 32   --prompt-style final_label_typed --out results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/ppl_sharp/raw_eval450.json > results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/ppl_sharp/logs/eval450.log 2>&1
python3 -c "import json,statistics as s;d=json.load(open('results/domain_shift/task_aware/solid_v2/related_work_v1/ours_trained/ppl_sharp/raw_eval450.json'));print('[$(date)] best_ppl_sharp eval-450 = %.4f (vs best_uniform_v2 eval-450 0.7467)'%s.mean([v['accuracy'] for v in d['summary'].values()]))"

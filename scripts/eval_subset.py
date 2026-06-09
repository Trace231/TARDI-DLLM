import argparse, json, os, re, time, sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
try:
    from peft import PeftModel
except Exception:
    PeftModel = None
try:
    from nara_adapter import is_nara_adapter, load_nara_adapter
except Exception:
    is_nara_adapter = None
    load_nara_adapter = None

sys.path.insert(0, '/data/llada_eval/LLaDA')
from generate import generate as llada_generate


def normalize_num(s):
    nums = re.findall(r'-?\d+(?:\.\d+)?', s.replace(',', ''))
    return nums[-1] if nums else ''


def parse_letter(s, choices='ABCDE'):
    m = re.search(r'(?i)(?:answer\s*(?:is|:)?\s*)?\b([' + choices + r'])\b', s.strip())
    return m.group(1).upper() if m else ''


def parse_bool(s):
    low = s.lower()
    yes = re.search(r'\byes\b|\btrue\b', low)
    no = re.search(r'\bno\b|\bfalse\b', low)
    if yes and (not no or yes.start() < no.start()): return 'yes'
    if no: return 'no'
    return ''


def build_samples(task, limit, seed):
    import random
    rng = random.Random(seed)
    samples = []
    if task == 'gsm8k':
        ds = load_dataset('openai/gsm8k', 'main', split='test')
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            gold = normalize_num(ex['answer'].split('####')[-1])
            prompt = 'Solve the math problem. Show brief reasoning, then end with "Final answer: <number>".\n\nProblem: ' + ex['question']
            samples.append({'id': str(i), 'prompt': prompt, 'gold': gold, 'metric': 'number'})
    elif task == 'arc_challenge':
        ds = load_dataset('ai2_arc', 'ARC-Challenge', split='test')
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            labels = ex['choices']['label']
            texts = ex['choices']['text']
            mapping = []
            for j, txt in enumerate(texts):
                letter = chr(ord('A') + j)
                mapping.append(f'{letter}. {txt}')
            label_to_letter = {lab: chr(ord('A') + j) for j, lab in enumerate(labels)}
            gold = label_to_letter.get(ex['answerKey'], ex['answerKey'])
            prompt = 'Answer the multiple-choice science question. Reply with the letter only.\n\nQuestion: ' + ex['question'] + '\n' + '\n'.join(mapping) + '\nAnswer:'
            samples.append({'id': str(i), 'prompt': prompt, 'gold': gold, 'metric': 'letter'})
    elif task == 'hellaswag':
        ds = load_dataset('Rowan/hellaswag', split='validation')
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            opts = [f'{chr(ord("A") + j)}. {ending}' for j, ending in enumerate(ex['endings'])]
            gold = chr(ord('A') + int(ex['label']))
            prompt = 'Choose the most plausible continuation. Reply with the letter only.\n\nContext: ' + ex['ctx'] + '\n' + '\n'.join(opts) + '\nAnswer:'
            samples.append({'id': str(i), 'prompt': prompt, 'gold': gold, 'metric': 'letter'})
    elif task == 'boolq':
        ds = load_dataset('google/boolq', split='validation')
        idxs = rng.sample(range(len(ds)), min(limit, len(ds)))
        for i in idxs:
            ex = ds[i]
            gold = 'yes' if ex['answer'] else 'no'
            prompt = 'Read the passage and answer the question with yes or no only.\n\nPassage: ' + ex['passage'] + '\nQuestion: ' + ex['question'] + '\nAnswer:'
            samples.append({'id': str(i), 'prompt': prompt, 'gold': gold, 'metric': 'bool'})
    else:
        raise ValueError(task)
    return samples


def numeric_equal(pred, gold):
    try:
        return abs(float(str(pred).replace(',', '')) - float(str(gold).replace(',', ''))) < 1e-6
    except Exception:
        return str(pred) == str(gold)


def score(metric, text, gold):
    if metric == 'number':
        pred = normalize_num(text)
        ok = numeric_equal(pred, gold)
    elif metric == 'letter':
        pred = parse_letter(text)
        ok = pred == gold
    elif metric == 'bool':
        pred = parse_bool(text)
        ok = pred == gold
    else:
        pred = text.strip()
        ok = pred == gold
    return pred, ok


def load_llada(path, adapter=None):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tok.padding_side = 'left'
    model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to('cuda')
    if adapter:
        if is_nara_adapter is not None and is_nara_adapter(adapter):
            model = load_nara_adapter(model, adapter)
            model.to('cuda')
            model.eval()
            return tok, model
        if PeftModel is None:
            raise RuntimeError('peft is required for --adapter')
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def load_ar(path, adapter=None):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda').eval()
    if adapter:
        if PeftModel is None:
            raise RuntimeError('peft is required for --adapter')
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
    return tok, model


def chat_prompt(tok, prompt):
    try:
        return tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False)
    except Exception:
        return prompt


def generate_llada(tok, model, prompts, args):
    texts = [chat_prompt(tok, p) for p in prompts]
    enc = tok(texts, add_special_tokens=False, padding=True, return_tensors='pt')
    input_ids = enc['input_ids'].to('cuda')
    attention_mask = enc['attention_mask'].to('cuda')
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = llada_generate(model, input_ids, attention_mask=attention_mask, steps=args.steps, gen_length=args.gen_length,
                         block_length=args.block_length, temperature=args.temperature, cfg_scale=args.cfg,
                         remasking=args.remasking, logits_eos_inf=args.logits_eos_inf,
                         confidence_eos_eot_inf=args.confidence_eos_eot_inf)
    dt = time.time() - t0
    decoded = tok.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    return decoded, dt, torch.cuda.max_memory_allocated() / 1024**3


def generate_ar(tok, model, prompts, args):
    texts = [chat_prompt(tok, p) for p in prompts]
    enc = tok(texts, padding=True, return_tensors='pt').to('cuda')
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    dt = time.time() - t0
    decoded = tok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
    return decoded, dt, torch.cuda.max_memory_allocated() / 1024**3


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['llada','ar'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', default=None, help='Optional PEFT adapter path for LLaDA backend')
    p.add_argument('--tasks', default='gsm8k,arc_challenge,boolq,hellaswag')
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--gen-length', type=int, default=128)
    p.add_argument('--block-length', type=int, default=32)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--cfg', type=float, default=0.0)
    p.add_argument('--remasking', default='low_confidence')
    p.add_argument('--logits-eos-inf', action='store_true')
    p.add_argument('--confidence-eos-eot-inf', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=128)
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok, model = load_llada(args.model, args.adapter) if args.backend == 'llada' else load_ar(args.model)
    all_rows, summary = [], {}
    for task in args.tasks.split(','):
        samples = build_samples(task, args.limit, args.seed)
        correct = 0
        total_time = 0.0
        max_mem = 0.0
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start:start + args.batch_size]
            outs, dt, mem = generate_llada(tok, model, [x['prompt'] for x in batch], args) if args.backend == 'llada' else generate_ar(tok, model, [x['prompt'] for x in batch], args)
            total_time += dt
            max_mem = max(max_mem, mem)
            for ex, out in zip(batch, outs):
                pred, ok = score(ex['metric'], out, ex['gold'])
                correct += int(ok)
                row = dict(task=task, id=ex['id'], gold=ex['gold'], pred=pred, correct=ok, output=out, prompt=ex['prompt'])
                all_rows.append(row)
                print(json.dumps({k: row[k] for k in ['task','id','gold','pred','correct']}, ensure_ascii=False), flush=True)
        summary[task] = {'accuracy': correct / len(samples), 'n': len(samples), 'seconds': total_time, 'peak_mem_gb': max_mem}
    payload = {'backend': args.backend, 'model': args.model, 'args': vars(args), 'summary': summary, 'rows': all_rows}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

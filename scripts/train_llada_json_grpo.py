#!/usr/bin/env python3
from dataclasses import dataclass, field

from datasets import load_dataset
from peft import LoraConfig
from trl import ModelConfig, TrlParser

import sys

sys.path.insert(0, "/data/llada_eval/dllm")

import dllm
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
from dllm.pipelines.rl import DiffuGRPOConfig, DiffuGRPOTrainer
from dllm.pipelines.rl.grpo.rewards.math import (
    typed_final_label_correctness_reward_func,
    typed_final_label_format_reward_func,
)


logger = dllm.utils.get_default_logger(__name__)


@dataclass
class TrainingArguments(DiffuGRPOConfig):
    output_dir: str = "/data/llada_eval/results/adaptation/llada_json_grpo_lora"
    train_jsonl: str = field(default="", metadata={"help": "Shared JSONL with prompt/answer rows."})


def train():
    parser = TrlParser((TrainingArguments, ModelConfig))
    training_args, model_config = parser.parse_args_and_config()
    if not training_args.train_jsonl:
        raise ValueError("--train_jsonl is required")
    if not model_config.model_name_or_path:
        model_config.model_name_or_path = "/data/hf/models/GSAI-ML/LLaDA-8B-Instruct"

    dataset = load_dataset("json", data_files=training_args.train_jsonl, split="train")
    train_set = dataset.shuffle(seed=training_args.seed)
    reward_functions = [typed_final_label_format_reward_func, typed_final_label_correctness_reward_func]

    model_args = dllm.utils.ModelArguments(
        model_name_or_path=model_config.model_name_or_path,
        load_in_4bit=(model_config.load_in_4bit if hasattr(model_config, "load_in_4bit") else False),
    )
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    model.config.use_cache = False

    peft_config = None
    if model_config.lora_r and model_config.lora_r > 0:
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            task_type="CAUSAL_LM",
            lora_dropout=model_config.lora_dropout,
        )

    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    sampler_config = MDLMSamplerConfig(
        steps=training_args.steps,
        max_new_tokens=training_args.max_completion_length,
        block_size=training_args.block_size,
        temperature=training_args.temperature or 0.0,
        cfg_scale=training_args.cfg_scale,
        remasking=training_args.remasking,
    )

    logger.info("Start controlled JSON GRPO training...")
    trainer = DiffuGRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_set,
        processing_class=tokenizer,
        peft_config=peft_config,
        sampler=sampler,
        sampler_config=sampler_config,
    )
    trainer.train()


if __name__ == "__main__":
    train()

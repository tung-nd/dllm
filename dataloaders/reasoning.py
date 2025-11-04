from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch_utils.distributed import get_rank, get_world_size
from dataloaders.sampler import InfiniteSampler
from transformers import AutoTokenizer

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""

def collate_fn_s1k(batch, tokenizer: AutoTokenizer):
    batch_inputs = []
    batch_prompts = []
    batch_inputs_without_final_answer = []
    for item in batch:
        question = SYSTEM_PROMPT + "\n\n" + item["question"]
        trajectory = f"<reasoning>{item['thinking_trajectories'][0]}</reasoning>\n\n<answer>{item['attempt']}</answer>"
        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": trajectory}]
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        inputs_without_final_answer = inputs.split("\n\n<answer>")[0]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
        
        batch_inputs.append(inputs)
        batch_prompts.append(prompt)
        batch_inputs_without_final_answer.append(inputs_without_final_answer)
        
        # tokenized_input = tokenizer(
        #     inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        # ).input_ids.squeeze(0)
        # tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        # # tokenized_input_without_final_answer = tokenizer(inputs_without_final_answer, return_tensors="pt", truncation=True, max_length=max_length)
        # input_ids.append(tokenized_input)
        # prompt_lengths.append(tokenized_prompt.attention_mask.sum(-1))
    
    return {
        "inputs": batch_inputs,
        "prompts": batch_prompts,
        "inputs_without_final_answer": batch_inputs_without_final_answer,
    }

def load_s1k_dataset(
    local_path: str,
    batch_size: int,
    split: str = 'train', 
    tokenizer: AutoTokenizer = None,
    num_workers: int = 8,
    rank: int = None,
    num_replicas: int = None,
    seed: int = 112, 
):
    ds = load_dataset(local_path, split=split)
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)
    if rank is not None and num_replicas is not None:
        sampler = InfiniteSampler(
            ds, rank=rank, num_replicas=num_replicas, 
        )
    else:
        sampler = InfiniteSampler(
            ds, rank=get_rank(), num_replicas=get_world_size(), 
        )

    dl = DataLoader(
        ds, collate_fn=lambda batch: collate_fn_s1k(batch, tokenizer),
        batch_size=batch_size, sampler=sampler, 
        num_workers=num_workers, pin_memory=True, 
    )

    return dl
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

def collate_fn_s1k(batch, tokenizer: AutoTokenizer, max_length: int):
    """
    This function returns:
    - input_ids: the tokenized input ids [B, N]
    - prompt_lengths: the length of the prompt [B]
    - prompt_cot_lengths: the length of the prompt and reasoning tokens [B]
    - attention_mask: causal for prompt+reasoning, no mask for answer [B, N, N]
    - used_cot_ratio: the ratio of the used reasoning tokens [B]
    """
    batch_input_ids = []
    batch_prompt_lengths = []
    batch_prompt_cot_lengths = []
    batch_attention_mask = []
    batch_used_cot_ratio = []
    for item in batch:
        question = SYSTEM_PROMPT + "\n\n" + item["question"]
        trajectory = f"<reasoning>{item['thinking_trajectories'][0]}</reasoning>\n\n<answer>{item['attempt']}</answer>"
        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": trajectory}]
        input = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        prompt_cot = input.split("\n\n<answer>")[0]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
        
        # tokenize with no padding or truncation
        tokenized_prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        prompt_length = tokenized_prompt_ids.shape[0]
        tokenized_prompt_cot_ids = tokenizer(prompt_cot, return_tensors="pt").input_ids[0]
        cot_length = tokenized_prompt_cot_ids.shape[0] - prompt_length
        tokenized_input_ids = tokenizer(input, return_tensors="pt").input_ids[0]
        answer_length = tokenized_input_ids.shape[0] - (cot_length + prompt_length)
        
        # randomly cutoff the reasoning tokens to have prompt + partial reasoning + answer
        max_used_cot_length = min(cot_length, max_length - prompt_length - answer_length)
        used_cot_length = np.random.randint(0, max_used_cot_length + 1) if max_used_cot_length > 0 else 0
        used_cot_ids = tokenized_prompt_cot_ids[:used_cot_length]
        used_cot_ratio = used_cot_length / cot_length
        prompt_cot_length = used_cot_length + prompt_length
        
        full_input_ids = torch.cat([tokenized_prompt_ids, used_cot_ids, tokenized_input_ids[-answer_length:]], dim=0)
        # truncate if needed
        full_input_ids = full_input_ids[:max_length]
        
        # padding if needed
        full_input_ids = torch.cat([full_input_ids, torch.full((max_length - full_input_ids.shape[0],), tokenizer.pad_token_id)])
        
        # construct attention mask
        attention_mask = torch.ones((max_length, max_length))
        attention_mask[:prompt_cot_length, :prompt_cot_length] = torch.tril(torch.ones((prompt_cot_length, prompt_cot_length)))
        attention_mask[:prompt_cot_length, prompt_cot_length:] = 0
        
        batch_input_ids.append(full_input_ids)
        batch_prompt_lengths.append(prompt_length)
        batch_prompt_cot_lengths.append(prompt_cot_length)
        batch_attention_mask.append(attention_mask)
        batch_used_cot_ratio.append(used_cot_ratio)
    
    return (
        torch.stack(batch_input_ids),
        torch.tensor(batch_prompt_lengths),
        torch.tensor(batch_prompt_cot_lengths),
        torch.stack(batch_attention_mask),
        torch.tensor(batch_used_cot_ratio),
    )

def load_s1k_dataset(
    local_path: str,
    batch_size: int,
    tokenizer: AutoTokenizer,
    max_length: int = 4096,
    num_workers: int = 8,
    rank: int = None,
    num_replicas: int = None,
    seed: int = 112, 
    val_ratio: float = 0.01,
):
    ds = load_dataset(local_path, split='train')
    ds = ds.with_format('torch')
    ds = ds.shuffle(seed=seed)
    # split train and val
    split = ds.train_test_split(test_size=val_ratio, seed=seed)
    train_ds = split['train']
    val_ds = split['test']
    if rank is not None and num_replicas is not None:
        sampler = InfiniteSampler(
            train_ds, rank=rank, num_replicas=num_replicas, 
        )
    else:
        sampler = InfiniteSampler(
            train_ds, rank=get_rank(), num_replicas=get_world_size(), 
        )

    train_dl = DataLoader(
        train_ds, collate_fn=lambda batch: collate_fn_s1k(batch, tokenizer, max_length),
        batch_size=batch_size, sampler=sampler, 
        num_workers=num_workers, pin_memory=True, 
    )
    val_dl = DataLoader(
        val_ds, collate_fn=lambda batch: collate_fn_s1k(batch, tokenizer, max_length),
        batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, 
    )

    return train_dl, val_dl

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
    ds = load_dataset("simplescaling/s1K", split="train")
    batch = ds.select(range(10))
    batch = collate_fn_s1k(batch, tokenizer, 4096)
    for key, value in batch.items():
        print(key, value.shape)
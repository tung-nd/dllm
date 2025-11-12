# This file is modified by MAPLE research lab, based on the original code from https://github.com/NVlabs/edm

# Original code is licensed under the following license:

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import numpy as np
import torch
import dnnlib
import wandb
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch_utils import distributed as dist
from torch_utils import misc
from tqdm.auto import tqdm
from networks.gated_lora import LoraGateContext, inject_gated_lora

def find_latest_wandb_id(wandb_dir):
    """
    Helper to find the most recent WANDB run id in the directory.
    """
    run_id = None
    if os.path.exists(wandb_dir) and os.path.isdir(wandb_dir):
        runs = [
            sub for sub in os.listdir(wandb_dir)
            if os.path.isdir(os.path.join(wandb_dir, sub)) and sub.startswith("run-")
        ]
        if runs:
            # Sort by mtime (most recent first)
            runs = sorted(runs, key=lambda s: os.path.getmtime(os.path.join(wandb_dir, s)), reverse=True)
            run_id = runs[0].split("-")[-1]
    return run_id

def forward_process(input_ids, prompt_cot_lengths, t, mask_token_id, eps=1e-3):
    B, N = input_ids.shape
    if t is None:
        t = torch.rand((B,), device=input_ids.device)

    t = (1 - eps) * t + eps
    t = t[:, None].repeat(1, N)

    mask_indices = torch.rand((B, N), device=input_ids.device) < t
    noisy_batch = torch.where(mask_indices, mask_token_id, input_ids)
    
    # do not mask the prompt+cot tokens
    token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
    prompt_cot_mask = (token_positions < prompt_cot_lengths.unsqueeze(1))
    noisy_batch = torch.where(prompt_cot_mask, input_ids, noisy_batch)
    
    return noisy_batch, t

def calculate_loss(model, noisy_batch, input_ids, lora_mask, prompt_lengths, prompt_cot_lengths, t, mask_token_id):
    # noisy_batch: [B, N]
    # input_ids: [B, N]
    # attention_mask: [B, N, N]
    # lora_mask: [B, N]
    # prompt_lengths: [B]
    # prompt_cot_lengths: [B]
    # t: [B]
    B, N = noisy_batch.shape
    
    with LoraGateContext(lora_mask):
        logits = model(input_ids=noisy_batch, prompt_cot_lengths=prompt_cot_lengths).logits # [B, N, vocab_size]
    
    position_ids = torch.arange(N, device=noisy_batch.device).unsqueeze(0).repeat(B, 1)
    
    # loss for cot tokens
    cot_mask = (position_ids >= prompt_lengths.unsqueeze(1)) & (position_ids <= prompt_cot_lengths.unsqueeze(1)) # [B, N]
    cot_ce_loss = F.cross_entropy(logits[cot_mask], input_ids[cot_mask], reduction='none').mean()
    masked_indices = (noisy_batch == mask_token_id)
    scaled_answer_loss = (F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / t[masked_indices]).mean()
    # dist.print0('cot_ce_loss', cot_ce_loss)
    # dist.print0('unscaled_answer_loss', unscaled_answer_loss)
    # dist.print0('scaled_answer_loss', scaled_answer_loss)
    # import sys
    # sys.exit()

    return cot_ce_loss, scaled_answer_loss

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    exp_name            = 'default',
    batch_size          = 512,
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    optimizer_kwargs    = {},       # Options for optimizer.
    seed                = 0,        # Global random seed.
    total_steps         = 200000,   # Training duration, measured in thousands of training images.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    grad_accumulation   = 1,
    lr_scheduler_kwargs = {},
    precision           = "fp16",
    resume_pt           = None,
    resume_state_dump   = None,
    resume_step         = 0,
    max_grad_norm       = 1000,
    val_frequency       = 100,
    val_max_iterations  = 100,
    checkpoint_frequency = 500,
    skip_spike_grad     = 10e10,
    tokenizer_kwargs    = {},
    activation_checkpointing = 'whole_layer',
    training_state_dir  = None,
    *args, **kwargs
):
    dist.print0(f"Useless parameters: \n {args}\n {kwargs}")
    opts = {
        "batch_size": batch_size,
        "data_loader_kwargs": data_loader_kwargs,
        "network_kwargs": network_kwargs,
        "optimizer_kwargs": optimizer_kwargs,
        "seed": seed,
        "total_steps": total_steps,
        "loss_scaling": loss_scaling,
        "val_frequency": val_frequency,
        "val_max_iterations": val_max_iterations,
        "checkpoint_frequency": checkpoint_frequency,
        "grad_accumulation": grad_accumulation,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
        "precision": precision,
        "resume_pt": resume_pt,
        "resume_state_dump": resume_state_dump,
        "resume_step": resume_step,
        "max_grad_norm": max_grad_norm,
        "skip_spike_grad": skip_spike_grad,
        "activation_checkpointing": activation_checkpointing,
    }
    # Initialize.
    rank = dist.get_rank()
    np.random.seed((seed * dist.get_world_size() + rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    precision_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    
    # Construct network.
    dist.print0('Constructing network...')
    model = dnnlib.util.construct_class_by_name(**network_kwargs) # subclass of torch.nn.Module
    # Inject PEFT-like LoRA with gating
    replaced = inject_gated_lora(
        model,
        target_keywords=("q_proj","k_proj","v_proj"),
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
    )
    print("Replaced:", len(replaced), "modules with gated lora")

    # Freeze everything except LoRA params (A/B) — mirrors PEFT training policy
    model.train()
    model.requires_grad_(True)
    # for n, p in model.named_parameters():
    #     if ".lora_A" in n or ".lora_B" in n:
    #         p.requires_grad = True
    #     else:
    #         p.requires_grad = False
    
    model_params = misc.count_parameters(model)
    
    # # model.model.set_activation_checkpointing(activation_checkpointing)
    model.gradient_checkpointing_enable()

    # tokenizer
    tokenizer = dnnlib.util.construct_class_by_name(**tokenizer_kwargs)
    if 'qwen' in tokenizer_kwargs.get('pretrained_model_name_or_path', '').lower() or 'llama' in tokenizer_kwargs.get('pretrained_model_name_or_path', '').lower():
        embedding_weights = model.model.embed_tokens.weight.data
        vocab_size = embedding_weights.shape[0]
        random_emb = torch.randn_like(embedding_weights[0]).unsqueeze(0)*0.01
        new_embedding_weights = torch.cat([embedding_weights, random_emb], dim=0)
        model.model.embed_tokens.weight = torch.nn.Parameter(new_embedding_weights)
        mask_token_id = vocab_size
        # mask_token_id = 62
    elif 'llada' in tokenizer_kwargs.get('pretrained_model_name_or_path', '').lower():
        mask_token_id = 126336
    
    # Load dataset.
    dist.print0('Loading dataset...')
    train_dataloader, val_dataloader = dnnlib.util.construct_class_by_name(
        **data_loader_kwargs,
        tokenizer=tokenizer,
    )

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    # Build parameter groups for different lrs (original vs new LoRA params)
    base_lr = optimizer_kwargs.get("lr", 1e-5)
    new_module_lr = optimizer_kwargs.get("new_module_lr", base_lr)
    new_module_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".lora_A" in name or ".lora_B" in name:
            new_module_params.append(param)
        else:
            other_params.append(param)
    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr})
    if new_module_params:
        param_groups.append({"params": new_module_params, "lr": new_module_lr})
    adamw_kwargs = {k: v for k, v in optimizer_kwargs.items() if k not in ["class_name", "lr", "new_module_lr"]}
    optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)

    # Setup LR scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=dnnlib.util.construct_class_by_name(**lr_scheduler_kwargs))

    accelerator = dist.get_accelerator()
    assert accelerator is not None
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
       model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    if resume_state_dump is not None and os.path.exists(resume_state_dump):
        dist.print0(f"Resume from {resume_state_dump}")
        accelerator.load_state(resume_state_dump)

    train_dataloader_iterator = iter(train_dataloader)
    if resume_state_dump is not None and os.path.exists(resume_state_dump):
        print(f"Resume from step {resume_step}, skipping training data ...")
        for i in range(resume_step):
            next(train_dataloader_iterator)

    # Train.
    training_step = resume_step # 0 for default
    
    dist.print0("parameters Required grad:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            dist.print0(name, p.shape)
    
    # tensorboard 
    if rank == 0:
        wandb_init_kwargs = dict(
            entity='tungnd',
            project="dllm",
            name=exp_name,
            dir=run_dir,
            config=opts,
        )
        # resume_id = find_latest_wandb_id(os.path.join(run_dir, "wandb"))
        # if resume_id is not None:
        #     wandb_init_kwargs.update({"id": resume_id, "resume": "must"})
        wandb.init(**wandb_init_kwargs)

    dist.print0(f'Training for {total_steps} steps in {precision_dtype}...')
    dist.print0(f"Model with Param: {model_params}")
    dist.print0(f"Total world size: {dist.get_world_size()}")
    dist.print0()

    pbar = tqdm(
        total=total_steps,
        initial=training_step,
        dynamic_ncols=True,
        desc="Training",
        disable=(rank != 0),
    )

    while True:
        if rank == 0 and not os.path.exists(run_dir):
            raise SystemError(f'Run directory "{run_dir}" does not exist.')
        
        optimizer.zero_grad(set_to_none=True)

        for round_idx in range(grad_accumulation):
            with misc.ddp_sync(model, sync=round_idx == grad_accumulation - 1):
                with torch.autocast(device_type="cuda", enabled=True, dtype=precision_dtype):
                    batch = next(train_dataloader_iterator)
                    input_ids, prompt_lengths, prompt_cot_lengths, lora_mask, used_cot_ratio = batch
                    input_ids = input_ids.to(device)
                    prompt_lengths = prompt_lengths.to(device)
                    prompt_cot_lengths = prompt_cot_lengths.to(device)
                    lora_mask = lora_mask.to(device)
                    used_cot_ratio = used_cot_ratio.to(device)
                    noisy_batch, t = forward_process(
                        input_ids,
                        prompt_cot_lengths=prompt_cot_lengths,
                        # t=(1-used_cot_ratio), # more cutoff -> more noise
                        t=None,
                        mask_token_id=mask_token_id,
                    ) # prompt+cot tokens are not masked
                    cot_loss, answer_loss = calculate_loss(
                        model,
                        noisy_batch,
                        input_ids.roll(-1, dims=1),
                        lora_mask,
                        prompt_lengths,
                        prompt_cot_lengths,
                        t,
                        mask_token_id=mask_token_id,
                    )
                    
                    loss = (cot_loss + answer_loss) / 2

                accelerator.backward(loss)

        _grad_norm = accelerator.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
        grad_norm = model.get_global_grad_norm() if hasattr(model, "get_global_grad_norm") else _grad_norm
        # In some cases the grad norm may not return a float
        if hasattr(grad_norm, "item"):
            grad_norm = grad_norm.item()
        
        scheduler.step(training_step)
        optimizer.step()
        
        # gather the loss and unscaled loss from all ranks for logging
        reduced_cot_loss = accelerator.reduce(cot_loss.detach(), reduction="mean")
        reduced_answer_loss = accelerator.reduce(answer_loss.detach(), reduction="mean")
        reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
        reduced_grad_norm = accelerator.reduce(torch.as_tensor(grad_norm, device=device, dtype=torch.float32), reduction="mean")

        if rank == 0:
            wandb.log(
                {
                    'training/lr': scheduler.get_lr()[0],
                    'training/grad_norm': reduced_grad_norm.item(),
                    'training/cot_loss': reduced_cot_loss.item(),
                    'training/answer_loss': reduced_answer_loss.item(),
                    'training/loss': reduced_loss.item(),
                },
                step=training_step
            )
        current_lr = scheduler.get_lr()[0]
        pbar.set_postfix({
            'loss': f"{reduced_loss.float().item():.4f}",
            'cot_loss': f"{reduced_cot_loss.float().item():.4f}",
            'answer_loss': f"{reduced_answer_loss.float().item():.4f}",
            'lr': f"{current_lr:.6f}",
            'grad': f"{reduced_grad_norm.float().item():.2f}",
        })
        pbar.update(1)

        training_step += 1

        if training_step % val_frequency == 0:
            model.eval()
            with torch.no_grad():
                # Inner validation progress over iterations for a specific t
                val_inner_pbar = tqdm(
                    total=val_max_iterations,
                    desc=f"Val",
                    dynamic_ncols=True,
                    leave=False,
                    disable=(rank != 0),
                )
                val_cot_loss, val_answer_loss = [], []
                val_iterations = 0
                for batch in val_dataloader:
                    if val_iterations >= val_max_iterations:
                        break
                    val_iterations += 1
                    input_ids, prompt_lengths, prompt_cot_lengths, lora_mask, used_cot_ratio = batch
                    t = 1 - used_cot_ratio
                    noisy_batch, t = forward_process(input_ids, prompt_cot_lengths, t=t, mask_token_id=mask_token_id)
                    cot_loss, answer_loss = calculate_loss(
                        model,
                        noisy_batch,
                        input_ids.roll(-1, dims=1),
                        lora_mask,
                        prompt_lengths,
                        prompt_cot_lengths,
                        t,
                        mask_token_id=mask_token_id,
                    )
                    val_cot_loss.append(cot_loss)
                    val_answer_loss.append(answer_loss)
                    val_inner_pbar.update(1)
                val_inner_pbar.close()
                val_cot_loss = torch.stack(val_cot_loss).mean()
                val_answer_loss = torch.stack(val_answer_loss).mean()
                # reduce the loss from all ranks
                val_cot_loss = accelerator.reduce(val_cot_loss, reduction="mean")
                val_answer_loss = accelerator.reduce(val_answer_loss, reduction="mean")
                if rank == 0:
                    val_dict = {f'validation/answer_loss': val_answer_loss.item()}
                    val_dict['validation/cot_loss'] = val_cot_loss.item()
                    wandb.log(val_dict, step=training_step)
            model.train()

        if training_step % checkpoint_frequency == 0:
            state_dict = accelerator.get_state_dict(model)
            save_path = os.path.join(training_state_dir, f'training-state-{training_step:06d}')
            accelerator.save_state(save_path)

            if rank == 0:
                save_path = os.path.join(run_dir, f'ckpt-{training_step:06d}')
                accelerator.unwrap_model(model).save_pretrained(
                    save_path, state_dict=state_dict, safe_serialization=True
                )
        
        accelerator.wait_for_everyone()
       
        if training_step >= total_steps:
            break
        
    # Done.
    pbar.close()
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------

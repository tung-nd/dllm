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

def forward_process(input_ids, t=None, mask_token_id=126336, eps=1e-3):
    B, N = input_ids.shape
    if t is None:
        t = torch.rand((B,), device=input_ids.device)

    t = (1 - eps) * t + eps
    t = t[:, None].repeat(1, N)

    mask_indices = torch.rand((B, N), device=input_ids.device) < t
    noisy_batch = torch.where(mask_indices, mask_token_id, input_ids)
    return noisy_batch, t, mask_indices

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
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
    # model.config.flash_attention = True
    model.train().requires_grad_(True).to(device)
    # model.eval().to(device)
    model_params = misc.count_parameters(model)
    
    model.model.set_activation_checkpointing(activation_checkpointing)

    # tokenizer
    tokenizer = dnnlib.util.construct_class_by_name(**tokenizer_kwargs)
    
    # Load dataset.
    dist.print0('Loading dataset...')
    dataloader_iterator = dnnlib.util.construct_class_by_name(
        **data_loader_kwargs,
        tokenizer=tokenizer,
    )

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    optimizer = dnnlib.util.construct_class_by_name(
        params=[p for p in model.parameters() if p.requires_grad],
        **optimizer_kwargs
    )

    # Setup LR scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=dnnlib.util.construct_class_by_name(**lr_scheduler_kwargs))

    accelerator = dist.get_accelerator()
    assert accelerator is not None
    model, optimizer, dataloader_iterator, scheduler = accelerator.prepare(
       model, optimizer, dataloader_iterator, scheduler
    )

    if resume_state_dump is not None and os.path.exists(resume_state_dump):
        dist.print0(f"Resume from {resume_state_dump}")
        accelerator.load_state(resume_state_dump)

    dataloader_iterator = iter(dataloader_iterator)
    if resume_state_dump is not None and os.path.exists(resume_state_dump):
        print(f"Resume from step {resume_step}, skipping training data ...")
        for i in range(resume_step):
            next(dataloader_iterator)

    # Train.
    training_step = resume_step # 0 for default
    
    dist.print0("parameters Required grad:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            dist.print0(name, p.shape)
    
    # tensorboard 
    if rank == 0:
        wandb.init(
            entity='tungnd',
            project="dllm",
            name=':'.join(run_dir.split('/')[-2:]),
            dir=run_dir,
            config=opts,
            # mode='offline'
        )

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
                    batch = next(dataloader_iterator)
                    inputs = batch['inputs']
                    prompts = batch['prompts']
                    inputs_without_final_answer = batch['inputs_without_final_answer']
                    input_ids = tokenizer(inputs, return_tensors='pt', truncation=True, max_length=model.config.max_sequence_length, padding='max_length').input_ids
                    input_ids = input_ids.to(device)
                    tokenized_prompts = tokenizer(prompts, return_tensors='pt', truncation=True, max_length=model.config.max_sequence_length)
                    prompt_lengths = tokenized_prompts.attention_mask.sum(dim=1).to(device)
                    
                    noisy_batch, t, _ = forward_process(input_ids, t=None)
                    
                    # do not mask the prompt
                    token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
                    prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
                    noisy_batch[prompt_mask] = input_ids[prompt_mask]
                    
                    # Calculate the answer length (including the padded <EOS> tokens)
                    prompt_mask = prompt_mask.to(torch.int64)    
                    answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
                    answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
                    
                    masked_indices = (noisy_batch == 126336)
                    logits = model(input_ids=noisy_batch).logits
                    
                    unscaled_token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none')
                    unscaled_ce_loss = torch.sum(unscaled_token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
                    token_loss = unscaled_token_loss / t[masked_indices]
                    ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]

                accelerator.backward(ce_loss)

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
        reduced_unscaled_ce_loss = accelerator.reduce(unscaled_ce_loss.detach(), reduction="mean")
        reduced_ce_loss = accelerator.reduce(ce_loss.detach(), reduction="mean")
        reduced_grad_norm = accelerator.reduce(torch.as_tensor(grad_norm, device=device, dtype=torch.float32), reduction="mean")

        if rank == 0:
            wandb.log(
                {
                    'training/lr': scheduler.get_lr()[0],
                    'training/grad_norm': reduced_grad_norm.item(),
                    'training/ce_loss': reduced_ce_loss.item(),
                    'training/unscaled_ce_loss': reduced_unscaled_ce_loss.item(),
                },
                step=training_step
            )
        current_lr = scheduler.get_lr()[0]
        pbar.set_postfix({
            'loss': f"{reduced_ce_loss.float().item():.4f}",
            'unscaled_loss': f"{reduced_unscaled_ce_loss.float().item():.4f}",
            'lr': f"{current_lr:.6f}",
            'grad': f"{reduced_grad_norm.float().item():.2f}",
        })
        pbar.update(1)

        training_step += 1

        if training_step % val_frequency == 0:
            
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

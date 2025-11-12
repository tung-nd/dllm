import os
import re
import json
import yaml
import click
import torch
import dnnlib
import glob
from datetime import datetime
from torch_utils import distributed as dist

def CommandWithConfigFile(config_file_param_name):

    class CustomCommandClass(click.Command):

        def invoke(self, ctx):
            config_file = ctx.params[config_file_param_name]
            if config_file is not None:
                with open(config_file) as f:
                    config_data = yaml.load(f, Loader=yaml.FullLoader)
                    for key, value in config_data.items():
                        ctx.params[key] = value
            return super(CustomCommandClass, self).invoke(ctx)

    return CustomCommandClass

#----------------------------------------------------------------------------

@click.command(cls=CommandWithConfigFile("config"))
@click.option("--config",    help="config file path",    type=click.Path(exists=True))
@click.option('--fsdp', is_flag=True, help='use fsdp')
def main(**kwargs):
    kwargs.pop("config")
    use_fsdp = kwargs.pop("fsdp", False)
    training_args = dnnlib.EasyDict(kwargs.pop("training_args"))
    opts = dnnlib.EasyDict(kwargs)
    # torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0("Distributed initialized.")

    try:
        training_args.batch_size = opts.data_loader_kwargs.get('batch_size', 1)
    except Exception:
        dist.print0("Batch size not set?")

    # Description string.
    dtype_str = training_args.precision
    desc = f'gpus{dist.get_world_size():d}-batch{training_args.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.pop("desc")}'
        
    date = datetime.now().strftime("%Y-%m-%d")
    training_args.run_dir = os.path.join(training_args.run_dir, training_args.exp_name)
    training_args.training_state_dir = os.path.join(training_args.run_dir, f'STATES-{os.environ.get("SLURM_JOB_ID") or "tmp"}')
    dist.print0(f"Save training_args at {training_args.training_state_dir}")

    # Auto-resume: if experiment directory exists, attempt to resume from the latest checkpoint/state.
    if os.path.isdir(training_args.run_dir):
        def _find_latest_step_and_path(paths, regex):
            latest_step = -1
            latest_path = None
            for p in paths:
                name = os.path.basename(p.rstrip('/'))
                m = re.match(regex, name)
                if m:
                    try:
                        step = int(m.group(1))
                    except Exception:
                        continue
                    if step > latest_step:
                        latest_step = step
                        latest_path = p
            return latest_step, latest_path

        # Use glob to list matching paths directly
        ckpt_paths = glob.glob(os.path.join(training_args.run_dir, 'ckpt-*'))
        state_paths = glob.glob(os.path.join(training_args.run_dir, 'STATES-*', 'training-state-*'))

        latest_ckpt_step, latest_ckpt_dir = _find_latest_step_and_path(ckpt_paths, r'^ckpt-(\d{6})$')
        latest_state_step, latest_state_dump = _find_latest_step_and_path(state_paths, r'^training-state-(\d{6})$')

        # Apply resume settings: prefer state dump; fall back to ckpt
        resume_step = max(latest_ckpt_step, latest_state_step)
        if resume_step > 0:
            training_args.resume_step = resume_step
            if latest_state_step == resume_step and latest_state_dump is not None and os.path.isdir(latest_state_dump):
                training_args.resume_state_dump = latest_state_dump
                dist.print0(f"Resuming from state dump: {latest_state_dump} (step {resume_step})")
            elif latest_ckpt_step == resume_step and latest_ckpt_dir is not None and os.path.isdir(latest_ckpt_dir):
                # Fall back to resuming weights from latest ckpt by overriding pretrained path
                if hasattr(opts, 'network_kwargs') and isinstance(opts.network_kwargs, dict):
                    opts.network_kwargs['pretrained_model_name_or_path'] = latest_ckpt_dir
                dist.print0(f"Resuming model weights from ckpt: {latest_ckpt_dir} (step {resume_step})")
            else:
                dist.print0("No valid resume artifacts found despite existing run directory.")

    # Print options.
    dump_dict = opts.copy()
    dump_dict.update(training_args)
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(dump_dict, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {training_args.run_dir}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {training_args.batch_size}')
    dist.print0(f'Precision:               {training_args.precision}')
    dist.print0()

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(training_args.run_dir, exist_ok=True)
        with open(os.path.join(training_args.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(dump_dict, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(training_args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    dnnlib.util.call_func_by_name(
        **opts,
        **training_args,
    )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------


'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/train.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
import os
import random
import subprocess
import time
import tqdm
import yaml
import argparse
import time
from collections import defaultdict
from contextlib import nullcontext, redirect_stdout
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
import bridgevla.config as exp_cfg_mod
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.mvt.config as mvt_cfg_mod

from bridgevla.mvt.mvt import MVT
from utils.get_dataset import get_dataset
from bridgevla.utils.rvt_utils import (
    get_num_feat,
    RLBENCH_TASKS,
)
from utils.peract_utils_rlbench import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    DATA_FOLDER,
    TRAIN_REPLAY_STORAGE_DIR,
)
from training_utils import (
    build_batch_plan,
    freeze_for_oracle_fusion,
    optimizer_steps_per_epoch,
)

def _scalar_metrics(values):
    metrics = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().item()
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def train(
    agent,
    dataset,
    training_iterations,
    epoch,
    rank=0,
    writer=None,
    wandb_run=None,
    loss_print_interval=10,
):
    agent.train()
    log = defaultdict(list)

    data_iter = iter(dataset)
    iter_command = range(training_iterations)

    progress = tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    )
    for iteration in progress:

        raw_batch = next(data_iter)
        dist.barrier()
        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if type(v) == torch.Tensor
        }
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = raw_batch["lang_goal"]
        update_args={
                "replay_sample": batch,
                "backprop": True,
                "reset_log": (iteration == 0),
            }
        out=agent.update(**update_args)
        dist.barrier()
        if rank == 0:
            step=epoch*training_iterations+iteration
            metrics = _scalar_metrics(out)
            if writer is not None:
                for key, value in metrics.items():
                    writer.add_scalar(f'train/{key}', value, step)
                writer.add_scalar(
                    'train/learning_rate',
                    agent._optimizer.param_groups[0]['lr'],
                    step,
                )
            if wandb_run is not None:
                wandb_metrics = {
                    f'train/{key}': value
                    for key, value in metrics.items()
                }
                wandb_metrics['train/learning_rate'] = (
                    agent._optimizer.param_groups[0]['lr']
                )
                wandb_run.log(wandb_metrics, step=step)

            visible_losses = {
                key: f'{metrics[key]:.4f}'
                for key in (
                    'total_loss',
                    'trans_loss',
                    'trans_loss_raw',
                    'rot_loss_x',
                    'rot_loss_y',
                    'rot_loss_z',
                    'grip_loss',
                    'collision_loss',
                )
                if key in metrics
            }
            progress.set_postfix(visible_losses, refresh=False)
            if (
                loss_print_interval > 0
                and (
                    iteration % loss_print_interval == 0
                    or iteration == training_iterations - 1
                )
            ):
                loss_text = ' '.join(
                    f'{key}={value:.6f}'
                    for key, value in metrics.items()
                    if 'loss' in key
                )
                tqdm.tqdm.write(
                    f'[train] epoch={epoch} iter={iteration} '
                    f'step={step} {loss_text}'
                )
    return log


def train_with_accumulation(
    agent,
    dataset,
    optimizer_steps,
    epoch,
    gradient_accumulation_steps,
    global_step_offset=0,
    rank=0,
    writer=None,
    wandb_run=None,
    loss_print_interval=10,
):
    agent.train()
    data_iter = iter(dataset)
    progress = tqdm.tqdm(
        range(optimizer_steps), disable=(rank != 0), position=0, leave=True
    )

    for optimizer_step_index in progress:
        agent.zero_grad()
        metric_totals = defaultdict(float)
        metric_counts = defaultdict(int)

        for micro_step in range(gradient_accumulation_steps):
            raw_batch = next(data_iter)
            batch = {
                key: value.to(agent._device)
                for key, value in raw_batch.items()
                if type(value) == torch.Tensor
            }
            batch['tasks'] = raw_batch['tasks']
            batch['lang_goal'] = raw_batch['lang_goal']

            synchronize = micro_step == gradient_accumulation_steps - 1
            if isinstance(agent._network, DDP) and not synchronize:
                sync_context = agent._network.no_sync()
            else:
                sync_context = nullcontext()

            with sync_context:
                out = agent.update(
                    replay_sample=batch,
                    backprop=True,
                    reset_log=(optimizer_step_index == 0 and micro_step == 0),
                    loss_scale=1.0 / gradient_accumulation_steps,
                    reset_gradients=False,
                    step_optimizer=False,
                )

            for key, value in _scalar_metrics(out).items():
                metric_totals[key] += value
                metric_counts[key] += 1

        agent.optimizer_step()
        metrics = {
            key: total / metric_counts[key]
            for key, total in metric_totals.items()
        }
        global_step = global_step_offset + optimizer_step_index

        if rank == 0:
            if writer is not None:
                for key, value in metrics.items():
                    writer.add_scalar(f'train/{key}', value, global_step)
                writer.add_scalar(
                    'train/learning_rate',
                    agent._optimizer.param_groups[0]['lr'],
                    global_step,
                )
                writer.add_scalar(
                    'system/max_memory_allocated_gib',
                    torch.cuda.max_memory_allocated(agent._device) / (1024 ** 3),
                    global_step,
                )
            if wandb_run is not None:
                wandb_metrics = {
                    f'train/{key}': value for key, value in metrics.items()
                }
                wandb_metrics['train/learning_rate'] = (
                    agent._optimizer.param_groups[0]['lr']
                )
                wandb_metrics['system/max_memory_allocated_gib'] = (
                    torch.cuda.max_memory_allocated(agent._device) / (1024 ** 3)
                )
                wandb_run.log(wandb_metrics, step=global_step)

            visible_losses = {
                key: f'{metrics[key]:.4f}'
                for key in (
                    'total_loss',
                    'trans_loss',
                    'trans_loss_raw',
                    'rot_loss_x',
                    'rot_loss_y',
                    'rot_loss_z',
                    'grip_loss',
                    'collision_loss',
                )
                if key in metrics
            }
            progress.set_postfix(visible_losses, refresh=False)
            if (
                loss_print_interval > 0
                and (
                    global_step % loss_print_interval == 0
                    or optimizer_step_index == optimizer_steps - 1
                )
            ):
                loss_text = ' '.join(
                    f'{key}={value:.6f}'
                    for key, value in metrics.items()
                    if 'loss' in key
                )
                tqdm.tqdm.write(
                    f'[train] epoch={epoch} optimizer_step={global_step} '
                    f'{loss_text}'
                )

    return global_step_offset + optimizer_steps

def save_agent(
    agent, path, epoch, optimizer_step=None, include_optimizer=False
):
    model = agent._network

    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
    }
    if optimizer_step is not None:
        checkpoint['optimizer_step'] = int(optimizer_step)
    if include_optimizer:
        checkpoint["optimizer_state"] = agent._optimizer.state_dict()

    # Keep the previous checkpoint intact if the process is interrupted while
    # writing the new one.
    tmp_path = f"{path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_training_checkpoint(agent, path):
    checkpoint = torch.load(path, map_location="cpu")
    model = agent._network

    if isinstance(model, DDP):
        model = model.module

    model.load_state_dict(checkpoint["model_state"])

    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state is None:
        print(
            "WARNING: the checkpoint has no optimizer state; "
            "model weights were restored but the optimizer starts fresh.",
            flush=True,
        )
    else:
        agent._optimizer.load_state_dict(optimizer_state)

    checkpoint_epoch = int(checkpoint["epoch"])
    print(f"Resumed training from {path} (completed epoch {checkpoint_epoch}).", flush=True)
    optimizer_step = checkpoint.get('optimizer_step')
    return checkpoint_epoch + 1, optimizer_step


def load_initial_model_checkpoint(agent, path):
    """Initialize O2 from a baseline model without restoring its optimizer."""
    checkpoint = torch.load(path, map_location='cpu')
    model = agent._network
    if isinstance(model, DDP):
        model = model.module
    incompatible = model.load_state_dict(
        checkpoint['model_state'], strict=False
    )
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [
        key for key in incompatible.missing_keys
        if 'oracle_prior_fusion' not in key
        and not key.endswith('language_model.lm_head.weight')
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            'Baseline checkpoint is incompatible with O2 initialization: '
            f'missing={disallowed_missing}, unexpected={unexpected}'
        )
    print(
        f'Initialized model weights from baseline checkpoint: {path}',
        flush=True,
    )



def get_tasks(exp_cfg):
    parsed_tasks = exp_cfg.tasks.split(",")
    if parsed_tasks[0] == "all":
        tasks = RLBENCH_TASKS
    else:
        tasks = parsed_tasks
    return tasks



def get_time():
    import datetime
    now = datetime.datetime.now()
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    #  'MM-DD-HH-MM'
    folder_name = f"{month:02d}_{day:02d}_{hour:02d}_{minute:02d}"
    return folder_name


def get_logdir(cmd_args, exp_cfg,dist):
    if cmd_args.resume_checkpoint:
        log_dir = os.path.dirname(os.path.abspath(cmd_args.resume_checkpoint))
        if dist.get_rank() == 0:
            os.makedirs(log_dir, exist_ok=True)
        return log_dir

    log_dir = os.path.join(cmd_args.log_dir,"train" ,exp_cfg.exp_id,cmd_args.exp_note)
    if cmd_args.debug==True:
        log_dir = os.path.join(log_dir,"debug")

    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
    trial_time=get_time()
    log_dir = os.path.join(log_dir,f"{trial_time}")
    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/mvt_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(mvt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)



def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            # os.environ["MASTER_PORT"] = "29566"
            os.environ["MASTER_PORT"] = str(29567 + num_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        if os.getenv('DEBUG', 'false').lower() == 'true':
            print("Can not find RANK and WORLD_SIZE, Debug Mode")
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "9001"
            os.environ["LOCAL_RANK"] = "0"
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )



def set_training_seed(seed, rank):
    rank_seed = int(seed) + int(rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)


def freeze_backbone_modules(
    backbone, freeze_vision_tower, freeze_language_model=False,
    freeze_gemma_prefix_layers=0, freeze_multimodal_projector=False,
):
    freeze_names = ['lm_head', 'embed_tokens']
    if freeze_vision_tower:
        freeze_names.append('vision_tower')
    if freeze_language_model:
        freeze_names.append('language_model')

    frozen = 0
    for name, parameter in backbone.named_parameters():
        if any(freeze_name in name for freeze_name in freeze_names):
            if parameter.requires_grad:
                parameter.requires_grad = False
                frozen += parameter.numel()

    gemma_layers = backbone.mvt1.model.language_model.model.layers
    prefix_layers = int(freeze_gemma_prefix_layers)
    if not 0 <= prefix_layers <= len(gemma_layers):
        raise ValueError(
            'freeze_gemma_prefix_layers must be between 0 and '
            f'{len(gemma_layers)}, got {prefix_layers}'
        )
    for layer in gemma_layers[:prefix_layers]:
        for parameter in layer.parameters():
            if parameter.requires_grad:
                parameter.requires_grad = False
                frozen += parameter.numel()
    if freeze_multimodal_projector:
        for parameter in backbone.mvt1.model.multi_modal_projector.parameters():
            if parameter.requires_grad:
                parameter.requires_grad = False
                frozen += parameter.numel()
    return frozen


def experiment(cmd_args):
    if cmd_args.loss_print_interval < 0:
        raise ValueError('--loss_print_interval must be >= 0')
    if cmd_args.tensorboard_flush_secs <= 0:
        raise ValueError('--tensorboard_flush_secs must be > 0')
    if cmd_args.init_checkpoint and cmd_args.resume_checkpoint:
        raise ValueError(
            '--init_checkpoint and --resume_checkpoint are mutually exclusive.'
        )
    if (
        cmd_args.train_oracle_fusion_only
        and not (cmd_args.init_checkpoint or cmd_args.resume_checkpoint)
    ):
        raise ValueError(
            'Fusion-only training requires --init_checkpoint with a trained '
            'baseline, or --resume_checkpoint with an O2 checkpoint.'
        )

    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = f"cuda:{local_rank}"
    torch.cuda.set_device(device_id)
    exp_cfg = exp_cfg_mod.get_cfg_defaults()

    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

    ddp = int(os.environ['WORLD_SIZE']) > 1
    print(f"Total devices: {dist.get_world_size()}")
    if ddp:
        print(f"Running DDP on rank {dist.get_rank()}.")

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id
    
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.exp_id += f"_{cmd_args.exp_cfg_opts}"
    if cmd_args.mvt_cfg_opts != "":
        exp_cfg.exp_id += f"_{cmd_args.mvt_cfg_opts}"
    if (
        exp_cfg.rvt.oracle_prior_mode != 'none'
        and not exp_cfg.use_oracle_objects
    ):
        raise ValueError(
            'O2 prior requires use_oracle_objects=True and an augmented replay.'
        )
    reduced_hardware_mode = exp_cfg.global_batch_size > 0
    if reduced_hardware_mode and exp_cfg.checkpoint_every_epochs <= 0:
        raise ValueError('checkpoint_every_epochs must be > 0')
    exp_cfg.freeze()
    if reduced_hardware_mode:
        set_training_seed(exp_cfg.seed, dist.get_rank())

    BATCH_SIZE_TRAIN = exp_cfg.bs
    batch_plan = build_batch_plan(
        per_device_batch_size=exp_cfg.bs,
        world_size=dist.get_world_size(),
        target_global_batch_size=exp_cfg.global_batch_size,
    )
    OPTIMIZER_STEPS_PER_EPOCH = optimizer_steps_per_epoch(
        exp_cfg.train_iter, batch_plan.target_global_batch_size
    )
    if local_rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
        print(f"BATCH_SIZE_TRAIN={BATCH_SIZE_TRAIN}")
        print(
            "Batch plan: "
            f"micro_global={batch_plan.micro_global_batch_size}, "
            f"target_global={batch_plan.target_global_batch_size}, "
            "gradient_accumulation_steps="
            f"{batch_plan.gradient_accumulation_steps}, "
            f"optimizer_steps_per_epoch={OPTIMIZER_STEPS_PER_EPOCH}"
        )

    NUM_TRAIN = 100

    if exp_cfg.epochs!=cmd_args.epochs:
        print(f"cmd args epochs != exp cfg epochs You are using {cmd_args.epochs}")
    EPOCHS = cmd_args.epochs

    data_folder=DATA_FOLDER        
    log_dir = get_logdir(cmd_args, exp_cfg,dist)
    tasks = get_tasks(exp_cfg)
    print("Training on {} tasks: {}".format(len(tasks), tasks))
    t_start = time.time()
    get_dataset_func = lambda: get_dataset(
        tasks,
        BATCH_SIZE_TRAIN,
        None,
        TRAIN_REPLAY_STORAGE_DIR,
        None,
        data_folder,
        NUM_TRAIN,
        None,
        cmd_args.refresh_replay,
        device_id,
        num_workers=exp_cfg.num_workers,
        only_train=True,
        sample_distribution_mode=exp_cfg.sample_distribution_mode,
        use_oracle_objects=exp_cfg.use_oracle_objects,
        oracle_max_objects=exp_cfg.oracle_max_objects,
        oracle_num_points=exp_cfg.oracle_num_points,
    )
    train_dataset, _ = get_dataset_func()
    t_end = time.time()
    if local_rank== 0:
        print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if cmd_args.mvt_cfg_path != "":
        mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
    if cmd_args.mvt_cfg_opts != "":
        mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

    mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
    mvt_cfg.freeze()

    # for maintaining backward compatibility
    assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes, print(
        mvt_cfg.num_rot, exp_cfg.peract.num_rotation_classes
    )

    backbone = MVT(
        renderer_device=device_id,
        load_pretrain=cmd_args.load_pretrain,
        pretrain_path=cmd_args.pretrain_path,
        flash_attention_2=exp_cfg.flash_attention_2,
        oracle_prior_fusion=(
            exp_cfg.rvt.oracle_prior_mode == 'o2_gt_instance'
        ),
        oracle_prior_hidden_channels=exp_cfg.oracle_prior_hidden_channels,
        **mvt_cfg,
    )
    if cmd_args.train_oracle_fusion_only:
        if exp_cfg.rvt.oracle_prior_mode != 'o2_gt_instance':
            raise ValueError(
                '--train_oracle_fusion_only requires O2 mode.'
            )
        fusion_params = freeze_for_oracle_fusion(backbone)
        print(
            'Freeze original BridgeVLA; train Oracle fusion only: '
            f'{fusion_params / 1e6:.3f} million parameters'
        )
    if exp_cfg.efficient_paligemma_forward:
        backbone.mvt1.enable_efficient_paligemma_forward()
    if exp_cfg.gradient_checkpointing:
        backbone.mvt1.enable_gradient_checkpointing()
    if reduced_hardware_mode:
        frozen_params = freeze_backbone_modules(
            backbone, cmd_args.freeze_vision_tower,
            cmd_args.freeze_language_model,
            exp_cfg.freeze_gemma_prefix_layers,
            exp_cfg.freeze_multimodal_projector,
        )
        if exp_cfg.freeze_gemma_prefix_layers:
            print(
                'Freeze first '
                f'{exp_cfg.freeze_gemma_prefix_layers} Gemma layers'
            )
        if exp_cfg.freeze_multimodal_projector:
            print('Freeze PaliGemma multimodal projector weights')
        if cmd_args.freeze_language_model:
            print('Freeze Gemma language model')
        if cmd_args.freeze_vision_tower:
            print("Freeze vision tower")
        print(f"Frozen parameters: {frozen_params / 1e9:.2f} billion")
    backbone=backbone.to(local_rank)
    # if ddp:
    backbone = DDP(backbone, device_ids=[local_rank],find_unused_parameters=True)

    agent = bridgevla_agent.RVTAgent(
        network=backbone,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{log_dir}/test_run/",
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    if not reduced_hardware_mode:
        freeze_names = ["lm_head", "embed_tokens"]
        if cmd_args.freeze_vision_tower:
            freeze_names.append("vision_tower")
            print("Freeze vision tower")
        if cmd_args.freeze_language_model:
            freeze_names.append('language_model')
            print('Freeze Gemma language model')
        for name, module in agent._network.named_modules():
            for freeze_name in freeze_names:
                if freeze_name in name:
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    total_params = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
    total_params_billion = total_params / 1e9  
    print(f'Total trainable parameters: {total_params_billion:.2f} billion')


    agent.build(training=True, device=device_id)
    if cmd_args.init_checkpoint:
        if not os.path.isfile(cmd_args.init_checkpoint):
            raise FileNotFoundError(
                f'Initial checkpoint does not exist: {cmd_args.init_checkpoint}'
            )
        load_initial_model_checkpoint(agent, cmd_args.init_checkpoint)
    start_epoch = 0
    start_optimizer_step = 0
    if (
        dist.get_rank() == 0
        and reduced_hardware_mode
        and cmd_args.save_initial_checkpoint
        and not cmd_args.resume_checkpoint
    ):
        save_agent(
            agent,
            f'{log_dir}/model_step_0.pth',
            epoch=-1,
            optimizer_step=0,
            include_optimizer=cmd_args.save_optimizer_state,
        )
    dist.barrier()
    if cmd_args.resume_checkpoint:
        if not os.path.isfile(cmd_args.resume_checkpoint):
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {cmd_args.resume_checkpoint}"
            )
        start_epoch, checkpoint_step = load_training_checkpoint(
            agent, cmd_args.resume_checkpoint
        )
        start_optimizer_step = (
            checkpoint_step
            if checkpoint_step is not None
            else start_epoch * OPTIMIZER_STEPS_PER_EPOCH
        )
    end_epoch = EPOCHS
    target_optimizer_steps = (
        exp_cfg.max_optimizer_steps
        if exp_cfg.max_optimizer_steps > 0
        else EPOCHS * OPTIMIZER_STEPS_PER_EPOCH
    )
    if start_optimizer_step >= target_optimizer_steps:
        raise ValueError(
            "Checkpoint has already reached the configured optimizer-step target: "
            f"{start_optimizer_step} >= {target_optimizer_steps}"
        )

    if dist.get_rank() == 0:
        ## logging unchanged values to reproduce the same setting
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()
    # Logging is rank-0 only. Backends are imported lazily so selecting one
    # backend does not require the other package.
    writer = None
    wandb_run = None
    if dist.get_rank() == 0:
        if cmd_args.log_backend == 'tensorboard':
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    'TensorBoard logging was selected but tensorboard is not '
                    'installed. Run: pip install tensorboard'
                ) from exc
            tensorboard_dir = os.path.join(log_dir, 'tensorboard')
            writer = SummaryWriter(
                log_dir=tensorboard_dir,
                flush_secs=cmd_args.tensorboard_flush_secs,
            )
            print(f'TensorBoard logs: {tensorboard_dir}', flush=True)
        elif cmd_args.log_backend == 'wandb':
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError(
                    'W&B logging was selected but wandb is not installed. '
                    'Run: pip install wandb'
                ) from exc
            wandb_kwargs = {
                'project': cmd_args.wandb_project,
                'name': os.path.basename(log_dir),
                'dir': log_dir,
                'mode': cmd_args.wandb_mode,
                'config': {
                    'exp_cfg': yaml.safe_load(exp_cfg.dump()),
                    'mvt_cfg': yaml.safe_load(mvt_cfg.dump()),
                },
            }
            if cmd_args.wandb_entity:
                wandb_kwargs['entity'] = cmd_args.wandb_entity
            wandb_run = wandb.init(**wandb_kwargs)
            print(
                f'W&B logging: project={cmd_args.wandb_project} '
                f'mode={cmd_args.wandb_mode}',
                flush=True,
            )
        else:
            print('Metric backend disabled; tqdm/text loss remains enabled.')

    print(
        f"Start training at optimizer step {start_optimizer_step}; "
        f"target={target_optimizer_steps}",
        flush=True,
    )
    i = start_epoch
    global_optimizer_step = start_optimizer_step
    while i < end_epoch and global_optimizer_step < target_optimizer_steps:

        print(f"Rank [{dist.get_rank()}], Epoch [{i}]: Training on train dataset")

        steps_this_epoch = min(
            OPTIMIZER_STEPS_PER_EPOCH,
            target_optimizer_steps - global_optimizer_step,
        )
        if reduced_hardware_mode:
            global_optimizer_step = train_with_accumulation(
                agent,
                train_dataset,
                steps_this_epoch,
                epoch=i,
                gradient_accumulation_steps=(
                    batch_plan.gradient_accumulation_steps
                ),
                global_step_offset=global_optimizer_step,
                rank=dist.get_rank(),
                writer=writer,
                wandb_run=wandb_run,
                loss_print_interval=cmd_args.loss_print_interval,
            )
        else:
            train(
                agent,
                train_dataset,
                steps_this_epoch,
                epoch=i,
                rank=dist.get_rank(),
                writer=writer,
                wandb_run=wandb_run,
                loss_print_interval=cmd_args.loss_print_interval,
            )
            global_optimizer_step += steps_this_epoch

        if reduced_hardware_mode:
            should_save = (
                (i + 1) % exp_cfg.checkpoint_every_epochs == 0
                or i == end_epoch - 1
                or global_optimizer_step == target_optimizer_steps
            )
        else:
            should_save = i % 10 == 0 or i == end_epoch - 1
        if dist.get_rank() == 0 and should_save:
            save_agent(
                agent,
                f"{log_dir}/model_{i}.pth",
                i,
                optimizer_step=(
                    global_optimizer_step if reduced_hardware_mode else None
                ),
                include_optimizer=cmd_args.save_optimizer_state,
            )
            save_agent(
                agent,
                f"{log_dir}/model_last.pth",
                i,
                optimizer_step=(
                    global_optimizer_step if reduced_hardware_mode else None
                ),
                include_optimizer=cmd_args.save_optimizer_state,
            )
        i += 1
        dist.barrier()

    dist.barrier()
    if dist.get_rank() == 0:
        if writer is not None:
            writer.close()
        if wandb_run is not None:
            wandb_run.finish()
        print("[Finish]")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--freeze_language_model', action='store_true',
        help='Freeze Gemma language-model parameters while training other heads.',
    )
    parser.add_argument(
        '--train_oracle_fusion_only', action='store_true',
        help='Freeze original BridgeVLA and train only O2 fusion heads.',
    )
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--mvt_cfg_path", type=str, default="../bridgevla/mvt/configs/rvt2.yaml")
    parser.add_argument("--exp_cfg_path", type=str, default="configs/rlbench_config.yaml")
    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--exp_note", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--freeze_vision_tower", action="store_true")
    parser.add_argument("--load_pretrain", action="store_true")
    parser.add_argument("--pretrain_path", type=str, default=None)
    parser.add_argument(
        '--log_backend',
        choices=('tensorboard', 'wandb', 'none'),
        default='tensorboard',
        help='Metric logging backend. Live tqdm/text loss is always available.',
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='BridgeVLA',
        help='W&B project used when --log_backend wandb.',
    )
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default='',
        help='Optional W&B entity used when --log_backend wandb.',
    )
    parser.add_argument(
        '--wandb_mode',
        choices=('online', 'offline', 'disabled'),
        default='online',
        help='W&B mode used when --log_backend wandb.',
    )
    parser.add_argument(
        '--loss_print_interval',
        type=int,
        default=10,
        help='Print scalar losses every N iterations; 0 disables text output.',
    )
    parser.add_argument(
        '--tensorboard_flush_secs',
        type=int,
        default=10,
        help='How often TensorBoard event data is flushed to disk.',
    )
    parser.add_argument(
        '--init_checkpoint',
        type=str,
        default=None,
        help=(
            'Initialize model weights from a baseline checkpoint without '
            'restoring epoch or optimizer state.'
        ),
    )
    parser.add_argument(
        "--resume",
        "--resume_checkpoint",
        dest="resume_checkpoint",
        type=str,
        default=None,
        help=(
            "Resume from a training checkpoint. --epochs remains the total target "
            "epoch count, not the number of additional epochs."
        ),
    )
    parser.add_argument(
        "--save_optimizer_state",
        action="store_true",
        help=(
            "Include optimizer state for optimizer-continuous training resume. "
            "Disabled by default to keep evaluation/inference checkpoints small."
        ),
    )
    parser.add_argument(
        '--save_initial_checkpoint',
        action='store_true',
        help='Save the initialized RLBench policy before optimizer step 1.',
    )
    cmd_args = parser.parse_args()
    experiment(cmd_args)

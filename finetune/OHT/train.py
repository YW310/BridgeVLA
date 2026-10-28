"""Single-GPU / torchrun training with the existing absolute-pose policy."""
import argparse
import json
import os
import random
from contextlib import nullcontext
import numpy as np
from .config import load
from .data.common import write_json
from .data.dataset import OHTDataset, collate, to_device
from .data.sampling import TaskBalancedSampler
from .model import build_agent, load_weights, model_state, resume_fingerprint
from .action_modes import resolve_action
from .training_log import TensorBoardLogger, summary_writer_class
from .training_plan import resolve_training_plan, epoch_metrics
from .training_visualization import visualization_settings, capture_due, record_visualization
from .run_directory import prepare_run_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--replay", required=True)
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--output", help="Exact run directory (legacy behavior)")
    location.add_argument("--log-dir", "--log_dir", help="Timestamped run root (default: exp/OHT)")
    parser.add_argument("--exp-id", "--exp_id", help="Experiment name; default: config exp_id or YAML filename")
    parser.add_argument("--exp-note", "--exp_note", default="", help="Optional experiment subdirectory")
    parser.add_argument("--role-cache")
    parser.add_argument("--pretrain-path")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--init-checkpoint")
    group.add_argument("--resume")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--log-backend", choices=("tensorboard", "none"), default="tensorboard")
    parser.add_argument("--tensorboard-flush-secs", type=int, default=10)
    parser.add_argument("--epochs", type=int, help="Override total epochs in an epoch-budget config")
    args = parser.parse_args(argv)
    if args.tensorboard_flush_secs < 1:
        parser.error("--tensorboard-flush-secs must be a positive integer")
    config = load(args.config)
    visualization = visualization_settings(config, None if args.validate_only else args.log_backend)
    if args.epochs is not None:
        if args.epochs < 1 or config.get("epochs") is None:
            parser.error("--epochs requires an epoch-budget config and a positive total epoch count")
        config["epochs"] = args.epochs
    chunk_enabled = config.get('chunk_policy', {}).get('enabled', False)
    if chunk_enabled:
        from .data.chunk_dataset import OHTChunkDataset
        from .chunk_config import chunk_settings
        if args.role_cache:
            parser.error('Chunk baseline does not use role caches')
        dataset = OHTChunkDataset(args.replay, 'train')
        settings = chunk_settings(config, dataset.contract)
        if (settings['stage'] == 'frozen_key' and not args.validate_only
                and not (args.init_checkpoint or args.resume)):
            parser.error('frozen_key chunk training requires --init-checkpoint with a trained EEF model, '
                         'or --resume; use chunk_policy.stage=joint to train without an EEF checkpoint')
    else:
        dataset = OHTDataset(args.replay, "train", config["mode"], args.role_cache, config["point_count"], resolve_action(config))
    rank, local_rank = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    plan = resolve_training_plan(config, len(dataset), world)
    target_steps, accumulation = plan["optimizer_steps"], plan["accumulation_steps"]
    if args.validate_only:
        if chunk_enabled:
            from .chunk_normalization import resolve_training_normalization
            resolve_training_normalization(config, dataset)
        print(json.dumps(dict(valid=True, mode=config["mode"], samples=dataset.validate_all(), training_plan=plan)))
        return 0
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    if args.log_backend == "tensorboard":
        # Check on every rank before expensive model construction or collectives.
        summary_writer_class()
    if not torch.cuda.is_available():
        raise RuntimeError("Full BridgeVLA training requires CUDA and the point-renderer extension; use --validate-only on CPU")
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    seed = config.get("seed", 0) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = prepare_run_directory(args, config, rank=rank, world_size=world,
                                   distributed=dist if world > 1 else None)
    if rank == 0:
        print(f"Run directory: {output}", flush=True)
        if chunk_enabled:
            source = ('resume' if args.resume else 'checkpoint' if args.init_checkpoint
                      else 'bridgevla_pretrain' if args.pretrain_path else 'paligemma_base')
            print(f"Chunk initialization: stage={settings['stage']} source={source}", flush=True)
            print(json.dumps(dict(chunk_decoder=settings['decoder_type'],
                                  spatial_conditioning=settings['spatial_conditioning'],
                                  goal_conditioning=settings['goal_conditioning'],
                                  displacement_weight=settings['displacement_weight'],
                                  flow_steps=settings['flow_steps'] if settings['decoder_type'] == 'flow_matching' else None)), flush=True)
    if world > 1:
        dist.barrier()
    if chunk_enabled:
        from .chunk_normalization import resolve_training_normalization
        resolve_training_normalization(config, dataset, distributed=dist if world > 1 else None, rank=rank)
        settings = chunk_settings(config, dataset.contract)
        if rank == 0:
            print(json.dumps(dict(chunk_scales={name: settings[name] for name in
                ('action_position_scale', 'geometry_scale', 'position_loss_scale')},
                flow_time_sampling=settings['flow_time_sampling'], adaln=settings['adaln'],
                temporal_kernel_size=settings['temporal_kernel_size'],
                second_difference_weight=settings['second_difference_weight'])), flush=True)
            if config.get('chunk_normalization'):
                write_json(output / 'chunk_normalization.json', config['chunk_normalization'])
    remaining_draws = target_steps * accumulation * config["batch_size"]
    sampler = TaskBalancedSampler(dataset.samples, remaining_draws, seed)
    loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler,
                        num_workers=config.get("num_workers", 0), collate_fn=collate,
                        pin_memory=True)
    agent = build_agent(config, dataset.contract, device, args.pretrain_path, distributed=world > 1)
    cache_hash = dataset.role_cache.manifest["sha256"] if dataset.role_cache else None
    fingerprint = resume_fingerprint(config, dataset.contract, cache_hash, world)
    step = 0
    if args.init_checkpoint:
        load_weights(agent, args.init_checkpoint, initialize=True)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("oht_resume_fingerprint") != fingerprint:
            raise ValueError("Resume configuration/replay/role cache/world-size mismatch")
        load_weights(agent, checkpoint)
        agent._optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in agent._optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        step = int(checkpoint["optimizer_step"])
    if step >= target_steps:
        raise ValueError("Checkpoint already reached the requested training budget; increase optimizer_steps or epochs/max_optimizer_steps")
    sampler.start = step * accumulation * config["batch_size"]
    if rank == 0:
        write_json(output / "config.json", config)
        write_json(output / "contract.json", dataset.contract)
        write_json(output / "training_plan.json", plan)
        print(json.dumps(dict(mode=config["mode"], samples=len(dataset),
                              resume_step=step, **plan)), flush=True)
        print(json.dumps(dict(freeze_vision_tower=config.get("freeze_vision_tower", False),
                              freeze_language_model=config.get("freeze_language_model", False),
                              freeze_gemma_prefix_layers=config.get("freeze_gemma_prefix_layers", 0),
                              freeze_multimodal_projector=config.get("freeze_multimodal_projector", False),
                              trainable_parameters=sum(p.numel() for p in agent._net_mod.parameters() if p.requires_grad),
                              frozen_parameters=sum(p.numel() for p in agent._net_mod.parameters() if not p.requires_grad))), flush=True)
        if chunk_enabled:
            write_json(output / 'trainable_parameters.json',
                       [name for name, parameter in agent._net_mod.named_parameters() if parameter.requires_grad])
    try:
        with TensorBoardLogger(output / "tensorboard", rank=rank, backend=args.log_backend,
                               flush_secs=args.tensorboard_flush_secs,
                               resume_step=step if args.resume else None) as logger:
            if rank == 0 and args.log_backend == "tensorboard":
                print(f"TensorBoard logs: {output / 'tensorboard'}", flush=True)
            iterator = iter(loader)
            while step < target_steps:
                losses = {}
                capture = capture_due(visualization, step + 1, rank)
                image_payload = None
                for micro in range(accumulation):
                    batch = to_device(next(iterator), device)
                    synchronize = micro == accumulation - 1
                    sync_context = (agent._network.no_sync() if world > 1 and not synchronize else nullcontext())
                    with sync_context:
                        image_options = {"return_visualization": True} if capture and synchronize else {}
                        result = agent.update(batch, loss_scale=1 / accumulation,
                                              reset_gradients=micro == 0,
                                              step_optimizer=synchronize, **image_options)
                    if capture and synchronize:
                        image_payload = result.get("train_visualization")
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            losses[key] = losses.get(key, 0) + value / accumulation
                step += 1
                if rank == 0:
                    progress = epoch_metrics(step, plan)
                    row = dict(step=step, **progress, **losses)
                    with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    logger.log(step, dict(losses, **progress), [group["lr"] for group in agent._optimizer.param_groups])
                    if capture:
                        record_visualization(image_payload, step=step, output=output,
                                             settings=visualization, batch=batch, logger=logger)
                    epoch_finished = bool(plan["steps_per_epoch"] and step % plan["steps_per_epoch"] == 0)
                    if step == 1 or step % 10 == 0 or epoch_finished or step == target_steps:
                        print(json.dumps(row), flush=True)
                    if step % plan["checkpoint_interval"] == 0 or step == target_steps:
                        checkpoint = dict(model_state=model_state(agent), optimizer_state=agent._optimizer.state_dict(),
                                          optimizer_step=step, config=config, oht_contract=dataset.contract,
                                          oht_resume_fingerprint=fingerprint, role_cache_sha256=cache_hash,
                                          role_cache_provenance=dataset.role_cache.manifest["provenance"] if dataset.role_cache else None,
                                          resume_rng_restored=False)
                        checkpoint["oht_training_plan"] = plan
                        if plan["steps_per_epoch"] is not None:
                            checkpoint.update(epochs_completed=step // plan["steps_per_epoch"],
                                              epoch_step=step % plan["steps_per_epoch"])
                        temporary = output / "model_last.pth.tmp"
                        torch.save(checkpoint, temporary)
                        temporary.replace(output / "model_last.pth")
                        logger.flush()
                        if epoch_finished:
                            torch.save(checkpoint, output / f"model_epoch_{checkpoint['epochs_completed']}.pth")
                        if step == target_steps:
                            torch.save(checkpoint, output / f"model_step_{step}.pth")
    finally:
        if world > 1:
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

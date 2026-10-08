"""Single-GPU / torchrun training with the existing absolute-pose policy."""
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
from .config import load
from .data.common import write_json
from .data.dataset import OHTDataset, collate, to_device
from .data.sampling import TaskBalancedSampler
from .model import build_agent, load_weights, model_state, resume_fingerprint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--role-cache")
    parser.add_argument("--pretrain-path")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--init-checkpoint")
    group.add_argument("--resume")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    config = load(args.config)
    dataset = OHTDataset(args.replay, "train", config["mode"], args.role_cache, config["point_count"])
    if args.validate_only:
        print(json.dumps(dict(valid=True, mode=config["mode"], samples=dataset.validate_all())))
        return 0
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    if not torch.cuda.is_available():
        raise RuntimeError("Full BridgeVLA training requires CUDA and the point-renderer extension; use --validate-only on CPU")
    rank, local_rank = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Use a new training output directory: {output}")
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    seed = config.get("seed", 0) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    remaining_draws = config["optimizer_steps"] * config["accumulation_steps"] * config["batch_size"]
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
    if step >= config["optimizer_steps"]:
        raise ValueError("Checkpoint already reached the requested optimizer_steps")
    sampler.start = step * config["accumulation_steps"] * config["batch_size"]
    if rank == 0:
        write_json(output / "config.json", config)
        write_json(output / "contract.json", dataset.contract)
        print(json.dumps(dict(mode=config["mode"], samples=len(dataset),
                              effective_batch=config["batch_size"] * config["accumulation_steps"] * world,
                              resume_step=step)), flush=True)
    iterator = iter(loader)
    while step < config["optimizer_steps"]:
        losses = {}
        for micro in range(config["accumulation_steps"]):
            batch = to_device(next(iterator), device)
            result = agent.update(batch, loss_scale=1 / config["accumulation_steps"],
                                  reset_gradients=micro == 0,
                                  step_optimizer=micro == config["accumulation_steps"] - 1)
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    losses[key] = losses.get(key, 0) + value / config["accumulation_steps"]
        step += 1
        if rank == 0:
            row = dict(step=step, **losses)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if step == 1 or step % 10 == 0:
                print(json.dumps(row), flush=True)
            if step % config["checkpoint_interval"] == 0 or step == config["optimizer_steps"]:
                checkpoint = dict(model_state=model_state(agent), optimizer_state=agent._optimizer.state_dict(),
                                  optimizer_step=step, config=config, oht_contract=dataset.contract,
                                  oht_resume_fingerprint=fingerprint, role_cache_sha256=cache_hash,
                                  role_cache_provenance=dataset.role_cache.manifest["provenance"] if dataset.role_cache else None,
                                  resume_rng_restored=False)
                temporary = output / "model_last.pth.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(output / "model_last.pth")
                if step == config["optimizer_steps"]:
                    torch.save(checkpoint, output / f"model_step_{step}.pth")
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Checkpoint-bound inference; online external predictor must match training."""
from ..data.common import SCHEMA, digest, read_config
from ..data.observation import validate_data_config
from ..model import build_agent, load_weights
from .policy import Policy
from .predicted_wrapper import load_predictor, PredictedObjectWrapper


def load_policy(path, device="cuda:0", pretrain_path=None, predictor=None, provenance=None, *, progress=None):
    import torch
    def report(stage, **details):
        if progress is not None:
            progress.update(stage, **details)

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("BridgeVLA inference requires CUDA and its point-renderer extension")
    report("read_checkpoint", checkpoint=path, device=device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    report("validate_contract")
    config, contract = checkpoint["config"], checkpoint["oht_contract"]
    contents = dict(contract)
    expected = contents.pop("sha256")
    if expected != digest(contents):
        raise ValueError("Checkpoint OHT contract checksum mismatch")
    chunk = config.get('chunk_policy', {}).get('enabled', False)
    if contract.get("schema") != ('oht_bridgevla_v4' if chunk else SCHEMA):
        raise ValueError("Legacy OHT checkpoint data contract; rebuild replay and retrain with corrected source metadata")
    validate_data_config(contract["data_config"], resolved=True)
    wrapper = None
    if config["mode"] == "predicted_external":
        if not predictor or not provenance:
            raise ValueError("External inference requires --predictor and --predictor-provenance")
        declared = read_config(provenance)
        declared["factory"] = predictor
        if declared != checkpoint.get("role_cache_provenance"):
            raise ValueError("Online predictor provenance differs from the training prediction cache")
        report("load_predictor", predictor=predictor)
        wrapper = PredictedObjectWrapper(load_predictor(predictor), contract["data_config"]["cameras"],
                                         config["point_count"])
    elif predictor or provenance:
        raise ValueError("This checkpoint does not use an external predictor")
    report("build_agent")
    agent = build_agent(config, contract, torch.device(device), pretrain_path, training=False)
    report("apply_checkpoint")
    load_weights(agent, checkpoint)
    report("release_checkpoint")
    del checkpoint
    policy = Policy(agent, contract, device, config["mode"], wrapper)
    policy.training_config = config
    report("model_ready")
    return policy

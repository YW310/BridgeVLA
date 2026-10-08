"""Checkpoint-bound inference; online external predictor must match training."""
from ..data.common import digest, read_config
from ..data.observation import validate_data_config
from ..model import build_agent, load_weights
from .policy import Policy
from .predicted_wrapper import load_predictor, PredictedObjectWrapper


def load_policy(path, device="cuda:0", pretrain_path=None, predictor=None, provenance=None):
    import torch
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("BridgeVLA inference requires CUDA and its point-renderer extension")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config, contract = checkpoint["config"], checkpoint["oht_contract"]
    contents = dict(contract)
    expected = contents.pop("sha256")
    if expected != digest(contents):
        raise ValueError("Checkpoint OHT contract checksum mismatch")
    validate_data_config(contract["data_config"])
    wrapper = None
    if config["mode"] == "predicted_external":
        if not predictor or not provenance:
            raise ValueError("External inference requires --predictor and --predictor-provenance")
        declared = read_config(provenance)
        declared["factory"] = predictor
        if declared != checkpoint.get("role_cache_provenance"):
            raise ValueError("Online predictor provenance differs from the training prediction cache")
        wrapper = PredictedObjectWrapper(load_predictor(predictor), contract["data_config"]["cameras"],
                                         config["point_count"])
    elif predictor or provenance:
        raise ValueError("This checkpoint does not use an external predictor")
    agent = build_agent(config, contract, torch.device(device), pretrain_path, training=False)
    load_weights(agent, checkpoint)
    del checkpoint
    return Policy(agent, contract, device, config["mode"], wrapper)

"""Send one real replay observation to /predict; never execute the returned action."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.OHT.data.common import read_jsonl, inside, file_digest
from finetune.OHT.data.replay import load_contract
from finetune.OHT.runtime.inference_api import API_VERSION, OBSERVATION_FORMAT
from finetune.OHT.runtime.transport import pack_observation


def smoke(server, replay, *, split="val", index=0, timeout=60., episode_id=None):
    if index < 0:
        raise ValueError("index must be nonnegative")
    server = server.rstrip("/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(server + path, data=data,
                                         headers={"Content-Type": "application/json"})
        # A control request is never automatically retried after a timeout.
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)

    info = call("/model_info")
    if info["api_version"] != API_VERSION or info["observation_format"] != OBSERVATION_FORMAT:
        raise ValueError("Server does not support the prepared BridgeVLA /predict API")
    replay = Path(replay)
    schema = json.loads((replay / 'contract.json').read_text(encoding='utf-8')).get('schema')
    if schema == 'oht_bridgevla_v4':
        from finetune.OHT.data.chunk_dataset import load_chunk_contract
        contract = load_chunk_contract(replay)
        if 'chunk_spec' not in info:
            raise ValueError('V4 smoke requires a chunk checkpoint/server')
    else:
        contract = load_contract(replay)
    if info["contract_sha256"] != contract["sha256"]:
        raise ValueError("Server checkpoint and smoke replay contracts differ")
    rows = [row for row in read_jsonl(replay / "samples.jsonl") if row["split"] == split]
    row = rows[index]
    path = inside(replay, row["observation"])
    if file_digest(path) != row["observation_sha256"]:
        raise ValueError("Smoke observation checksum mismatch")
    with np.load(path, allow_pickle=False) as source:
        # Only declared policy observations are sent. Labels/teacher caches are never read.
        observation = {name: source[name] for name in info["observation_schema"]}
    episode_id = episode_id or "smoke-" + uuid.uuid4().hex
    identity = dict(api_version=API_VERSION, model_id=info["model_id"],
                    contract_sha256=info["contract_sha256"], episode_id=episode_id)
    reset = call("/reset", dict(**identity, request_id=episode_id + "-reset"))
    if reset.get("episode_id") != episode_id or reset.get("reset") is not True:
        raise ValueError("Unexpected reset response")
    request = dict(**identity, request_id=episode_id + "-0", step=0, timestamp=0.,
                   instruction=row["goal"], observation_format=OBSERVATION_FORMAT,
                   observation=pack_observation(observation))
    if 'chunk_spec' in info:
        request['prediction_schema'] = info['chunk_spec']['schema']
    body_size = len(json.dumps(request, allow_nan=False).encode("utf-8"))
    if body_size > info["max_request_bytes"]:
        raise ValueError("Smoke request exceeds server max_request_bytes")
    response = call("/predict", request)
    if any(response.get(key) != request[key] for key in (
            "api_version", "model_id", "contract_sha256", "request_id", "episode_id", "step", "timestamp")):
        raise ValueError("Stale/mismatched prediction response")
    if response.get("action_spec") != info["action_spec"]:
        raise ValueError("Prediction output description changed")
    if 'chunk_spec' in info:
        from finetune.OHT.runtime.chunk_protocol import validate_prediction
        if response.get('chunk_spec') != info['chunk_spec']:
            raise ValueError('Chunk output description changed')
        validate_prediction(response, info['chunk_spec'])
    return dict(sample_id=row["id"], task=row["task"], instruction=row["goal"], **response)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8010")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.)
    parser.add_argument("--episode-id")
    args = parser.parse_args(argv)
    print(json.dumps(smoke(**vars(args)), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

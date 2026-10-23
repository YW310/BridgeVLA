"""Offline navigation over bounded per-frame HTML previews; no data service."""
import json
import hashlib
import io
from pathlib import Path
from urllib.parse import quote


def save_replay_viewer(path, frames, contract_sha256):
    path = Path(path)
    if not frames:
        raise ValueError("No HTML frame previews available for the replay viewer")
    entries, seen = [], set()
    for frame in frames:
        if frame["sample_id"] in seen or type(frame["frame"]) is not int:
            raise ValueError("Replay viewer requires unique samples with integer frame numbers")
        seen.add(frame["sample_id"])
        source = Path(frame["html"]).resolve()
        relative = source.relative_to(path.parent.resolve())
        if source.suffix != ".html" or not source.is_file():
            raise ValueError("Replay viewer requires existing HTML files within its directory")
        entries.append(dict(sample_id=frame["sample_id"], episode=frame["sample_id"].rsplit("/",1)[0],
                            frame=frame["frame"], target_frame=frame.get("target_frame"),
                            timestamp=frame.get("timestamp"), html=quote(relative.as_posix(),safe="/")))
        if frame.get("image") is not None:
            image = Path(frame["image"]).resolve()
            image_relative = image.relative_to(path.parent.resolve())
            if image.suffix.lower() != ".png" or not image.is_file():
                raise ValueError("Replay viewer requires existing PNG previews within its directory")
            entries[-1]["image"] = quote(image_relative.as_posix(),safe="/")
    payload = dict(schema="oht_replay_viewer_v1",contract_sha256=contract_sha256,frames=entries)
    serialized = json.dumps(payload,ensure_ascii=True,allow_nan=False,separators=(",",":")).replace("<","\\u003c")
    template = Path(__file__).with_name("assets").joinpath("replay-viewer.html").read_text(encoding="utf-8")
    if template.count("__OHT_REPLAY_VIEWER_DATA__") != 1:
        raise ValueError("Invalid replay viewer template")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        stream.write(template.replace("__OHT_REPLAY_VIEWER_DATA__",serialized))
    return path


def export_replay_viewer(replay, output, *, every=1, max_points=40_000, source=None,
                         object_cloud_config=None, object_root=None, object_manifest=None):
    """Export a complete replay to grouped PNG/HTML previews without cache edits."""
    import numpy as np
    from .common import inside, read_jsonl
    from .geometry_diagnostic import _validate_frame
    from .html_preview import html_point_limit, save_point_cloud_html
    from .object_cloud_build import BuildObjectCloudWriter
    from .object_preview import load_source_objects
    from .replay import load_contract
    from .source_config import sample_data_config
    from .visualization import save_preview
    if type(every) is not int or every < 1:
        raise ValueError("Viewer every must be a positive integer")
    html_point_limit(max_points)
    if source is not None and source not in ("xyz","depth"):
        raise ValueError("Viewer source must be xyz or depth")
    if bool(object_root) != bool(object_manifest):
        raise ValueError("Use --object-root and --object-manifest together")
    replay, output = Path(replay), Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new replay viewer directory: {output}")
    is_chunk = json.loads((replay / 'contract.json').read_text(encoding='utf-8')).get('schema') == 'oht_bridgevla_v4'
    if is_chunk:
        from .chunk_dataset import load_chunk_contract
        contract = load_chunk_contract(replay)
    else:
        contract = load_contract(replay)
    options = object_cloud_config if object_cloud_config is not None else contract.get("diagnostic_object_clouds")
    selector = None if options is None else BuildObjectCloudWriter(output,options)
    selected_source = source or (selector.options["source"] if selector else "depth")
    frames, counts = [], {}
    for sample in read_jsonl(replay/"samples.jsonl"):
        episode = sample["id"].rsplit("/",1)[0]
        count = counts.get(episode,0);counts[episode]=count+1
        if count % every:
            continue
        config = sample_data_config(contract,sample)
        content = inside(replay,sample["observation"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != sample["observation_sha256"]:
            raise ValueError(f"Observation changed: {sample['id']}")
        with np.load(io.BytesIO(content),allow_pickle=False) as data:
            observation = {key:data[key] for key in data.files}
        chunk_options = {}
        if is_chunk:
            from .chunk_actions import validate_chunk_targets
            from .chunk_replay import _target_bounds
            content = inside(replay, sample['targets']).read_bytes()
            if hashlib.sha256(content).hexdigest() != sample['targets_sha256']:
                raise ValueError(f"Targets changed: {sample['id']}")
            with np.load(io.BytesIO(content), allow_pickle=False) as data:
                targets = {key: data[key] for key in data.files}
            validate_chunk_targets(targets, contract['chunk'], sample)
            _target_bounds(targets, config['scene_bounds'])
            sample = dict(sample, labels={})
            if bool(targets['key_valid']):
                sample['labels']['gripper_pose'] = targets['key_action'][:7].tolist()
            chunk_options['chunk_targets'] = targets
        _validate_frame(observation,config,sample)
        if object_root:
            sample = dict(sample,diagnostic_objects=load_source_objects(object_root,object_manifest,contract,sample))
        sample = dict(sample,diagnostic_contract_sha256=contract["sha256"])
        boxes = None if selector is None else dict(selector.specification(sample),source=selected_source,
                                                   contract_sha256=contract["sha256"])
        path = inside(output,sample["id"]+"/point-cloud-roi.html")
        image = save_preview(path.parent/"preview.png",observation,config,sample,current_tcp=sample.get("current_tcp"),
                             object_boxes=boxes, **chunk_options)
        save_point_cloud_html(path,observation,config,sample,max_points,source=selected_source,object_boxes=boxes,
                              preview_image=image)
        frames.append(dict(sample_id=sample["id"],frame=sample["frame"],target_frame=sample["target_frame"],
                           timestamp=sample.get("timestamp"),html=path,image=image))
    save_replay_viewer(output/"index.html",frames,contract["sha256"])
    return len(frames)

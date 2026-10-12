"""Optional build-time object diagnostics; independent of policy observations."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .common import inside, write_json
from .geometry import array
from .html_preview import bind_point_cloud_html_contract, save_point_cloud_html
from .object_cloud import export_object_clouds
from .object_preview import object_payload
from .replay_viewer import save_replay_viewer


def object_cloud_build_options(value):
    if not isinstance(value, dict) or set(value) - {"source", "default_size", "objects"}:
        raise ValueError("Object-cloud build config supports source/default_size/objects; "
                         "do not use the per-frame HTML boxes JSON")
    result = deepcopy(value)
    result.setdefault("source", "depth")
    if result["source"] not in ("xyz", "depth"):
        raise ValueError("Object-cloud build source must be xyz or depth")

    def size(value):
        dimensions = array(value, (3,), "object-cloud full XYZ size")
        if np.any(dimensions <= 0):
            raise ValueError("Object-cloud size must be positive world metres")
        return dimensions.tolist()

    result["default_size"] = size(result.get("default_size", [.2, .2, .2]))
    entries = result.setdefault("objects", None)
    if entries is not None:
        if not isinstance(entries, list) or not entries:
            raise ValueError("Object-cloud objects must be null (all objects) or a nonempty list")
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - {"index", "name", "size"}:
                raise ValueError("Each build object supports index/name/size")
            index = entry.get("index")
            if type(index) is not int or index < 0 or index in seen:
                raise ValueError("Build object indices must be nonnegative and unique")
            seen.add(index)
            entry["size"] = size(entry.get("size", result["default_size"]))
            if "name" in entry and (not isinstance(entry["name"], str) or
                                    not entry["name"].strip() or len(entry["name"]) > 128):
                raise ValueError("Object-cloud name must be nonempty text, at most 128 characters")
    return result


class BuildObjectCloudWriter:
    def __init__(self, output, options, *, preview_output=None):
        self.output = Path(output)
        self.preview_output = Path(preview_output) if preview_output is not None else self.output / "preview"
        self.options = object_cloud_build_options(options)
        self.reports = []
        self.html = []
        self.frames = []

    def specification(self, sample):
        entries = self.options["objects"]
        if entries is None:
            entries = [dict(index=obj["index"], size=self.options["default_size"])
                       for obj in object_payload(sample)]
        return dict(schema="oht_object_boxes_v1", sample_id=sample["id"], frame=sample["frame"],
                    observation_sha256=sample["observation_sha256"], contract_sha256=None,
                    source=self.options["source"], objects=entries)

    def write(self, observation, config, sample, *, visualize=False, preview_image=None):
        specification = self.specification(sample)
        relative = "preview/" + sample["id"] + "/object-clouds.json"
        report_path = inside(self.output, relative)
        export_object_clouds(report_path.parent, observation, config, sample, specification, None,
                             dict(metadata_source="replay_build", completion_checked=False,
                                  observation_checksum_recorded=True, observation_checksum_verified=False,
                                  source_parquet_checksum_verified=True), allow_empty=True)
        self.reports.append(report_path)
        if visualize:
            self.write_preview(observation, config, sample, preview_image=preview_image)
        return relative

    def write_preview(self, observation, config, sample, *, preview_image=None):
        """Write HTML after object export and the optional linked PNG exist."""
        path = inside(self.preview_output, sample["id"] + "/point-cloud-roi.html")
        preview_sample = dict(sample, diagnostic_contract_sha256=None,
                              diagnostic_note="随 replay 构建生成，仅诊断；整批完成状态请检查 complete.json。")
        save_point_cloud_html(path, observation, config, preview_sample, source=self.options["source"],
                              object_boxes=self.specification(sample), preview_image=preview_image)
        self.html.append(path)
        self.frames.append(dict(sample_id=sample["id"],frame=sample["frame"],target_frame=sample.get("target_frame"),
                                timestamp=sample.get("timestamp"),html=path,image=preview_image))

    def bind_contract(self, checksum):
        # The final index changes the contract hash. Bind sidecars only after
        # that hash exists; complete.json is written after all bindings succeed.
        for path in self.reports:
            report = json.loads(path.read_text(encoding="utf-8"))
            report["contract_sha256"] = checksum
            write_json(path, report)
        for path in self.html:
            bind_point_cloud_html_contract(path, checksum)
        if self.frames:
            save_replay_viewer(self.preview_output/"index.html",self.frames,checksum)

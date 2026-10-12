"""Approximate box crops preserve source points/RGB and never edit replay."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from finetune.OHT.cli import main
from finetune.OHT.data.common import digest, file_digest, read_jsonl, write_json
from finetune.OHT.data.html_preview import point_cloud_payload, save_point_cloud_html
from finetune.OHT.data.object_cloud import box_mask, export_object_clouds
from finetune.OHT.data.replay import load_contract
from tests.test_oht_html_preview import cloud
from tests.test_oht_migration import replay_fixture


def setup_crop(cloud, source="xyz"):
    observation, config, sample = cloud
    sample["diagnostic_objects"] = dict(frame=sample["frame"], positions=[[.5,.5,.5],[3,3,3]],
                                        quaternions=[[1,0,0,0]]*2)
    sample["observation_sha256"] = "observation-checksum"
    return dict(schema="oht_object_boxes_v1", sample_id=sample["id"], frame=sample["frame"], source=source,
                observation_sha256=sample["observation_sha256"], contract_sha256="contract-checksum",
                objects=[dict(index=0,name="../../wheel",size=[1,1,1]),dict(index=1,name="empty",size=[.2,.2,.2])])


def test_crop_uses_lower_inclusive_upper_exclusive_world_bounds():
    points = np.array([[0,0,0],[1,1,1],[.5,.5,.5],[np.nan,0,0],[-.01,0,0]])
    assert box_mask(points,[0,0,0,1,1,1]).tolist() == [True,False,True,False,False]


def test_export_uses_all_cached_points_not_html_sample_and_preserves_inputs(cloud, tmp_path):
    specification = setup_crop(cloud)
    observation, config, sample = cloud
    before = deepcopy(cloud)
    assert point_cloud_payload(*cloud,max_points=1)["displayed_points"] == 1
    output = tmp_path/"objects"
    report = export_object_clouds(output,*cloud,specification,"contract-checksum",{})
    assert report["approximate"] and not report["display_sampling"]
    assert report["total_source_points"] == 3
    assert [entry["points"] for entry in report["objects"]] == [2,0]
    assert report["objects"][1]["valid"] is False
    with np.load(output/"object_0.npz",allow_pickle=False) as data:
        np.testing.assert_array_equal(data["xyz"], observation["wrist_point_cloud"].reshape(3,-1).T[:2])
        np.testing.assert_array_equal(data["rgb"], observation["wrist_rgb"].reshape(3,-1).T[:2])
        np.testing.assert_array_equal(data["camera_ids"],[0,0])
    text = (output/"object_0.ply").read_text(encoding="ascii")
    assert "element vertex 2\n" in text and "not instance segmentation" in text
    assert len(text.split("end_header\n")[1].splitlines()) == 2
    assert (output/"object_1.ply").read_text(encoding="ascii").endswith("end_header\n")
    assert not (tmp_path/"wheel.npz").exists()  # Display names never become paths.
    for key in observation:
        np.testing.assert_array_equal(observation[key],before[0][key])
    assert config == before[1] and sample == before[2]
    with pytest.raises(FileExistsError):
        export_object_clouds(output,*cloud,specification,"contract-checksum",{})


def test_depth_crop_recovers_old_roi_points_without_changing_cached_xyz(cloud, tmp_path):
    specification = setup_crop(cloud,source="depth")
    observation, config, sample = cloud
    observation["wrist_point_cloud"][:] = np.nan
    observation.update(wrist_depth=np.array([[[1.,1.2,0.,np.nan]]],np.float32),
                       wrist_camera_intrinsics=np.eye(3),wrist_camera_extrinsics=np.eye(4))
    config["depth"] = dict(kind="z",limits=[.001,3])
    sample["diagnostic_objects"]["positions"][0] = [0,0,1]
    specification["objects"] = [dict(index=0,size=[.2,.2,.2])]
    report = export_object_clouds(tmp_path/"depth",*cloud,specification,"contract-checksum",{})
    assert report["objects"][0]["points"] == 1 and report["source"] == "depth"
    with np.load(tmp_path/"depth/object_0.npz") as data:
        np.testing.assert_array_equal(data["xyz"],[[0,0,1]])
    assert np.isnan(observation["wrist_point_cloud"]).all()


def test_overlapping_crops_and_multiple_cameras_keep_provenance(cloud, tmp_path):
    specification = setup_crop(cloud)
    observation, config, sample = cloud
    config["cameras"]["global_left"] = {}
    observation["global_left_point_cloud"] = observation["wrist_point_cloud"].copy()
    observation["global_left_rgb"] = np.full_like(observation["wrist_rgb"],17)
    sample["diagnostic_objects"]["positions"][1] = [.5,.5,.5]
    specification["objects"][1]["size"] = [1,1,1]
    report = export_object_clouds(tmp_path/"overlap",*cloud,specification,"contract-checksum",{})
    assert [entry["points"] for entry in report["objects"]] == [4,4]
    assert report["objects"][0]["per_camera"] == dict(wrist=2,global_left=2)
    with np.load(tmp_path/"overlap/object_0.npz") as data:
        np.testing.assert_array_equal(data["camera_ids"],[0,0,1,1])
        assert (data["rgb"][:2] == 180).all() and (data["rgb"][2:] == 17).all()


@pytest.mark.parametrize("damage", ["schema","sample_id","frame","observation_sha256","contract_sha256",
                                    "source","size","negative","nan","index","duplicate","empty","missing_poses",
                                    "configuration_type","entry_type","frame_type"])
def test_invalid_crop_configuration_creates_no_output(cloud, tmp_path, damage):
    specification = setup_crop(cloud)
    if damage in ("schema","sample_id","observation_sha256","contract_sha256","source"):
        specification[damage] = "wrong"
    elif damage == "frame":
        specification[damage] += 1
    elif damage == "size":
        specification["objects"][0]["size"] = [.2,.2]
    elif damage in ("negative","nan"):
        specification["objects"][0]["size"][0] = -1 if damage == "negative" else float("nan")
    elif damage == "index":
        specification["objects"][0]["index"] = 999
    elif damage == "duplicate":
        specification["objects"][1]["index"] = 0
    elif damage == "empty":
        specification["objects"] = []
    elif damage == "configuration_type":
        specification = []
    elif damage == "entry_type":
        specification["objects"][0] = None
    elif damage == "frame_type":
        specification["frame"] = float(specification["frame"])
    else:
        cloud[2].pop("diagnostic_objects")
    output = tmp_path/"invalid"
    with pytest.raises(ValueError):
        export_object_clouds(output,*cloud,specification,"contract-checksum",{})
    assert not output.exists()


def test_cli_exports_current_object_clouds_and_preserves_replay(replay_fixture, tmp_path):
    f = replay_fixture
    row = read_jsonl(f.replay/"samples.jsonl")[0]
    contract = load_contract(f.replay)
    specification = dict(schema="oht_object_boxes_v1",sample_id=row["id"],frame=row["frame"],source="xyz",
                         objects=[dict(index=0,size=[10,10,10])],contract_sha256=contract["sha256"],
                         observation_sha256=row["observation_sha256"])
    boxes = tmp_path/"boxes.json"
    write_json(boxes,specification)
    inputs = [f.replay/"contract.json",f.replay/"samples.jsonl",f.replay/row["observation"]]
    before = [file_digest(path) for path in inputs]
    output = tmp_path/"export"
    args = ["extract-object-clouds","--replay",str(f.replay),"--sample-id",row["id"],
            "--boxes",str(boxes),"--output",str(output)]
    assert main(args) == 0
    report = json.loads((output/"object-clouds.json").read_text(encoding="utf-8"))
    with np.load(f.replay/row["observation"]) as observation:
        expected = sum(int(np.isfinite(observation[f"{camera}_point_cloud"]).all(axis=0).sum())
                       for camera in f.config["cameras"])
    assert report["objects"][0]["points"] == expected
    assert report["validation"]["observation_checksum_verified"]
    assert [file_digest(path) for path in inputs] == before
    with pytest.raises(FileExistsError):
        main(args)


def test_cli_restores_legacy_objects_and_roi_masked_points_from_depth(replay_fixture, tmp_path):
    f = replay_fixture
    replay = tmp_path/"legacy"
    shutil.copytree(f.replay,replay)
    rows = read_jsonl(replay/"samples.jsonl")
    for row in rows:
        row.pop("diagnostic_objects")
    row = rows[0]
    observation_path = replay/row["observation"]
    with np.load(observation_path,allow_pickle=False) as data:
        observation = {key:data[key] for key in data.files}
    expected = 0
    for camera in f.config["cameras"]:
        observation[f"{camera}_point_cloud"][:] = np.nan
        expected += int(np.isfinite(observation[f"{camera}_depth"]).sum())
    np.savez_compressed(observation_path,**observation)
    row["observation_sha256"] = file_digest(observation_path)
    index = replay/"samples.jsonl"
    index.write_text("".join(json.dumps(entry)+"\n" for entry in rows),encoding="utf-8")
    contract = json.loads((replay/"contract.json").read_text(encoding="utf-8"))
    contract.pop("sha256")
    contract["index_sha256"] = file_digest(index)
    contract["sha256"] = digest(contract)
    write_json(replay/"contract.json",contract)
    write_json(replay/"complete.json",dict(contract_sha256=contract["sha256"],samples=len(rows),
                                         index_sha256=contract["index_sha256"]))
    specification = dict(schema="oht_object_boxes_v1",sample_id=row["id"],frame=row["frame"],source="depth",
                         objects=[dict(index=0,size=[10,10,10])],contract_sha256=contract["sha256"],
                         observation_sha256=row["observation_sha256"])
    boxes = tmp_path/"boxes.json"
    write_json(boxes,specification)
    inputs = [replay/"contract.json",replay/"complete.json",index,observation_path,f.manifest]
    before = [file_digest(path) for path in inputs]
    args = ["extract-object-clouds","--replay",str(replay),"--sample-id",row["id"],"--boxes",str(boxes)]
    with pytest.raises(ValueError,match="available"):
        main([*args,"--output",str(tmp_path/"without-source")])
    assert not (tmp_path/"without-source").exists()
    output = tmp_path/"restored"
    assert main([*args,"--output",str(output),"--object-root",str(f.root),
                 "--object-manifest",str(f.manifest)]) == 0
    report = json.loads((output/"object-clouds.json").read_text(encoding="utf-8"))
    assert expected > 0 and report["objects"][0]["points"] == expected
    assert report["diagnostic_objects"]["frame"] == row["frame"]
    assert report["source"] == "depth" and report["validation"]["observation_checksum_verified"]
    assert [file_digest(path) for path in inputs] == before


def test_browser_previews_selected_box_points_and_exports_reusable_configuration(cloud, tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    executable = os.environ.get("OHT_BROWSER_EXECUTABLE")
    if not executable or not Path(executable).is_file():
        pytest.skip("Set OHT_BROWSER_EXECUTABLE for optional browser smoke test")
    setup_crop(cloud)
    cloud[2]["diagnostic_contract_sha256"] = "contract-checksum"
    path = save_point_cloud_html(tmp_path/"crops.html",*cloud)
    with playwright.sync_playwright() as p:
        context = p.chromium.launch_persistent_context(str(tmp_path/"browser-profile"),headless=True,
            executable_path=executable,offline=True,viewport=dict(width=1000,height=1100),timeout=30_000)
        page = context.pages[0]
        page.set_default_timeout(10_000)
        errors=[]
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.goto(path.resolve().as_uri(),wait_until="domcontentloaded")
        page.wait_for_selector('#oht-roi[data-ready="true"]')
        original_yaml = page.locator("#yaml").input_value()
        page.locator("#only-object-clouds").check()
        page.wait_for_selector('#point-count[data-visible="0"]')
        page.locator("#clear-objects").click()
        page.get_by_role("checkbox",name="显示 object_0",exact=True).check()
        for axis in ("X","Y","Z"):
            page.get_by_role("spinbutton",name=f"对象 0 框 {axis} 全长",exact=True).fill("1")
        page.wait_for_selector('#point-count[data-visible="2"]')
        page.locator("#show-object-boxes").check()
        page.locator("#fit-objects").click()
        page.get_by_role("textbox",name="对象 0 名称",exact=True).fill("wheel")
        size = page.get_by_role("spinbutton",name="对象 0 框 X 全长",exact=True)
        size.fill("0")
        page.wait_for_selector("#download-object-boxes:disabled")
        assert page.locator("#object-crop-error").inner_text()
        size.fill("1")
        page.wait_for_selector("#download-object-boxes:enabled")
        with page.expect_download() as download:
            page.locator("#download-object-boxes").click()
        specification = json.loads(Path(download.value.path()).read_text(encoding="utf-8"))
        assert specification["objects"] == [dict(index=0,name="wheel",size=[1,1,1])]
        assert specification["source"] == "xyz" and specification["frame"] == cloud[2]["frame"]
        report = export_object_clouds(tmp_path/"from-browser",*cloud,specification,"contract-checksum",{})
        assert report["objects"][0]["points"] == 2
        assert page.locator("#yaml").input_value() == original_yaml
        page.locator("#clear-objects").click()
        page.wait_for_selector('#point-count[data-visible="0"]')
        assert page.locator("#download-object-boxes").is_disabled()
        page.locator("#select-all-objects").click()
        page.wait_for_selector('#point-count[data-visible="2"]')
        for width in (1000,320):
            page.set_viewport_size(dict(width=width,height=1100))
            page.wait_for_selector(f'#oht-roi[data-render-width="{width-32}"]')
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(tmp_path/f"object-crops-{width}.png"),full_page=True)
        assert not errors
        context.close()

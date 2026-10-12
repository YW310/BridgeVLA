"""Build-time box crops follow current objects and stay out of policy inputs."""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from finetune.OHT.cli import main
from finetune.OHT.data.common import file_digest, read_jsonl
from finetune.OHT.data.dataset import OHTDataset
from finetune.OHT.data.html_preview import point_cloud_payload
from finetune.OHT.data.object_cloud import export_object_clouds
from finetune.OHT.data.object_cloud_build import BuildObjectCloudWriter, object_cloud_build_options
from finetune.OHT.data.replay import build, load_contract
from finetune.OHT.data.visualization import OBJECT_COLORS
from tests.test_oht_html_preview import cloud, payload_from_html
from tests.test_oht_migration import replay_fixture


def objects(cloud):
    cloud[2]["diagnostic_objects"] = dict(frame=cloud[2]["frame"], positions=[[.5,.5,.5],[3,3,3]],
                                         quaternions=[[1,0,0,0]]*2)
    cloud[2]["observation_sha256"] = "observation-checksum"


@pytest.mark.parametrize("config", [[], {"source":"wrong"}, {"default_size":[1,0,1]},
    {"default_size":[1,float("nan"),1]}, {"objects":[]}, {"objects":[None]},
    {"objects":[{"index":True}]}, {"objects":[{"index":-1}]},
    {"objects":[{"index":0},{"index":0}]}, {"objects":[{"index":0,"name":""}]},
    {"schema":"oht_object_boxes_v1","frame":0}])
def test_bad_build_object_config_fails_before_output(config, tmp_path):
    output = tmp_path/"invalid"
    with pytest.raises(ValueError):
        BuildObjectCloudWriter(output,config)
    assert not output.exists()


def test_build_options_are_normalized_without_mutating_input():
    config = dict(objects=[dict(index=0,size=[1,2,3]),dict(index=1)])
    before = deepcopy(config)
    options = object_cloud_build_options(config)
    assert options["source"] == "depth" and options["default_size"] == [.2,.2,.2]
    assert options["objects"][1]["size"] == [.2,.2,.2]
    assert config == before


@pytest.mark.parametrize("preview_location,visualize_every", [
    ("default",100), ("default",1), ("shared",100), ("external",100),
])
def test_cli_build_generates_every_sample_and_binds_previews_without_changing_inputs(
        replay_fixture,tmp_path,preview_location,visualize_every):
    f = replay_fixture
    data_config, object_config = tmp_path/"dataset.yaml",tmp_path/"objects.yaml"
    data_config.write_text(yaml.safe_dump(f.config),encoding="utf-8")
    options = dict(default_size=[10,10,10],objects=[dict(index=0,name="selected"),dict(index=1,size=[.1,.1,.1])])
    object_config.write_text(yaml.safe_dump(options),encoding="utf-8")
    output = tmp_path/"replay"
    previews = tmp_path/"previews" if preview_location == "external" else output/"preview"
    preview_args = [] if preview_location == "default" else ["--visualize-output-dir",str(previews)]
    assert main(["build","--root",str(f.root),"--manifest",str(f.manifest),"--config",str(data_config),
                 "--output",str(output),"--sample-stride","2","--object-cloud-config",str(object_config),
                 "--visualize-every",str(visualize_every),*preview_args]) == 0
    contract = load_contract(output)
    assert contract["diagnostic_object_clouds"] == object_cloud_build_options(options)
    rows, baseline = read_jsonl(output/"samples.jsonl"),read_jsonl(f.replay/"samples.jsonl")
    assert len(rows) == len(baseline) == f.count
    for row, original in zip(rows,baseline):
        assert {key:value for key,value in row.items() if key != "object_clouds"} == original
        assert file_digest(output/row["observation"]) == file_digest(f.replay/original["observation"])
        path = output/row["object_clouds"]
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["frame"] == row["frame"] and report["sample_id"] == row["id"]
        assert report["contract_sha256"] == contract["sha256"]
        assert report["observation_sha256"] == row["observation_sha256"]
        assert report["diagnostic_objects"] == row["diagnostic_objects"]
        assert report["objects"][0]["points"] == 5*63  # Plane at Z=1, one invalid depth pixel per camera.
        assert report["objects"][1]["points"] == 0 and not report["objects"][1]["valid"]
        with np.load(path.parent/"object_0.npz",allow_pickle=False) as data:
            assert data["xyz"].shape == (5*63,3)
            assert set(data["camera_ids"]) == set(range(5))
    episodes = {(row["task"],row["episode_index"]) for row in rows}
    html = list(previews.rglob("point-cloud-roi.html"))
    assert (previews/"index.html").is_file()
    from tests.test_oht_replay_viewer import viewer_payload
    viewer = viewer_payload(previews/"index.html")
    assert viewer["contract_sha256"] == contract["sha256"]
    preview_rows = rows if visualize_every == 1 else [row for row in rows if row["frame"]==0]
    assert {entry["sample_id"] for entry in viewer["frames"]} == {row["id"] for row in preview_rows}
    assert all((previews/entry["html"]).is_file() for entry in viewer["frames"])
    assert len(html) == len(preview_rows) == len(list(previews.rglob("*.png")))
    if visualize_every != 1:
        assert len(html) == len(episodes)
        assert all(path.parent.name == "000000" for path in html)
    # Each original preview contains object surfaces in its appended section.
    with Image.open(previews/rows[0]["id"]/"preview.png") as image:
        assert image.size == (1024,2150)
        pixels = np.asarray(image)
        assert np.any(np.all(pixels[1442:] == OBJECT_COLORS[0], axis=-1))
    assert not (output/"object_clouds").exists()
    for path in html:
        payload = payload_from_html(path)
        assert payload["contract_sha256"] == contract["sha256"]  # Not the provisional build hash.
        assert payload["preview_image"] == "preview.png"
        assert (path.parent/payload["preview_image"]).is_file()
        assert [entry["index"] for entry in payload["object_box_preset"]] == [0,1]
        assert payload["object_box_preset"][0]["name"] == "selected"
        assert payload["object_box_union_points"] == 5*63
    first, second = OHTDataset(f.replay)[0],OHTDataset(output)[0]
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key],second[key])


@pytest.mark.parametrize("selection", [None,[dict(index=0)]])
def test_build_writer_follows_current_frame_centers_without_preview_or_mutation(cloud,tmp_path,selection):
    objects(cloud)
    cloud[2]["diagnostic_objects"]["positions"][0] = [.25,.25,.25]
    before = deepcopy(cloud)
    writer = BuildObjectCloudWriter(tmp_path/"replay",dict(source="xyz",objects=selection))
    relative = writer.write(*cloud)
    first = json.loads((writer.output/relative).read_text(encoding="utf-8"))
    assert [entry["index"] for entry in first["objects"]] == ([0,1] if selection is None else [0])
    sample = deepcopy(cloud[2])
    sample.update(id="task/000000/000011",frame=11)
    sample["diagnostic_objects"]["frame"] = 11
    sample["diagnostic_objects"]["positions"][0] = [.75,.75,.75]
    relative = writer.write(cloud[0],cloud[1],sample)
    second = json.loads((writer.output/relative).read_text(encoding="utf-8"))
    for report, point in ((first,[.25,.25,.25]),(second,[.75,.75,.75])):
        assert report["objects"][0]["center"] == point and report["objects"][0]["points"] == 1
        with np.load(writer.output/"preview"/report["sample_id"]/"object_0.npz") as data:
            np.testing.assert_array_equal(data["xyz"],[point])
    assert not list(writer.output.rglob("*.html"))
    for key in cloud[0]:
        np.testing.assert_array_equal(cloud[0][key],before[0][key])
    assert cloud[1:] == before[1:]


def test_empty_object_frame_is_recorded_without_inventing_objects(cloud,tmp_path):
    objects(cloud)
    cloud[2]["diagnostic_objects"].update(positions=[],quaternions=[])
    writer = BuildObjectCloudWriter(tmp_path/"empty",dict(source="xyz"))
    relative = writer.write(*cloud,visualize=True)
    writer.bind_contract("final-checksum")
    report = json.loads((writer.output/relative).read_text(encoding="utf-8"))
    assert report["objects"] == [] and not list(writer.output.rglob("object_*.npz"))
    payload = payload_from_html(writer.output/relative.replace("object-clouds.json","point-cloud-roi.html"))
    assert payload["objects"] == [] and payload["object_box_preset"] == []
    assert payload["contract_sha256"] == "final-checksum"


def test_object_preview_sampling_keeps_small_crop_before_background(cloud):
    objects(cloud)
    cloud[2]["diagnostic_objects"]["positions"][0] = [1.25,.5,.5]
    spec = dict(schema="oht_object_boxes_v1",sample_id=cloud[2]["id"],frame=cloud[2]["frame"],source="xyz",
                objects=[dict(index=0,size=[.1,.1,.1])])
    payload = point_cloud_payload(*cloud,max_points=1,object_boxes=spec)
    assert payload["object_box_union_points"] == payload["displayed_points"] == 1
    points = np.frombuffer(base64.b64decode(payload["xyz"]),"<f4").reshape(-1,3)
    np.testing.assert_array_equal(points,[[1.25,.5,.5]])


@pytest.mark.parametrize("phase", ["export", "binding"])
def test_failed_object_export_does_not_mark_replay_complete(replay_fixture,tmp_path,monkeypatch,phase):
    from finetune.OHT.data import object_cloud_build
    def fail(*args,**kwargs):
        raise OSError("object export failed")
    if phase == "export":
        monkeypatch.setattr(object_cloud_build,"export_object_clouds",fail)
    else:
        monkeypatch.setattr(BuildObjectCloudWriter,"bind_contract",fail)
    f = replay_fixture
    output = tmp_path/"failed"
    with pytest.raises(OSError,match="object export failed"):
        build(f.root,f.manifest,f.config,output,object_cloud_config={})
    assert not (output/"complete.json").exists()


def test_browser_build_preview_restores_selected_boxes_and_downloads_final_contract(cloud,tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    executable = os.environ.get("OHT_BROWSER_EXECUTABLE")
    if not executable or not Path(executable).is_file():
        pytest.skip("Set OHT_BROWSER_EXECUTABLE for optional browser smoke test")
    objects(cloud)
    writer = BuildObjectCloudWriter(tmp_path/"built",dict(source="xyz",objects=[dict(index=0,name="wheel",size=[1,1,1])]))
    relative = writer.write(*cloud,visualize=True)
    writer.bind_contract("final-checksum")
    path = (writer.output/relative).parent/"point-cloud-roi.html"
    with playwright.sync_playwright() as p:
        context = p.chromium.launch_persistent_context(str(tmp_path/"browser-profile"),headless=True,
            executable_path=executable,offline=True,viewport=dict(width=1000,height=1100),timeout=30_000)
        page = context.pages[0]
        page.set_default_timeout(10_000)
        errors = []
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.goto(path.resolve().as_uri(),wait_until="domcontentloaded")
        page.wait_for_selector('#oht-roi[data-ready="true"]')
        page.wait_for_selector('#point-count[data-visible="2"]')
        assert page.locator("#only-object-clouds").is_checked()
        assert page.locator("#show-object-boxes").is_checked()
        assert page.get_by_role("checkbox",name="显示 object_0",exact=True).is_checked()
        assert not page.get_by_role("checkbox",name="显示 object_1",exact=True).is_checked()
        assert page.get_by_role("textbox",name="对象 0 名称",exact=True).input_value() == "wheel"
        assert page.get_by_role("spinbutton",name="对象 0 框 X 全长",exact=True).input_value() == "1"
        with page.expect_download() as download:
            page.locator("#download-object-boxes").click()
        spec = json.loads(Path(download.value.path()).read_text(encoding="utf-8"))
        assert spec["contract_sha256"] == "final-checksum"
        report = export_object_clouds(tmp_path/"reexport",*cloud,spec,"final-checksum",{})
        assert report["objects"][0]["points"] == 2
        for width in (1000,320):
            page.set_viewport_size(dict(width=width,height=1100))
            page.wait_for_selector(f'#oht-roi[data-render-width="{width-32}"]')
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(tmp_path/f"build-object-preview-{width}.png"),full_page=True)
        assert not errors
        context.close()

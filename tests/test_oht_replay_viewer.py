"""Offline frame navigation preserves display state, never old-frame geometry."""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
from PIL import Image

import pytest

from finetune.OHT.cli import main
from finetune.OHT.data.common import file_digest, read_jsonl
from finetune.OHT.data.html_preview import save_point_cloud_html
from finetune.OHT.data.replay import load_contract
from finetune.OHT.data.replay_viewer import export_replay_viewer, save_replay_viewer
from tests.test_oht_html_preview import cloud, payload_from_html
from tests.test_oht_migration import replay_fixture
from tests.test_oht_object_cloud_build import objects


def viewer_payload(path):
    return json.loads(re.search(r'<script id="oht-replay-data" type="application/json">(.*?)</script>',
                               path.read_text(encoding="utf-8"),re.S)[1])


def test_viewer_escapes_names_encodes_local_paths_and_refuses_overwrite(tmp_path):
    child = tmp_path/"frames"/"name #中文.html"
    child.parent.mkdir();child.write_text("preview",encoding="utf-8")
    image = child.with_suffix(".png")
    Image.new("RGB",(2,2),"orange").save(image)
    sample = "</script><script>injected=true</script>/000010"
    entries = [dict(sample_id=sample,frame=10,target_frame=30,timestamp=1.2,html=child,image=image)]
    output = save_replay_viewer(tmp_path/"index.html",entries,"checksum")
    payload = viewer_payload(output)
    assert payload["contract_sha256"] == "checksum"
    assert payload["frames"][0]["sample_id"] == sample
    assert payload["frames"][0]["html"] == "frames/name%20%23%E4%B8%AD%E6%96%87.html"
    assert payload["frames"][0]["image"] == "frames/name%20%23%E4%B8%AD%E6%96%87.png"
    assert "</script><script>injected=true" not in output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_replay_viewer(output,entries,"checksum")


@pytest.mark.parametrize("case",["empty","duplicate","noninteger","outside","missing","outside-image","missing-image"])
def test_invalid_viewer_inputs_do_not_publish_index(tmp_path,case):
    child = tmp_path/"frames"/"frame.html"
    child.parent.mkdir();child.write_text("preview",encoding="utf-8")
    entry = dict(sample_id="task/000000/000010",frame=10,html=child)
    entries = [entry]
    if case == "empty": entries=[]
    elif case == "duplicate": entries.append(dict(entry))
    elif case == "noninteger": entry["frame"]=True
    elif case == "outside": entry["html"]=tmp_path.parent/"outside.html"
    elif case == "missing": entry["html"]=tmp_path/"missing.html"
    elif case == "outside-image": entry["image"]=tmp_path.parent/"outside.png"
    elif case == "missing-image": entry["image"]=tmp_path/"missing.png"
    with pytest.raises(ValueError):
        save_replay_viewer(tmp_path/"index.html",entries,"checksum")
    assert not (tmp_path/"index.html").exists()


def test_export_viewer_selects_emitted_frames_per_episode_without_cache_changes(replay_fixture,tmp_path):
    f = replay_fixture
    rows = read_jsonl(f.replay/"samples.jsonl")
    inputs = [f.replay/"contract.json",f.replay/"samples.jsonl",*[f.replay/row["observation"] for row in rows]]
    hashes = [file_digest(path) for path in inputs]
    output = tmp_path/"viewer"
    assert main(["export-viewer","--replay",str(f.replay),"--output",str(output),"--every","2",
                 "--max-points","5"]) == 0
    payload = viewer_payload(output/"index.html")
    episodes = dict.fromkeys(row["id"].rsplit("/",1)[0] for row in rows)
    expected = [row for episode in episodes for row in [r for r in rows if r["id"].rsplit("/",1)[0]==episode][::2]]
    assert {entry["sample_id"] for entry in payload["frames"]} == {row["id"] for row in expected}
    assert payload["contract_sha256"] == load_contract(f.replay)["sha256"]
    for entry in payload["frames"]:
        row = next(row for row in rows if row["id"]==entry["sample_id"])
        assert entry["frame"] == row["frame"] and entry["target_frame"] == row["target_frame"]
        child = payload_from_html(output/entry["html"])
        assert child["displayed_points"] == 5
        assert child["geometry_source"] == "cached_metric_depth_backprojection"
        assert child["contract_sha256"] == payload["contract_sha256"]
        assert child["objects"]  # Raw object metadata remains available without a preset.
        assert child["preview_image"] == "preview.png"
        assert (output/entry["html"]).with_name("preview.png").is_file()
    assert [file_digest(path) for path in inputs] == hashes
    with pytest.raises(FileExistsError):
        export_replay_viewer(f.replay,output)


def test_export_viewer_explicit_object_config_generates_presets(replay_fixture,tmp_path):
    output = tmp_path/"objects"
    count = export_replay_viewer(replay_fixture.replay,output,every=100,max_points=5,source="xyz",
                               object_cloud_config=dict(objects=[dict(index=0,name="fixture",size=[10,10,10])]))
    payload = viewer_payload(output/"index.html")
    assert count == len(payload["frames"])
    for entry in payload["frames"]:
        child = payload_from_html(output/entry["html"])
        assert child["object_box_preset"][0]["name"] == "fixture"
        assert child["object_box_preset"][0]["size"] == [10,10,10]
        assert child["geometry_source"] == "cached_xyz"
    assert not list(output.rglob("*.npz"))  # HTML export does not duplicate object surfaces on disk.


def test_changed_observation_does_not_publish_viewer_index(replay_fixture,tmp_path):
    replay=tmp_path/"tampered-replay"
    shutil.copytree(replay_fixture.replay,replay)
    sample=read_jsonl(replay/"samples.jsonl")[0]
    with (replay/sample["observation"]).open("ab") as stream:
        stream.write(b"changed")
    output=tmp_path/"viewer"
    with pytest.raises(ValueError,match="Observation changed"):
        export_replay_viewer(replay,output)
    assert not (output/"index.html").exists()


@pytest.mark.parametrize("options",[dict(every=0),dict(every=True),dict(max_points=0),dict(source="bad"),
                                   dict(object_root="missing")])
def test_invalid_export_options_fail_before_output(replay_fixture,tmp_path,options):
    output = tmp_path/"invalid"
    with pytest.raises(ValueError):
        export_replay_viewer(replay_fixture.replay,output,**options)
    assert not output.exists()


def test_browser_navigates_offline_retains_view_and_uses_new_frame_object_centers(cloud,tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    executable = os.environ.get("OHT_BROWSER_EXECUTABLE")
    if not executable or not Path(executable).is_file():
        pytest.skip("Set OHT_BROWSER_EXECUTABLE for optional browser smoke test")
    objects(cloud)
    frames=[]
    for episode,number,center in [(0,10,.5),(0,30,1.25),(1,7,.25)]:
        sample=deepcopy(cloud[2]);sample.update(id=f"task/{episode:06d}/{number:06d}",frame=number,target_frame=number+5,
            diagnostic_contract_sha256="final-checksum",observation_sha256=f"observation-{number}")
        sample["diagnostic_objects"]["frame"]=number
        sample["diagnostic_objects"]["positions"][0]=[center,center,center]
        spec=dict(schema="oht_object_boxes_v1",sample_id=sample["id"],frame=number,source="xyz",
                  objects=[dict(index=0,name="wheel",size=[1,1,1]),dict(index=1,size=[1,1,1])])
        path=tmp_path/"preview"/sample["id"]/"point-cloud-roi.html"
        path.parent.mkdir(parents=True)
        image=path.parent/"preview.png"
        Image.new("RGB",(64,96),"orange").save(image)
        path=save_point_cloud_html(path,cloud[0],cloud[1],sample,object_boxes=spec,preview_image=image)
        frames.append(dict(sample_id=sample["id"],frame=number,target_frame=number+5,html=path,image=image))
    index=save_replay_viewer(tmp_path/"preview/index.html",frames,"final-checksum")
    with playwright.sync_playwright() as p:
        context=p.chromium.launch_persistent_context(str(tmp_path/"browser-profile"),headless=True,
            executable_path=executable,offline=True,viewport=dict(width=1000,height=1100),timeout=30_000)
        page=context.pages[0];page.set_default_timeout(10_000)
        errors=[];page.on("pageerror",lambda error:errors.append(str(error)))
        page.goto(index.resolve().as_uri(),wait_until="domcontentloaded")
        def ready(number,episode=0):
            page.wait_for_selector(f'#oht-replay-viewer[data-ready="true"][data-current-sample="task/{episode:06d}/{number:06d}"]')
        ready(10)
        child=page.frame_locator("#preview")
        assert page.locator("#frame-preview-section").is_visible()
        assert page.locator("#frame-preview").evaluate("image => image.complete && image.naturalWidth === 64")
        assert not child.locator("#frame-preview-section").is_visible()  # Parent shows PNG once, above sandboxed 3D.
        assert page.locator("#frame-preview").bounding_box()["y"] < page.locator("#preview").bounding_box()["y"]
        assert page.locator("#previous").is_disabled()
        assert page.locator("#frame option").all_text_contents()==["10","30"]
        page.locator('[data-view="xy"]').click()
        child.locator('#oht-roi[data-view-pitch="1.5707963267948966"]').wait_for()
        assert "俯仰 90.0°" in page.locator("#camera-angle").inner_text()
        child.locator('input[aria-label="对象 0 名称"]').fill("custom wheel")
        for axis in "XYZ":
            child.locator(f'input[aria-label="对象 0 框 {axis} 全长"]').fill("2")
        child.locator('input[aria-label="显示 object_1"]').uncheck()
        child.locator("#cloud-opacity").fill("50")
        child.locator("#cloud-opacity").dispatch_event("input")
        # Wait for the last draw before navigation so the state is published.
        child.locator('#oht-roi[data-object-count="1"]').wait_for()
        page.locator("#next").click();ready(30)
        assert page.locator("#next").is_disabled()
        assert "frame 30" in page.locator("#frame-label").inner_text()
        assert child.locator('input[aria-label="对象 0 名称"]').input_value()=="custom wheel"
        assert child.locator('input[aria-label="对象 0 框 X 全长"]').input_value()=="2"
        assert not child.locator('input[aria-label="显示 object_1"]').is_checked()
        assert child.locator("#cloud-opacity").input_value()=="50"
        assert float(child.locator("#oht-roi").get_attribute("data-view-pitch"))==pytest.approx(1.57079632679)
        assert "world [1.25, 1.25, 1.25]" in child.locator("#object-controls").inner_text()
        with page.expect_download() as downloaded:
            child.locator("#download-object-boxes").click()
        config=json.loads(Path(downloaded.value.path()).read_text(encoding="utf-8"))
        assert config["frame"]==30 and config["sample_id"]=="task/000000/000030"
        assert config["observation_sha256"]=="observation-30"
        assert config["objects"][0]["name"]=="custom wheel" and config["objects"][0]["size"]==[2,2,2]
        page.locator("#episode").select_option("task/000001");ready(7,1)
        assert child.locator('input[aria-label="对象 0 名称"]').input_value()=="wheel"
        assert child.locator('input[aria-label="显示 object_1"]').is_checked()
        page.locator("#episode").select_option("task/000000");ready(10)
        assert child.locator('input[aria-label="对象 0 名称"]').input_value()=="custom wheel"
        page.locator("#keep-view").uncheck()
        page.locator("#frame-slider").fill("1");ready(30)
        assert float(child.locator("#oht-roi").get_attribute("data-view-pitch"))==pytest.approx(.65)
        for width in (1000,320):
            page.set_viewport_size(dict(width=width,height=1100))
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert child.locator("body").evaluate("() => document.documentElement.scrollWidth <= innerWidth")
            page.locator('[data-view="3d"]').click()
            child.locator('#oht-roi[data-view-pitch="0.65"]').wait_for()
            page.screenshot(path=str(tmp_path/f"replay-viewer-{width}.png"),full_page=True)
        assert not errors
        standalone=context.new_page()
        standalone.goto(frames[0]["html"].resolve().as_uri(),wait_until="load")
        assert standalone.locator("#frame-preview-section").is_visible()
        assert standalone.locator("#frame-preview").evaluate("image => image.complete && image.naturalWidth === 64")
        assert standalone.locator("#frame-preview").bounding_box()["y"] < standalone.locator("#cloud").bounding_box()["y"]
        context.close()

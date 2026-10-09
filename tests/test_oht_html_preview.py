"""Offline ROI export, cache isolation, and optional real-browser interactions."""
from copy import deepcopy
import base64
import json
import os
from pathlib import Path
import re

import numpy as np
import pytest
import yaml

from finetune.OHT.cli import main
from finetune.OHT.data.common import file_digest, read_jsonl
from finetune.OHT.data.html_preview import MAX_HTML_POINTS, point_cloud_payload, save_point_cloud_html
from finetune.OHT.data.point_filter import point_cloud_mask
from tests.test_oht_migration import replay_fixture


@pytest.fixture
def cloud():
    points = np.array([[.25,.25,.25], [.75,.75,.75], [1.25,.5,.5], [np.nan,np.nan,np.nan]], np.float32)
    observation = dict(wrist_point_cloud=points.T[:,None,:],
                       wrist_rgb=np.full((3,1,4), [180], np.uint8))
    config = dict(cameras={"wrist": {}}, scene_bounds=[0,0,0,1,1,1], point_cloud_filter=dict(
        enabled=True, keep_bounds=[0,0,0,1,1,1], exclude_boxes=[[.7,.7,.7,.8,.8,.8]]))
    sample = dict(id="task/000000/000010", frame=10, target_frame=20,
                  current_tcp=[.25,.25,.25,0,0,0,1], labels=dict(gripper_pose=[.75,.75,.75,0,0,0,1]))
    return observation, config, sample


def payload_from_html(path):
    text = path.read_text(encoding="utf-8")
    return json.loads(re.search(r'<script id="oht-cloud-data" type="application/json">(.*?)</script>', text, re.S)[1])


def test_payload_keeps_finite_outside_workspace_and_roi_points_without_mutation(cloud):
    observation, config, sample = cloud
    before = ({key: value.copy() for key,value in observation.items()}, deepcopy(config), deepcopy(sample))
    payload = point_cloud_payload(*cloud)
    points = np.frombuffer(base64.b64decode(payload["xyz"]), "<f4").reshape(-1,3)
    np.testing.assert_array_equal(points, observation["wrist_point_cloud"].reshape(3,-1).T[:3])
    assert payload["total_points"] == payload["displayed_points"] == 3
    assert payload["cameras"][0]["invalid_points"] == 1
    assert point_cloud_mask(points, config["point_cloud_filter"]).tolist() == [True,False,False]
    assert "cannot be recovered" in payload["source_note"]
    for key in observation:
        np.testing.assert_array_equal(observation[key], before[0][key])
    assert config == before[1] and sample == before[2]


def test_display_sampling_is_deterministic_and_keeps_rgb_and_camera_alignment(cloud):
    observation, config, sample = cloud
    observation["wrist_rgb"] = np.arange(12, dtype=np.uint8).reshape(3,1,4)
    payload = point_cloud_payload(*cloud, max_points=2)
    assert payload == point_cloud_payload(*cloud, max_points=2)
    assert payload["total_points"] == 3 and payload["displayed_points"] == 2
    expected = observation["wrist_rgb"].reshape(3,-1).T[[0,2]]
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(payload["rgb"]), "u1").reshape(-1,3), expected)
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(payload["camera_ids"]), "u1"), [0,0])


def test_camera_provenance_is_preserved_for_each_point(cloud):
    observation, config, _ = cloud
    config["cameras"]["global_left"] = {}
    observation["global_left_point_cloud"] = observation["wrist_point_cloud"] + 1
    observation["global_left_rgb"] = np.full((3,1,4),17,np.uint8)
    payload = point_cloud_payload(*cloud)
    assert [camera["name"] for camera in payload["cameras"]] == ["wrist","global_left"]
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(payload["camera_ids"]),"u1"),[0,0,0,1,1,1])
    colors = np.frombuffer(base64.b64decode(payload["rgb"]),"u1").reshape(-1,3)
    assert (colors[:3] == 180).all() and (colors[3:] == 17).all()


def test_depth_source_recovers_roi_masked_points_without_redecoding_or_cache_edits(cloud):
    observation, config, _ = cloud
    observation["wrist_point_cloud"][:] = np.nan
    observation.update(wrist_depth=np.array([[[1.,1.2,0.,np.nan]]],np.float32),
                       wrist_camera_intrinsics=np.eye(3), wrist_camera_extrinsics=np.eye(4))
    config["depth"] = dict(kind="ray",limits=[.001,10])
    config["cameras"]["wrist"]["depth"] = dict(kind="z",limits=[.001,10])  # per-camera contract wins
    before = {key:value.copy() for key,value in observation.items()}
    payload = point_cloud_payload(*cloud,source="depth")
    assert payload["geometry_source"] == "cached_metric_depth_backprojection"
    points = np.frombuffer(base64.b64decode(payload["xyz"]),"<f4").reshape(-1,3)
    np.testing.assert_allclose(points,[[0,0,1],[1.2,0,1.2]])  # depth was already metric
    assert "ROI-masked" in payload["source_note"]
    for key,value in before.items():
        np.testing.assert_array_equal(observation[key],value)


def test_depth_source_requires_metric_depth_and_cached_calibration(cloud,tmp_path):
    with pytest.raises(ValueError,match="cached metric depth/K"):
        save_point_cloud_html(tmp_path/"roi.html",*cloud,source="depth")
    assert not (tmp_path/"roi.html").exists()


def test_depth_source_refuses_raw_integer_codes(cloud):
    observation, config, _ = cloud
    observation.update(wrist_depth=np.ones((1,1,4),np.uint16),
                       wrist_camera_intrinsics=np.eye(3), wrist_camera_extrinsics=np.eye(4))
    config["depth"] = dict(kind="ray",limits=[.001,10])
    with pytest.raises(ValueError,match="already-metric"):
        point_cloud_payload(*cloud,source="depth")


@pytest.mark.parametrize("limit", [0,-1,MAX_HTML_POINTS+1,True,1.5])
def test_invalid_point_budget_does_not_create_output(cloud, tmp_path, limit):
    output = tmp_path / "missing" / "roi.html"
    with pytest.raises(ValueError, match="max_points"):
        save_point_cloud_html(output,*cloud,max_points=limit)
    assert not output.parent.exists()


def test_empty_geometry_refuses_html_instead_of_inventing_points(cloud,tmp_path):
    cloud[0]["wrist_point_cloud"][:] = np.nan
    with pytest.raises(ValueError, match="No finite cached XYZ"):
        save_point_cloud_html(tmp_path/"roi.html",*cloud)
    assert not (tmp_path/"roi.html").exists()


def test_html_is_offline_escapes_untrusted_data_and_refuses_overwrite(cloud,tmp_path):
    cloud[2]["id"] = '</script><script>window.injected=true</script>测试'
    path = save_point_cloud_html(tmp_path/"roi.html",*cloud)
    content = path.read_text(encoding="utf-8")
    assert payload_from_html(path)["sample_id"] == cloud[2]["id"]
    assert '</script><script>window.injected=true' not in content
    assert "__OHT_POINT_CLOUD_DATA__" not in content
    assert not re.search(r'<script[^>]+src=|fetch\(|XMLHttpRequest|WebSocket|https?://',content)
    assert "connect-src 'none'" in content
    before = file_digest(path)
    with pytest.raises(FileExistsError):
        save_point_cloud_html(path,*cloud)
    assert file_digest(path) == before


def test_maximum_budget_stays_under_two_megabytes(cloud,tmp_path):
    cloud[0]["wrist_point_cloud"] = np.tile(np.array([.1,.2,.3],np.float32)[:,None,None],(1,1,MAX_HTML_POINTS))
    cloud[0]["wrist_rgb"] = np.full((3,1,MAX_HTML_POINTS),120,np.uint8)
    path = save_point_cloud_html(tmp_path/"large.html",*cloud,max_points=MAX_HTML_POINTS)
    assert path.stat().st_size < 2_000_000


def test_geometry_cli_html_is_opt_in_and_preserves_cached_files(replay_fixture,tmp_path):
    f = replay_fixture
    row = read_jsonl(f.replay/"samples.jsonl")[0]
    inputs = [f.replay/"contract.json",f.replay/"samples.jsonl",f.replay/row["observation"]]
    before = [file_digest(path) for path in inputs]
    output = tmp_path/"geometry-html"
    args = ["diagnose-geometry","--replay",str(f.replay),"--sample-id",row["id"],"--output",str(output),"--html","--max-points","2"]
    assert main(args) == 0
    payload = payload_from_html(output/"point-cloud-roi.html")
    assert payload["sample_id"] == row["id"] and payload["displayed_points"] == 2
    assert [file_digest(path) for path in inputs] == before
    recovered = tmp_path/"depth-html"
    assert main([*args[:5],"--output",str(recovered),"--html","--html-source","depth","--max-points","2"]) == 0
    assert payload_from_html(recovered/"point-cloud-roi.html")["geometry_source"] == "cached_metric_depth_backprojection"
    assert [file_digest(path) for path in inputs] == before
    with pytest.raises(FileExistsError):
        main(args)
    invalid = tmp_path/"bad-budget"
    with pytest.raises(ValueError,match="max_points"):
        main([*args[:5],"--output",str(invalid),"--html","--max-points","0"])
    assert not invalid.exists()


def test_browser_edits_bounds_and_exports_matching_yaml_offline(cloud,tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    executable = os.environ.get("OHT_BROWSER_EXECUTABLE")
    if not executable or not Path(executable).is_file():
        pytest.skip("Set OHT_BROWSER_EXECUTABLE for optional browser smoke test")
    path = save_point_cloud_html(tmp_path/"roi.html",*cloud)
    with playwright.sync_playwright() as p:
        # Use a task-owned ordinary profile, not an incognito context (which
        # managed browsers may disallow). Never reuse the user's browser data.
        context = p.chromium.launch_persistent_context(str(tmp_path/"browser-profile"),
            headless=True,executable_path=executable,offline=True,
            viewport=dict(width=1000,height=1100),timeout=30_000)
        assert context.pages, "Browser did not open its initial test page"
        page = context.pages[0]
        page.set_default_timeout(10_000)
        page.set_default_navigation_timeout(15_000)
        errors, network = [], []
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.on("request",lambda request:network.append(request.url) if request.url.startswith(("http:","https:")) else None)
        page.goto(path.resolve().as_uri(),wait_until="domcontentloaded")
        page.wait_for_selector('#oht-roi[data-ready="true"]')
        count = page.locator("#point-count")
        assert count.get_attribute("data-retained") == "1"
        assert count.get_attribute("data-visible") == "3"
        page.locator("#box-select").select_option("scene")
        field = page.locator('[data-bound="3"]')
        field.fill("1.5"); field.press("Tab")
        page.wait_for_selector('#point-count[data-retained="1"]')
        page.locator("#filter-enabled").uncheck()
        page.wait_for_selector('#point-count[data-retained="3"]')
        page.locator("#keep-enabled").uncheck()
        page.locator("#add-exclude").click()
        parsed = yaml.safe_load(page.locator("#yaml").input_value())
        assert parsed["scene_bounds"] == [0,0,0,1.5,1,1]
        assert parsed["point_cloud_filter"]["keep_bounds"] is None
        assert len(parsed["point_cloud_filter"]["exclude_boxes"]) == 2
        page.locator("#remove-exclude").click()
        assert len(yaml.safe_load(page.locator("#yaml").input_value())["point_cloud_filter"]["exclude_boxes"]) == 1
        page.locator("#filter-enabled").check()
        page.wait_for_selector('#point-count[data-retained="2"]')
        data = yaml.safe_load(page.locator("#yaml").input_value())
        xyz = cloud[0]["wrist_point_cloud"].reshape(3,-1).T[:3]
        assert int(point_cloud_mask(xyz,data["point_cloud_filter"]).sum()) == 2
        before = page.locator("#yaml").input_value()
        field.fill("0"); field.press("Tab")
        assert page.locator("#error").inner_text() and page.locator("#download-yaml").is_disabled()
        assert page.locator("#yaml").input_value() == before
        field.fill("1.5"); field.press("Tab")
        assert not page.locator("#error").inner_text()
        page.locator("#color-mode").select_option("camera")
        page.get_by_role("checkbox",name="wrist",exact=True).uncheck()
        page.wait_for_selector('#point-count[data-visible="0"]')
        page.get_by_role("checkbox",name="wrist",exact=True).check()
        page.locator('[data-view="xz"]').click()
        old_image = page.locator("#cloud").evaluate("canvas => canvas.toDataURL()")
        bounds = page.locator("#cloud").bounding_box()
        page.mouse.move(bounds["x"]+bounds["width"]/2,bounds["y"]+bounds["height"]/2)
        page.mouse.down(); page.mouse.move(bounds["x"]+bounds["width"]/2+60,bounds["y"]+bounds["height"]/2+30,steps=3); page.mouse.up()
        page.wait_for_timeout(100)
        assert page.locator("#cloud").evaluate("canvas => canvas.toDataURL()") != old_image
        with page.expect_download() as download:
            page.locator("#download-yaml").click()
        assert yaml.safe_load(Path(download.value.path()).read_text(encoding="utf-8")) == data
        page.bring_to_front()
        for width in (1000,320):
            page.set_viewport_size(dict(width=width,height=1100))
            page.wait_for_selector(f'#oht-roi[data-render-width="{width-32}"]')
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert page.locator("#cloud").evaluate("canvas => canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data.some(v => v !== 0)"), errors
            page.screenshot(path=str(tmp_path/f"roi-{width}.png"),full_page=True)
        field.fill("1.500000123456789"); field.press("Tab")
        assert field.input_value() == "1.500000123456789"
        assert yaml.safe_load(page.locator("#yaml").input_value())["scene_bounds"][3] == 1.500000123456789
        # Richer synthetic tabletop/box for visual QA, not a claim about OHT GT.
        a,b = np.meshgrid(np.linspace(0,1,80),np.linspace(0,1,80))
        tabletop = np.c_[a.ravel(),b.ravel(),np.full(a.size,.15)]
        a,b = np.meshgrid(np.linspace(.3,.6,25),np.linspace(.3,.6,25))
        top = np.c_[a.ravel(),b.ravel(),np.full(a.size,.45)]
        sides = np.r_[np.c_[a.ravel(),np.full(a.size,.3),b.ravel()-.15],
                      np.c_[np.full(a.size,.6),a.ravel(),b.ravel()-.15]]
        points = np.r_[tabletop,top,sides].astype(np.float32)
        colors = np.r_[np.tile([140,150,160],(len(tabletop),1)),np.tile([230,110,45],(len(top)+len(sides),1))].astype(np.uint8)
        dense = {"wrist_point_cloud":points.T[:,None,:],"wrist_rgb":colors.T[:,None,:]}
        path = save_point_cloud_html(tmp_path/"dense.html",dense,cloud[1],cloud[2])
        page.goto(path.resolve().as_uri(),wait_until="domcontentloaded")
        page.wait_for_selector('#oht-roi[data-ready="true"]')
        for width in (1000,320):
            page.set_viewport_size(dict(width=width,height=1100))
            page.wait_for_selector(f'#oht-roi[data-render-width="{width-32}"]')
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(tmp_path/f"roi-dense-{width}.png"),full_page=True)
        page.emulate_media(color_scheme="dark")
        page.wait_for_timeout(100)
        assert page.locator("#cloud").evaluate("canvas => canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data.some(v => v !== 0)"), errors
        page.screenshot(path=str(tmp_path/"roi-dense-dark-320.png"),full_page=True)
        assert not errors and not network
        print("ROI browser previews:",tmp_path)
        context.close()

"""Same-image object diagnostics keep surfaces, occlusion and source isolation."""
from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw

from finetune.OHT.data import visualization as preview
from finetune.OHT.data.object_cloud import export_object_clouds
from tests.test_oht_html_preview import cloud
from tests.test_oht_object_cloud_build import objects
from tests.test_oht_visualization import _minimal_preview


def specification(sample, source="xyz", size=(1,1,1), indices=(0,)):
    return dict(schema="oht_object_boxes_v1",sample_id=sample["id"],frame=sample["frame"],source=source,
                objects=[dict(index=index,size=list(size)) for index in indices])


def test_box_overlay_has_unique_union_colors_and_explicit_shared_membership(cloud):
    objects(cloud)
    cloud[2]["diagnostic_objects"]["positions"][1] = [.5,.5,.5]
    overlay = preview._object_cloud_overlay(*cloud,specification(cloud[2],indices=(0,1)),None)
    assert len(overlay["points"]) == 2  # Shared box membership never duplicates the source points.
    assert [box["source_points"] for box in overlay["boxes"]] == [2,2]
    np.testing.assert_array_equal(overlay["colors"],[preview.OBJECT_OVERLAP]*2)
    assert overlay["boxes"][0]["color"] != overlay["boxes"][1]["color"]
    single = preview._object_cloud_overlay(*cloud,specification(cloud[2]),None)
    np.testing.assert_array_equal(single["colors"],[preview.OBJECT_COLORS[0]]*2)


def test_depth_overlay_matches_export_even_when_cached_xyz_is_roi_masked(cloud,tmp_path):
    objects(cloud)
    observation,config,sample = cloud
    observation["wrist_point_cloud"][:] = np.nan
    observation.update(wrist_depth=np.array([[[1.,1.2,0.,np.nan]]],np.float32),
                       wrist_camera_intrinsics=np.eye(3),wrist_camera_extrinsics=np.eye(4))
    config.update(depth=dict(kind="z",limits=[.001,3]),scene_bounds=[-1,-1,0,2,2,2])
    sample["diagnostic_objects"]["positions"][0] = [0,0,1]
    spec = specification(sample,source="depth",size=(.2,.2,.2))
    before = deepcopy(cloud)
    overlay = preview._object_cloud_overlay(*cloud,spec,[-.1,-.1,.9,.1,.1,1.1])
    report = export_object_clouds(tmp_path/"export",*cloud,spec,None,{})
    assert overlay["boxes"][0]["source_points"] == report["objects"][0]["points"] == 1
    assert overlay["boxes"][0]["world_points"] == overlay["boxes"][0]["local_points"] == 1
    with np.load(tmp_path/"export/object_0.npz") as data:
        np.testing.assert_array_equal(overlay["points"],data["xyz"])
    for key in observation:
        np.testing.assert_array_equal(observation[key],before[0][key])
    assert cloud[1:] == before[1:]


def test_overlay_uses_same_depth_test_as_scene_and_keeps_single_pixel_surface():
    box = dict(index=0,name="test",center=[.65,.65,.2],bounds=[.4,.4,.1,.9,.9,.3],
               color=preview.OBJECT_COLORS[0],source_points=1)
    overlay = dict(points=np.array([[.5,.5,.2]]),colors=np.array([box["color"]],np.uint8),boxes=[box])
    bounds = [0,0,0,1,1,1]
    foreground = np.array([[.5,.5,.8]])
    front_color = np.array([[0,0,255]],np.uint8)
    hidden = np.asarray(preview._orthographic(foreground,front_color,bounds,(0,1),None,None,None,None,
                                             object_overlay=overlay))
    np.testing.assert_array_equal(hidden[127,169],[0,0,255])
    np.testing.assert_array_equal(hidden[127,170],[0,0,0])
    visible = np.asarray(preview._orthographic(overlay["points"],np.array([[160,160,160]],np.uint8),
                                              bounds,(0,1),None,None,None,None,object_overlay=overlay))
    np.testing.assert_array_equal(visible[127,169],box["color"])
    np.testing.assert_array_equal(visible[127,170],[0,0,0])


def test_object_surface_is_sampled_independently_from_scene_and_png_preserves_inputs(monkeypatch,tmp_path):
    point = np.array([.45,.48,.5])
    observation,config,sample = _minimal_preview(np.r_[np.tile([.9,.9,.9],(100,1)),point[None]], [.5,.5,.5])
    sample["diagnostic_objects"] = dict(frame=0,positions=[[.5,.5,.5]],quaternions=[[1,0,0,0]])
    before = deepcopy((observation,config,sample))
    spec = specification(sample,size=(.2,.2,.2))
    calls = []
    original = preview._orthographic
    def record(points,colors,bounds,*args,**kwargs):
        calls.append((points.copy(),kwargs["object_overlay"]))
        return original(points,colors,bounds,*args,**kwargs)
    monkeypatch.setattr(preview,"MAX_PREVIEW_POINTS",1)
    monkeypatch.setattr(preview,"_orthographic",record)
    path = preview.save_preview(tmp_path/"overlay.png",observation,config,sample,object_boxes=spec)
    assert len(calls) == 6
    for base,overlay in calls[:3]:
        np.testing.assert_array_equal(base,[[.9,.9,.9]])
        np.testing.assert_array_equal(overlay["points"],[point])
    for base,overlay in calls[3:]:
        np.testing.assert_array_equal(base,[point])
        np.testing.assert_array_equal(overlay["points"],[point])
    with Image.open(path) as image:
        assert image.size == (1024,1054)  # Original one-camera image plus two header lines.
    for key in observation:
        np.testing.assert_array_equal(observation[key],before[0][key])
    assert config == before[1] and sample == before[2]


def test_empty_and_out_of_view_clouds_have_actual_counts_and_do_not_invent_geometry(cloud,tmp_path,monkeypatch):
    objects(cloud)
    spec = specification(cloud[2],indices=(1,))
    overlay = preview._object_cloud_overlay(*cloud,spec,[0,0,0,1,1,1])
    assert len(overlay["points"]) == 0
    assert overlay["boxes"][0]["source_points"] == overlay["boxes"][0]["world_points"] == 0
    calls = []
    original = ImageDraw.ImageDraw.text
    def text(self,xy,value,*args,**kwargs):
        calls.append(value)
        return original(self,xy,value,*args,**kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw,"text",text)
    # A populated box can still be entirely outside the shown workspace.
    cloud[1]["scene_bounds"] = [0,0,0,.1,.1,.1]
    cloud[1]["depth"] = dict(kind="z",limits=[.001,3])
    cloud[0].update(wrist_depth=np.ones((1,1,4)),wrist_camera_intrinsics=np.eye(3),wrist_camera_extrinsics=np.eye(4))
    spec = specification(cloud[2],size=(.2,.2,.2))
    cloud[2]["diagnostic_objects"]["positions"][0] = [.25,.25,.25]
    preview.save_preview(tmp_path/"outside.png",*cloud,object_boxes=spec)
    assert any("source=1 world=0" in value for value in calls)

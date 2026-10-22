"""Simulator overlays are cleared across every policy sensor read."""
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_vla_server import fixture  # noqa: F401
from tests.test_oht_isaaclab_http import raw_sample
from tests.test_oht_native_runtime import environment
from tests.test_oht_standalone_client import sensors_for
from finetune.OHT.runtime.isaaclab_bridge import IsaacLabBridge
from finetune.OHT.runtime.isaaclab_runtime import IsaacLabRuntime
from finetune.OHT.runtime.live_markers import LiveMarkers
from finetune.OHT.runtime.raw_observation import raw_schema


class Drawer:
    def __init__(self):
        self.points, self.lines = [], []
    def clear_points(self):
        self.points.clear()
    def clear_lines(self):
        self.lines.clear()
    def draw_points(self, positions, colors, widths):
        self.points.extend(positions)
    def draw_lines(self, starts, ends, colors, widths):
        self.lines.extend(zip(starts, ends))


@pytest.mark.parametrize('read_failure', [False, True])
def test_live_markers_are_absent_during_render_and_sensor_reads(fixture, replay_fixture, read_failure):
    tensors, _, raw = raw_sample(replay_fixture.replay, fixture.policy)
    env = environment('eef', raw['task'])
    env.action_manager.get_term('arm_action').cfg.body_name = 'tool'
    cameras = sensors_for(tensors, fixture.policy.contract['data_config']['cameras'])
    runtime = IsaacLabRuntime(env, cameras, ee_link='tool', quaternion_order='wxyz', simulation_app=None)
    drawer = Drawer()
    markers = LiveMarkers(runtime, env, drawer=drawer)
    client = SimpleNamespace(action_config=fixture.policy.action_config,
                             info={'raw_observation': raw_schema(fixture.policy)})
    bridge = IsaacLabBridge(runtime, env, client, raw['task'], raw['data_profile'], cameras, visualizer=markers)
    target = [.2, 0, .5, 0, 0, 0, 1, 1]
    markers.update(bridge.measured_state(), target)
    assert drawer.points and drawer.lines
    reads = []

    def render():
        assert not drawer.points and not drawer.lines
        reads.append('render')
    original_rgb = runtime.camera_rgb
    def rgb(camera):
        assert not drawer.points and not drawer.lines
        reads.append('rgb')
        if read_failure:
            raise ValueError('Camera unavailable')
        return original_rgb(camera)
    env.sim.render = render
    runtime.camera_rgb = rgb
    try:
        if read_failure:
            with pytest.raises(ValueError, match='Camera unavailable'):
                bridge.capture()
        else:
            measured = bridge.capture()
            assert env.common_step_counter == 0
            # Redraw is restored only after independent RGB-D snapshots exist.
            assert drawer.points and drawer.lines
            for name in cameras:
                np.testing.assert_array_equal(measured[name + '_rgb'], tensors[name + '_rgb'])
            bridge.execute(np.asarray(target), measured)
            assert env.common_step_counter == 1
            assert any(np.allclose(point, target[:3], atol=1e-7) for point in drawer.points)
        assert reads[0] == 'render' and 'rgb' in reads
        assert drawer.points and drawer.lines
    finally:
        markers.close()
    assert not drawer.points and not drawer.lines

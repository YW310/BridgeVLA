"""Optional real-browser coverage for the online 3D/sensor diagnostics page."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import numpy as np
import pytest

from finetune.OHT.runtime.online_visualization import OnlineVisualizer, viewer_html
from finetune.OHT.runtime.sensor_visualization import raw_sensor_snapshot


def _sensor_snapshot(step):
    height, width = 96, 128
    yy, xx = np.indices((height, width))
    rgb = np.stack((xx * 2, yy * 2, (xx + yy + step * 20) % 256), axis=-1).astype(np.uint8)
    depth = (1 + .1 * np.sin(xx / 15) * np.cos(yy / 12)).astype(np.float32)
    depth[0, 0] = np.nan
    intrinsics = np.array([[100, 0, width / 2], [0, 100, height / 2], [0, 0, 1]], np.float32)
    names = ('global_left', 'wrist')
    config = dict(cameras={name: {} for name in names}, depth=dict(limits=[.001, 10.]))
    values = {}
    for index, name in enumerate(names):
        values.update({name + '_rgb': rgb.copy(), name + '_depth': depth.copy(),
                       name + '_intrinsics': intrinsics.copy(),
                       name + '_pose': np.array([index * .22, 0, 0, 0, 1, 0, 0], np.float32)})
    return raw_sensor_snapshot(values, config, np.array([.1, .02, .8, 0, 0, 0, 1]))


def _publish(writer, step):
    size = 48
    yy, xx = np.indices((size, size))
    rgb = np.stack((xx / size, yy / size, np.full_like(xx, .35, dtype=float)), axis=-1).astype(np.float32)
    heatmap = np.exp(-((xx - 22 - step) ** 2 + (yy - 20) ** 2) / 60).astype(np.float32)
    heatmap /= heatmap.sum()
    projection = np.zeros((3, 2, 4), np.float32)
    projection[:, 0, 0] = projection[:, 1, 1] = 15
    projection[:, :, 3] = 24
    stage = dict(rgb=np.repeat(rgb[None], 3, axis=0),
                 heatmap=np.repeat(heatmap[None], 3, axis=0), world_to_pixel=projection)
    payload = dict(stages={'mvt1': stage, 'mvt2': stage}, heatmap_waypoint_world=np.array([.12, .03, .9]))
    request = dict(request_id=f'prediction-{step}', episode_id='browser-test', step=step,
                   timestamp=step * .1, instruction='Disassemble the left assembly.', model_id='browser-model')
    action = np.array([.15 + step * .01, .02, .95, 0, 0, 0, 1, 1])
    response = writer.submit(payload, request, action, {}, {}, sensor_observation=_sensor_snapshot(step))
    assert response['status'] == 'queued'
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = writer.latest()
        if state['latest'] and state['latest']['entry_id'] == response['entry_id']:
            return state['latest']
        assert state['writer_status'] != 'error', state
        time.sleep(.01)
    raise AssertionError(f'Visualization export did not complete: {writer.latest()}')


def _server(writer):
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ('/', '/visualization'):
                body, content_type = viewer_html().encode('utf-8'), 'text/html; charset=utf-8'
            elif path == '/visualization/latest':
                body, content_type = json.dumps(writer.latest()).encode(), 'application/json'
            elif path.startswith('/visualization/files/'):
                parts = path.removeprefix('/visualization/files/').split('/')
                if len(parts) != 2:
                    self.send_error(404)
                    return
                target = (writer.root / parts[0] / parts[1]).resolve()
                if writer.root not in target.parents or not target.is_file():
                    self.send_error(404)
                    return
                body = target.read_bytes()
                content_type = {'.json': 'application/json', '.png': 'image/png',
                                '.npz': 'application/octet-stream'}.get(target.suffix, 'application/octet-stream')
            elif path == '/favicon.ico':
                self.send_response(204)
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _view_state(page):
    return page.evaluate('({yaw:viewer.yaw,pitch:viewer.pitch,distance:viewer.distance,target:viewer.target.slice()})')


def _drag(page, dx, dy, button='left'):
    canvas = page.locator('#scene-canvas')
    canvas.scroll_into_view_if_needed()
    box = canvas.bounding_box()
    x, y = box['x'] + box['width'] * .5, box['y'] + box['height'] * .5
    page.mouse.move(x, y)
    page.mouse.down(button=button)
    page.mouse.move(x + dx, y + dy, steps=8)
    page.mouse.up(button=button)


def test_browser_draws_sensor_geometry_and_preserves_interactive_view_across_updates(tmp_path):
    playwright = pytest.importorskip('playwright.sync_api')
    executable = os.environ.get('OHT_BROWSER_EXECUTABLE')
    if not executable or not Path(executable).is_file():
        pytest.skip('Set OHT_BROWSER_EXECUTABLE for optional browser smoke test')
    writer = OnlineVisualizer(tmp_path / 'online', every=1, keep=3)
    server = thread = None
    try:
        first = _publish(writer, 0)
        server, thread = _server(writer)
        origin = f'http://127.0.0.1:{server.server_port}'
        with playwright.sync_playwright() as p:
            # This task-owned temporary profile never reads the user's browser session.
            context = p.chromium.launch_persistent_context(str(tmp_path / 'browser-profile'),
                headless=True, executable_path=executable, viewport=dict(width=1360, height=1050),
                args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'], timeout=30_000)
            try:
                page = context.pages[0]
                page.set_default_timeout(15_000)
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(origin + '/visualization', wait_until='domcontentloaded')
                page.wait_for_function('entry => renderedEntry === entry', arg=first['entry_id'])
                page.wait_for_function('viewer.gl !== null && viewer.counts.points > 0 && !viewer.dirty')
                assert page.locator('#scene-empty').is_hidden()
                assert page.locator('#sensor-image img').evaluate('image => image.complete && image.naturalWidth === 256')
                assert page.locator('#images img').count() == 2
                assert page.locator('#sensor-projections .camera-info').count() == 2
                assert 'Original sensor RGB-D' in page.locator('#sensor-source').inner_text()
                pixels = page.evaluate('''() => {
                    viewer.draw(); const gl=viewer.gl, data=new Uint8Array(viewer.canvas.width*viewer.canvas.height*4);
                    gl.readPixels(0,0,viewer.canvas.width,viewer.canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,data);
                    let drawn=0;for(let i=0;i<data.length;i+=4)if(data[i]!==11||data[i+1]!==17||data[i+2]!==24)drawn++;
                    return {drawn,error:gl.getError(),points:viewer.counts.points};
                }''')
                assert pixels['drawn'] > 100 and pixels['error'] == 0
                assert pixels['points'] == first['scene']['point_count']
                original_view = _view_state(page)
                _drag(page, 75, 35)
                orbit = _view_state(page)
                assert orbit['yaw'] != original_view['yaw'] and orbit['pitch'] != original_view['pitch']
                _drag(page, 40, 20, button='right')
                panned = _view_state(page)
                assert panned['target'] != orbit['target'] and panned['yaw'] == orbit['yaw']
                page.mouse.wheel(0, 220)
                page.wait_for_function('distance => viewer.distance !== distance', arg=panned['distance'])
                zoomed = _view_state(page)
                assert zoomed['distance'] > panned['distance']
                point_count = page.evaluate('viewer.counts.points')
                cameras = page.locator('#camera-list input[type=checkbox]')
                cameras.first.uncheck()
                page.wait_for_function('count => viewer.counts.points < count', arg=point_count)
                assert page.evaluate('viewer.counts.points') == point_count // 2
                cameras.first.check()
                assert page.evaluate('viewer.counts.points') == point_count
                page.locator('#point-size').focus()
                page.locator('#point-size').press('End')
                assert page.evaluate('viewer.pointSize') == 8
                page.locator('#reset-view').click()
                reset = _view_state(page)
                for key in ('yaw', 'pitch', 'distance', 'target'):
                    assert reset[key] == pytest.approx(original_view[key])
                _drag(page, -55, 15)
                held_view = _view_state(page)
                cameras.first.uncheck()
                page.locator('#pause').click()
                assert page.locator('#pause').get_attribute('aria-pressed') == 'true'
                second = _publish(writer, 1)
                # Explicit refresh while paused is a no-op, even with a new server entry.
                page.evaluate('refresh()')
                page.wait_for_timeout(1100)
                assert page.evaluate('renderedEntry') == first['entry_id']
                page.locator('#pause').click()
                page.wait_for_function('entry => renderedEntry === entry', arg=second['entry_id'])
                assert page.locator('#pause').get_attribute('aria-pressed') == 'false'
                for key, value in held_view.items():
                    assert _view_state(page)[key] == pytest.approx(value)
                assert not page.locator('#camera-list input[type=checkbox]').first.is_checked()
                assert page.evaluate('viewer.pointSize') == 8
                assert page.locator('#sensor-image img').evaluate('image => image.complete && image.naturalWidth === 256')
                for width in (1360, 390):
                    page.set_viewport_size(dict(width=width, height=1050))
                    page.wait_for_function('!viewer.dirty')
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    screenshot = tmp_path / f'online-viewer-{width}.png'
                    page.screenshot(path=str(screenshot), full_page=True)
                    print(f'Online viewer screenshot: {screenshot}')
                assert not errors
            finally:
                context.close()
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        writer.close(wait=True)

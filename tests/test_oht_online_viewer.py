"""Exercise the real viewer JavaScript against browser interfaces, without a GPU."""
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from finetune.OHT.runtime.online_viewer import viewer_html


def _node():
    executable = shutil.which('node')
    if executable:
        return executable
    # Playwright bundles Node even when browser binaries have not been installed.
    candidate = Path(sys.executable).parent.parent / 'Lib/site-packages/playwright/driver/node.exe'
    if candidate.is_file():
        return str(candidate)
    pytest.skip('Node is required to exercise the browser viewer')


HARNESS = r'''
const assert=require('node:assert/strict');
class Node {
 constructor(tag='div'){this.tagName=tag;this.children=[];this.listeners={};this.textContent='';this.hidden=false;this.width=800;this.height=500;this.clientWidth=800;this.clientHeight=500;this.attributes={};}
 append(...children){this.children.push(...children);}
 replaceChildren(...children){this.children=children;this.textContent='';}
 addEventListener(name,callback){this.listeners[name]=callback;}
 setAttribute(name,value){this.attributes[name]=value;}
 setPointerCapture(){}
 getContext(){return fakeGL;}
}
const glCalls=[];
const fakeGL={VERTEX_SHADER:1,FRAGMENT_SHADER:2,COMPILE_STATUS:3,LINK_STATUS:4,DEPTH_TEST:5,ARRAY_BUFFER:6,STATIC_DRAW:7,FLOAT:8,POINTS:9,LINES:10,COLOR_BUFFER_BIT:16,DEPTH_BUFFER_BIT:32,
 createShader:()=>({}),shaderSource(){},compileShader(){},getShaderParameter:()=>true,getShaderInfoLog:()=>'',createProgram:()=>({}),attachShader(){},linkProgram(){},getProgramParameter:()=>true,getProgramInfoLog:()=>'',useProgram(){},getAttribLocation:()=>0,getUniformLocation:()=>({}),createBuffer:()=>({}),enable(value){glCalls.push('enable:'+value);},disable(value){glCalls.push('disable:'+value);},clearColor(){},bindBuffer(){},bufferData(){},viewport(){},clear(){},uniformMatrix4fv(){},vertexAttribPointer(){},enableVertexAttribArray(){},uniform1f(){},uniform1i(){},drawArrays(mode){glCalls.push('draw:'+mode);}};
const nodes=new Map();
global.document={getElementById(id){if(!nodes.has(id)){const node=new Node();node.parentElement=new Node();nodes.set(id,node);}return nodes.get(id);},createElement:tag=>new Node(tag),createTextNode:text=>({textContent:text})};
global.location={href:'http://localhost:8004/visualization',origin:'http://localhost:8004'};
global.devicePixelRatio=1.5;
global.ResizeObserver=class{constructor(callback){this.callback=callback;}observe(){}};
global.requestAnimationFrame=callback=>callback();
global.setInterval=()=>1;
const failedImages=new Set(),imageRequests=[];
global.Image=class extends Node{constructor(){super('img');}set src(value){this._src=value;imageRequests.push(value);queueMicrotask(()=>{if(failedImages.delete(value))this.onerror();else this.onload();});}get src(){return this._src;}};
const state={enabled:true,writer_status:'idle',dropped:0,latest:null};
const scenes=new Map(),failedScenes=new Set();let latestCalls=0,assetCalls=0,latestGate=null;
global.fetch=async path=>{if(path==='/visualization/latest'){latestCalls++;if(latestGate)await latestGate;return{ok:true,json:async()=>structuredClone(state)};}assetCalls++;if(failedScenes.delete(path))return{ok:false,status:503};return{ok:scenes.has(path),status:404,json:async()=>structuredClone(scenes.get(path))};};
const flush=async()=>{for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve));};
function sceneData(){return{version:1,coordinate_frame:'env_origin_relative_world',units:'metres',source:'raw_sensor',points:{positions:[[0,0,0],[1,0,0],[0,1,0]],colors:[[255,0,0],[0,255,0],[0,0,255]],camera_indices:[0,0,1]},cameras:[{index:0,name:'left',width:640,height:480,depth_kind:'z',world_from_optical:[[1,0,0,0],[0,1,0,-1],[0,0,1,1],[0,0,0,1]],frustum:[],projections:{policy:{status:'visible',pixel:[42,21]},heatmap:{status:'occluded',pixel:[40,20]}}},{index:1,name:'wrist',width:640,height:480,depth_kind:'z',frustum:[],projections:{policy:{status:'outside',pixel:[900,42]}}}],markers:[{name:'policy',color:'#ff5965',position:[.2,.3,.4]},{name:'measured',color:'#63e59b',position:[.1,.1,.1]}]};}
function entry(id,step=0){const base='/visualization/files/'+id;const scene=sceneData();scenes.set('http://localhost:8004'+base+'/scene.json',scene);return{entry_id:id,episode_id:'episode-0',step,task:'disassemble_left',instruction:'<script>unsafe()</script>',images:{mvt1:base+'/mvt1.png',mvt2:base+'/mvt2.png'},prediction_url:base+'/prediction.json',scene:{url:base+'/scene.json',point_count:3,source:'raw_sensor'},sensors:{source:'raw_sensor',montage_url:base+'/sensors.png',arrays_url:base+'/sensors.npz',cameras:scene.cameras}};}
'''


CASES = {
    '3d_controls': r'''
 viewer.setScene(sceneData());assert.equal(viewer.counts.points,3);assert.equal(viewer.counts.markers,3);
 const canvas=$('scene-canvas'),event=(extra={})=>({pointerId:1,clientX:100,clientY:100,button:0,shiftKey:false,preventDefault(){},...extra});
 const yaw=viewer.yaw;canvas.listeners.pointerdown(event());canvas.listeners.pointermove(event({clientX:130,clientY:120}));assert.notEqual(viewer.yaw,yaw);canvas.listeners.pointerup(event());
 const target=viewer.target.slice();canvas.listeners.pointerdown(event({button:2}));canvas.listeners.pointermove(event({clientX:125,clientY:105}));assert.notDeepEqual(viewer.target,target);canvas.listeners.pointerup(event());
 const shifted=viewer.target.slice();canvas.listeners.pointerdown(event({shiftKey:true}));canvas.listeners.pointermove(event({clientX:110,shiftKey:true}));assert.notDeepEqual(viewer.target,shifted);canvas.listeners.pointercancel(event());
 const distance=viewer.distance;canvas.listeners.wheel({deltaY:-100,preventDefault(){}});assert.ok(viewer.distance<distance);
 const original=viewer.pointSize;$('point-size').listeners.input({target:{value:'5'}});assert.notEqual(viewer.pointSize,original);assert.equal(viewer.pointSize,5);
 cameraControls(viewer.scene.cameras);const checkbox=$('camera-list').children[0].children[0];checkbox.checked=false;checkbox.listeners.change();assert.equal(viewer.counts.points,1);
 $('reset-view').listeners.click();assert.equal(viewer.yaw,-.8);assert.equal(viewer.pitch,.5);assert.ok(viewer.canvas.width>viewer.canvas.clientWidth);
 viewer.enabled.set('left',true);viewer.scene.cameras[0].frustum=Array.from({length:8},(_,i)=>[i%2,(i>>1)%2,(i>>2)%2]);viewer.upload();assert.equal(viewer.counts.lines,42);
 viewer.scene.cameras[0].frustum=viewer.scene.cameras[0].frustum.slice(0,4);viewer.upload();assert.equal(viewer.counts.lines,34);
 glCalls.length=0;viewer.draw();assert.deepEqual(glCalls,['draw:9','draw:10','disable:5','draw:9','enable:5']);
 canvas.listeners.webglcontextlost({preventDefault(){}});assert.equal(viewer.gl,null);assert.equal($('scene-empty').hidden,false);assert.ok(viewer.unavailable.includes('context was lost'));
 ''',
    'atomic_snapshots_and_view_preservation': r'''
 state.latest=entry('first',1);await refresh();assert.equal(current,'first');assert.equal(renderedEntry,'first');assert.equal($('sensor-image').children[0].src,'http://localhost:8004/visualization/files/first/sensors.png');assert.equal($('images').children.length,2);assert.equal($('scene-empty').hidden,true);
 assert.ok($('metadata').textContent.includes('<script>unsafe()</script>'));assert.ok($('sensor-projections').children[0].children[2].children[1].textContent.includes('visible'));
 viewer.yaw=1.7;viewer.target=[4,5,6];viewer.distance=9;const beforeCalls=assetCalls;await refresh();assert.equal(assetCalls,beforeCalls);
 state.latest=entry('second',2);await refresh();assert.equal(current,'second');assert.equal(viewer.yaw,1.7);assert.deepEqual(viewer.target,[4,5,6]);assert.equal(viewer.distance,9);assert.ok($('snapshot').textContent.includes('step 2'));assert.ok($('sensor-image').children[0].src.includes('/second/'));assert.ok($('images').children[0].children[1].children[0].src.includes('/second/'));
 ''',
    'asset_failures_retry_without_blocking_heatmaps': r'''
 state.latest=entry('retry',3);failedImages.add('http://localhost:8004/visualization/files/retry/mvt1.png');failedScenes.add('http://localhost:8004/visualization/files/retry/scene.json');await refresh();assert.equal(current,null);assert.equal(renderedEntry,'retry');assert.equal(viewer.scene,null);assert.ok($('images').children[1].children[1].children[0].src.includes('mvt2.png'));assert.ok($('status').textContent.includes('retrying'));await refresh();assert.equal(current,'retry');assert.ok(viewer.scene);assert.equal($('scene-empty').hidden,true);
 state.latest=entry('prepared',4);state.latest.scene=null;state.latest.sensors=null;await refresh();assert.equal(current,'prepared');assert.equal(viewer.scene,null);assert.equal($('scene-empty').hidden,false);assert.equal($('images').children.length,2);assert.equal($('sensor-projections').children.length,0);assert.equal($('sensor-links').children.length,0);
 state.latest=entry('sensor-error',5);state.latest.sensors.error='Sensor camera unavailable';delete state.latest.sensors.montage_url;await refresh();assert.equal($('sensor-error').textContent,'Sensor camera unavailable');assert.equal($('sensor-image').children[0].textContent,'Sensor camera unavailable');assert.equal($('images').children.length,2);
 assert.ok(sourceLabel('prepared_observation').includes('not original sensor pixels'));
 assert.throws(()=>assetURL('https://untrusted.example/a.png'),/Unexpected/);assert.throws(()=>assetURL('/etc/passwd'),/Unexpected/);
 ''',
    'serialized_fetch_and_pause_freezes_snapshot': r'''
 state.latest=entry('before');await refresh();let release;latestGate=new Promise(resolve=>{release=resolve;});state.latest=entry('during',5);const count=latestCalls,pending=refresh();await refresh();assert.equal(latestCalls,count+1);$('pause').listeners.click();assert.equal(paused,true);release();latestGate=null;await pending;assert.equal(renderedEntry,'before');assert.ok($('status').textContent.includes('Paused'));await refresh();assert.equal(renderedEntry,'before');$('pause').listeners.click();await flush();assert.equal(paused,false);assert.equal(renderedEntry,'during');assert.equal($('pause').attributes['aria-pressed'],'false');
 ''',
    'missing_webgl_still_shows_predictions': r'''
 const noGL=new Node('canvas');noGL.parentElement=new Node();noGL.getContext=()=>null;const fallback=new SceneViewer(noGL);fallback.setScene(sceneData());assert.equal(fallback.gl,null);assert.ok(fallback.unavailable.includes('WebGL'));assert.ok(fallback.unavailable.includes('Camera images'));
 state.latest=entry('normal');await refresh();assert.equal(current,'normal');assert.equal($('images').children.length,2);
 ''',
}


@pytest.mark.parametrize('case', CASES)
def test_viewer_browser_behaviour(tmp_path, case):
    script = re.search(r'<script>(.*?)</script>', viewer_html(), re.S).group(1)
    path = tmp_path / 'viewer-test.js'
    path.write_text(HARNESS + '\n' + script + '\n' +
                    '(async()=>{await flush();\n' + CASES[case] +
                    '\n})().catch(error=>{console.error(error);process.exitCode=1;});', encoding='utf-8')
    result = subprocess.run([_node(), str(path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_viewer_has_no_external_dependencies_or_executable_metadata():
    page = viewer_html()
    assert 'innerHTML' not in page
    assert 'textContent' in page
    assert not re.search(r'<(?:script|link)[^>]+(?:src|href)=', page)
    assert 'ground-truth targets' in page
    assert 'Waypoints stay visible through the cloud' in page
    assert "fetch('/visualization/latest'" in page
    assert 'setInterval(refresh,1000)' in page
    assert 'server:location.origin' in page and 'task:item.task' in page
    assert 'state.capture_error' in page and 'state.reason' in page

"""Dependency-free browser diagnostics for sampled online policy predictions."""


def viewer_html():
    """Return a self-contained viewer; dynamic values are always text nodes."""
    return r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BridgeVLA · OHT live predictions</title>
<style>
:root{color-scheme:dark;font-family:Inter,Segoe UI,Arial,sans-serif;background:#101419;color:#dce5ed;font-size:14px}
*{box-sizing:border-box}body{margin:0}header{padding:22px 28px 18px;border-bottom:1px solid #2a3540;background:#151c24}
h1{font-size:23px;margin:0 0 8px;font-weight:600;letter-spacing:-.4px}h2{font-size:17px;margin:0;font-weight:600}
h3{font-size:14px;margin:0 0 8px}p{line-height:1.5;margin:7px 0;color:#9dafbf}a{color:#87c7ff;text-underline-offset:3px}
main{max-width:1720px;padding:20px 28px 36px;margin:auto}.toolbar,.legend,.section-title{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.toolbar{margin-top:14px}.section-title{justify-content:space-between;margin-bottom:13px}.section-title span{color:#8c9fb0;font-size:12px}
button,select,input{font:inherit}button{border:1px solid #415469;background:#263746;color:#eff6fc;padding:7px 12px;border-radius:6px;cursor:pointer}
button:hover{background:#344d63}button:focus-visible,a:focus-visible,input:focus-visible{outline:2px solid #82c9ff;outline-offset:3px}
button[aria-pressed=true]{background:#594823;border-color:#a2864d}input[type=checkbox]{accent-color:#8dcaff}input[type=range]{width:100px;accent-color:#8dcaff}
.badge{padding:3px 8px;border:1px solid #3b4a59;border-radius:4px;font:12px ui-monospace,Consolas,monospace;overflow-wrap:anywhere}
#status{font-size:12px;color:#a6b7c7;flex:1;min-width:260px}section.panel{background:#171f28;border:1px solid #2b3948;border-radius:9px;padding:18px;margin-bottom:18px;overflow:hidden}
.workspace{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:18px}.viewport{position:relative;min-width:0;height:520px;border:1px solid #314252;border-radius:6px;background:#0b1118;overflow:hidden}
canvas{display:block;width:100%;height:100%;touch-action:none}.viewport-note{position:absolute;bottom:12px;left:12px;pointer-events:none;background:#0b1118bd;border-radius:4px;padding:6px 9px;color:#9fb4c6;font-size:12px}
.empty{display:grid;place-items:center;text-align:center;height:100%;padding:24px;color:#9dafbf}.empty[hidden]{display:none}.viewport .empty{position:absolute;inset:0;background:#0b1118ed}
.sidebar{min-width:0}.sidebar h3{margin-top:16px}.sidebar h3:first-child{margin-top:0}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.camera-list label{display:flex;align-items:center;gap:8px;padding:6px 0;overflow-wrap:anywhere}.swatch{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px}
.legend{font-size:12px;gap:16px;margin-top:14px}.hint{font-size:12px;line-height:1.6;color:#8ea2b4}.scene-stats{font:12px/1.7 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;color:#a6b7c7}
.notice{color:#e6bb78;font-size:12px;line-height:1.5}.notice:empty{display:none}.sensor-grid{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:18px;align-items:start}
.image-box{border:1px solid #2d3c4b;border-radius:6px;overflow:auto;background:#10161e;min-height:80px}.image-box img{display:block;width:100%;height:auto}
.projection-list{display:grid;gap:9px}.camera-info{border:1px solid #314252;border-radius:5px;padding:10px;font-size:12px;overflow-wrap:anywhere}.camera-info strong{font-weight:600;color:#dce5ed}.projection-row{display:flex;justify-content:space-between;gap:10px;color:#9dafbf;margin-top:6px}
.visible{color:#88c6a2}.occluded,.outside,.behind,.invalid_depth{color:#e6bb78}.heatmaps{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.heatmap-card{min-width:0}.heatmap-card .image-box{margin-top:10px}
details{border-top:1px solid #293947;margin-top:16px;padding-top:12px}summary{cursor:pointer;color:#9cb5ca}pre{font:12px/1.6 ui-monospace,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;color:#a3b8cb;margin:12px 0 0}
.snapshot{color:#b7d5ef;font:12px ui-monospace,Consolas,monospace}.download-links{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin-top:12px}.muted{font-size:12px;color:#859bad}
@media(min-width:1500px){.viewport{height:600px}}@media(max-width:1050px){.workspace{grid-template-columns:minmax(0,1fr) 220px}.sensor-grid{grid-template-columns:1fr}.projection-list{grid-template-columns:repeat(2,minmax(0,1fr))}.heatmaps{grid-template-columns:1fr}}
@media(max-width:680px){header{padding:18px}main{padding:16px 12px}section.panel{padding:14px}.workspace{grid-template-columns:1fr}.viewport{height:380px}.sidebar{display:flex;gap:18px;flex-wrap:wrap}.sidebar>div{flex:1;min-width:150px}.sidebar h3{margin-top:0}.projection-list{grid-template-columns:1fr}.viewport-note{font-size:10px}.section-title{align-items:flex-start}.toolbar{gap:8px}}
</style></head><body>
<header><h1>BridgeVLA <span style="color:#728da6">/</span> OHT live predictions</h1>
<p>One inference snapshot: reconstructed sensor geometry, camera images and virtual-view heatmaps. Predictions only; no ground-truth targets.</p>
<p>Chunk snapshots: purple is the predicted path; red is the advertised execution prefix. Client overrides and actual executed motion are recorded in client logs.</p>
<div class="toolbar"><button id="pause" type="button" aria-pressed="false">Pause updates</button><span id="snapshot" class="badge">No snapshot yet</span><span id="status" role="status">Waiting for the next sampled inference...</span></div></header>
<main>
<section class="panel"><div class="section-title"><h2>Interactive 3D scene</h2><span id="scene-source">Sensor RGB-D · world coordinates · metres</span></div>
<div class="workspace"><div class="viewport" id="viewport"><canvas id="scene-canvas" aria-label="Interactive sensor point cloud and predicted waypoints" tabindex="0"></canvas><div id="scene-empty" class="empty">Waiting for sensor geometry...</div><div class="viewport-note">Drag: orbit · Right / Shift-drag: pan · Wheel / pinch: zoom</div></div>
<aside class="sidebar"><div><h3>View</h3><div class="controls"><button id="reset-view" type="button">Reset view</button><label>Point size <input id="point-size" type="range" min="1" max="8" step=".5" value="2.5"></label></div><p class="hint">View position stays fixed across updates. Reset fits the current point cloud.</p><h3>Camera layers</h3><div id="camera-list" class="camera-list muted">No cameras in this snapshot</div></div><div><h3>Geometry</h3><div id="scene-stats" class="scene-stats">No scene loaded</div><p id="scene-error" class="notice"></p><p class="hint">RGB-D is reconstructed from the received sensor frame. Missing and occluded surfaces remain unobserved.</p></div></aside></div>
<div class="legend"><span><i class="swatch" style="background:#ff5965"></i>Policy waypoint</span><span><i class="swatch" style="background:#33d5ef"></i>Heatmap waypoint</span><span><i class="swatch" style="background:#63e59b"></i>Measured TCP</span><span><i class="swatch" style="background:#abc0d5"></i>Camera frustums</span><span>X red · Y green · Z blue</span></div><p class="hint">Waypoints stay visible through the cloud. Their visibility here does not indicate that a camera can observe them.</p></section>
<section class="panel"><div class="section-title"><h2>Original sensor images + waypoints</h2><span id="sensor-source">No sensor snapshot yet</span></div><p class="hint">Original camera pixels, with the same 3D waypoints projected onto each image. Projection status uses the measured depth when available.</p>
<div class="sensor-grid"><div id="sensor-image" class="image-box"><div class="empty">Waiting for raw RGB-D input...</div></div><div id="sensor-projections" class="projection-list"></div></div><p id="sensor-error" class="notice"></p><div id="sensor-links" class="download-links"></div></section>
<section class="panel"><div class="section-title"><h2>Virtual-view predictions</h2><span>Same policy forward · coarse / refine</span></div><p class="hint">Red: policy TCP; cyan: heatmap TCP. Heatmaps are auxiliary for relative EEF and joint policies. Joint actions have no inferred policy TCP.</p><div id="images" class="heatmaps"><p class="muted">Waiting for prediction heatmaps...</p></div><div id="prediction-links" class="download-links"></div><details><summary>Snapshot metadata</summary><pre id="metadata"></pre></details></section>
</main>
<script>
'use strict';
const $=id=>document.getElementById(id);
function element(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
function sourceLabel(source){return source==='raw_sensor'?'Original sensor RGB-D':source==='prepared_observation'?'Prepared observation · resized policy input, not original sensor pixels':source||'Source unavailable';}
function assetURL(path){const url=new URL(path,location.href);if(url.origin!==location.origin||!url.pathname.startsWith('/visualization/files/'))throw new Error('Unexpected visualization asset URL');return url.href;}
function download(path,label){const link=element('a',label);link.href=assetURL(path);link.target='_blank';link.rel='noopener';return link;}
async function jsonAsset(path){const response=await fetch(assetURL(path),{cache:'no-store'});if(!response.ok)throw new Error('HTTP '+response.status);return response.json();}
function imageAsset(path,alt){return new Promise((resolve,reject)=>{const image=new Image();image.alt=alt;image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('Image unavailable: '+alt));image.src=assetURL(path);});}
const add=(a,b)=>a.map((v,i)=>v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]),scale=(a,s)=>a.map(v=>v*s);
const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>Math.hypot(...a),unit=a=>scale(a,1/(norm(a)||1)),finite3=a=>Array.isArray(a)&&a.length===3&&a.every(Number.isFinite);
function multiply(a,b){const result=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)result[c*4+r]+=a[k*4+r]*b[c*4+k];return result;}
function viewMatrix(eye,target){const z=unit(sub(eye,target)),x=unit(cross([0,0,1],z)),y=cross(z,x);return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]);}
function perspective(aspect,near,far){const f=1/Math.tan(Math.PI/7);return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);}
function rgb(hex){return [1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)/255);}
class SceneViewer{
 constructor(canvas){this.canvas=canvas;this.scene=null;this.enabled=new Map();this.target=[0,0,0];this.distance=2;this.radius=1;this.yaw=-.8;this.pitch=.5;this.initialized=false;this.pointSize=2.5;this.pointers=new Map();this.dirty=false;
  this.gl=canvas.getContext('webgl',{alpha:false,antialias:true});
  if(this.gl){try{this.setupGL();}catch(error){this.gl=null;this.unavailable=String(error);}}else this.unavailable='WebGL is unavailable. Camera images, projection statuses and downloadable geometry remain available.';
  canvas.addEventListener('contextmenu',event=>event.preventDefault());canvas.addEventListener('pointerdown',event=>this.down(event));canvas.addEventListener('pointermove',event=>this.move(event));
  for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,event=>this.pointers.delete(event.pointerId));
  canvas.addEventListener('wheel',event=>{event.preventDefault();this.distance=Math.max(this.radius*.025,Math.min(this.radius*100,this.distance*Math.exp(event.deltaY*.001)));this.drawSoon();},{passive:false});
  this.observer=new ResizeObserver(()=>this.drawSoon());this.observer.observe(canvas.parentElement);
 }
 setupGL(){const gl=this.gl;const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;};
  const vs=shader(gl.VERTEX_SHADER,'attribute vec3 aPosition;attribute vec3 aColor;uniform mat4 uMatrix;uniform float uSize;varying vec3 vColor;void main(){gl_Position=uMatrix*vec4(aPosition,1.0);gl_PointSize=uSize;vColor=aColor;}');
  const fs=shader(gl.FRAGMENT_SHADER,'precision mediump float;varying vec3 vColor;uniform bool uRound;void main(){if(uRound&&distance(gl_PointCoord,vec2(.5))>.5)discard;gl_FragColor=vec4(vColor,1.0);}');
  this.program=gl.createProgram();gl.attachShader(this.program,vs);gl.attachShader(this.program,fs);gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(this.program));
  gl.useProgram(this.program);this.locations={position:gl.getAttribLocation(this.program,'aPosition'),color:gl.getAttribLocation(this.program,'aColor'),matrix:gl.getUniformLocation(this.program,'uMatrix'),size:gl.getUniformLocation(this.program,'uSize'),round:gl.getUniformLocation(this.program,'uRound')};
  this.buffers={points:gl.createBuffer(),lines:gl.createBuffer(),markers:gl.createBuffer()};this.counts={points:0,lines:0,markers:0};gl.enable(gl.DEPTH_TEST);gl.clearColor(.043,.067,.094,1);
  this.canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();this.gl=null;this.unavailable='3D graphics context was lost. Reload this page to restore the interactive view.';$('scene-empty').textContent=this.unavailable;$('scene-empty').hidden=false;});
 }
 down(event){event.preventDefault();this.canvas.setPointerCapture(event.pointerId);this.pointers.set(event.pointerId,{x:event.clientX,y:event.clientY,pan:event.button===2||event.shiftKey});}
 move(event){const pointer=this.pointers.get(event.pointerId);if(!pointer)return;const before=[...this.pointers.values()].map(p=>({...p}));const dx=event.clientX-pointer.x,dy=event.clientY-pointer.y;pointer.x=event.clientX;pointer.y=event.clientY;
  if(this.pointers.size===2){const after=[...this.pointers.values()];const oldDistance=Math.hypot(before[0].x-before[1].x,before[0].y-before[1].y),newDistance=Math.hypot(after[0].x-after[1].x,after[0].y-after[1].y);if(newDistance>0)this.distance=Math.max(this.radius*.025,Math.min(this.radius*100,this.distance*oldDistance/newDistance));this.pan(dx*.5,dy*.5);}
  else if(pointer.pan||event.shiftKey)this.pan(dx,dy);else{this.yaw-=dx*.007;this.pitch=Math.max(-1.5,Math.min(1.5,this.pitch+dy*.007));}this.drawSoon();
 }
 eye(){return add(this.target,scale([Math.cos(this.pitch)*Math.cos(this.yaw),Math.cos(this.pitch)*Math.sin(this.yaw),Math.sin(this.pitch)],this.distance));}
 pan(dx,dy){const forward=unit(sub(this.target,this.eye())),right=unit(cross(forward,[0,0,1])),up=cross(right,forward),factor=this.distance*.9/Math.max(1,this.canvas.clientHeight);this.target=add(this.target,add(scale(right,-dx*factor),scale(up,dy*factor)));}
 reset(){if(!this.scene)return;const positions=this.scene.points?.positions||[],markers=(this.scene.markers||[]).map(m=>m.position),all=positions.concat(markers).filter(finite3);let lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(const p of all)for(let i=0;i<3;i++){lo[i]=Math.min(lo[i],p[i]);hi[i]=Math.max(hi[i],p[i]);}if(!all.length){lo=[-.5,-.5,-.5];hi=[.5,.5,.5];}this.target=scale(add(lo,hi),.5);this.radius=Math.max(.05,norm(sub(hi,lo))*.5);this.distance=this.radius*2.8;this.yaw=-.8;this.pitch=.5;this.initialized=true;this.upload();this.drawSoon();}
 setScene(scene){this.scene=scene;if(!scene){this.counts={points:0,lines:0,markers:0};this.drawSoon();return;}for(const camera of scene.cameras||[])if(!this.enabled.has(camera.name))this.enabled.set(camera.name,true);if(!this.initialized)this.reset();else{this.upload();this.drawSoon();}}
 upload(){if(!this.gl||!this.scene)return;const gl=this.gl,scene=this.scene,points=[],lines=[],markers=[],positions=scene.points?.positions||[],colors=scene.points?.colors||[],indices=scene.points?.camera_indices||[],cameras=scene.cameras||[],enabled=index=>!cameras[index]||this.enabled.get(cameras[index].name)!==false;
  const vertex=(array,p,color)=>{if(finite3(p))array.push(...p,...color);},line=(a,b,color)=>{vertex(lines,a,color);vertex(lines,b,color);};
  for(let i=0;i<positions.length;i++){if(!enabled(indices[i]))continue;const c=colors[i]||[150,180,200];vertex(points,positions[i],c.map(v=>Math.min(1,Math.max(0,v/255))));}
  const axisLength=Math.max(.04,this.radius*.22);for(const [axis,color] of [[[1,0,0],[.95,.28,.32]],[[0,1,0],[.28,.87,.5]],[[0,0,1],[.3,.55,1]]])line([0,0,0],scale(axis,axisLength),color);
  for(const camera of cameras){if(this.enabled.get(camera.name)===false)continue;const matrix=camera.world_from_optical,position=matrix&&matrix.length===4?[matrix[0][3],matrix[1][3],matrix[2][3]]:camera.position,frustum=camera.frustum||[],color=[.48,.61,.74];if(finite3(position))vertex(markers,position,color);
   if(frustum.length===8){for(const [a,b] of [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])if(finite3(frustum[a])&&finite3(frustum[b]))line(frustum[a],frustum[b],color);}
   else if(finite3(position)&&frustum.length>=4){for(let i=0;i<4;i++){line(position,frustum[i],color);line(frustum[i],frustum[(i+1)%4],color);}}
  }
  for(const marker of scene.markers||[]){if(!finite3(marker.position))continue;const color=rgb(marker.color||'#eeeeee');vertex(markers,marker.position,color);const length=this.radius*.018;for(let i=0;i<3;i++){const delta=[0,0,0];delta[i]=length;line(sub(marker.position,delta),add(marker.position,delta),color);}}
  for(const trajectory of scene.trajectories||[]){const positions=trajectory.positions||[],color=rgb(trajectory.color||'#d4a8ff');for(let i=1;i<positions.length;i++)if(finite3(positions[i-1])&&finite3(positions[i]))line(positions[i-1],positions[i],color);}
  for(const [name,data] of Object.entries({points,lines,markers})){gl.bindBuffer(gl.ARRAY_BUFFER,this.buffers[name]);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.STATIC_DRAW);this.counts[name]=data.length/6;}
 }
 drawSoon(){if(this.dirty)return;this.dirty=true;requestAnimationFrame(()=>{this.dirty=false;this.draw();});}
 draw(){if(!this.gl)return;const gl=this.gl,ratio=Math.min(devicePixelRatio||1,2),width=Math.max(1,Math.round(this.canvas.clientWidth*ratio)),height=Math.max(1,Math.round(this.canvas.clientHeight*ratio));if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}gl.viewport(0,0,width,height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);if(!this.scene)return;gl.useProgram(this.program);const matrix=multiply(perspective(width/height,Math.max(.0001,this.distance/10000),Math.max(100,this.distance*100)),viewMatrix(this.eye(),this.target));gl.uniformMatrix4fv(this.locations.matrix,false,matrix);
  for(const [name,mode,size] of [['points',gl.POINTS,this.pointSize*ratio],['lines',gl.LINES,1],['markers',gl.POINTS,10*ratio]]){gl.bindBuffer(gl.ARRAY_BUFFER,this.buffers[name]);gl.vertexAttribPointer(this.locations.position,3,gl.FLOAT,false,24,0);gl.enableVertexAttribArray(this.locations.position);gl.vertexAttribPointer(this.locations.color,3,gl.FLOAT,false,24,12);gl.enableVertexAttribArray(this.locations.color);gl.uniform1f(this.locations.size,size);gl.uniform1i(this.locations.round,mode===gl.POINTS?1:0);if(name==='markers')gl.disable(gl.DEPTH_TEST);gl.drawArrays(mode,0,this.counts[name]);if(name==='markers')gl.enable(gl.DEPTH_TEST);}
 }
}
const viewer=new SceneViewer($('scene-canvas'));
$('reset-view').addEventListener('click',()=>viewer.reset());$('point-size').addEventListener('input',event=>{viewer.pointSize=Number(event.target.value);viewer.drawSoon();});
function cameraControls(cameras){const root=$('camera-list');root.replaceChildren();if(!cameras.length){root.textContent='No cameras in this snapshot';return;}for(const camera of cameras){const label=element('label'),input=element('input');input.type='checkbox';input.checked=viewer.enabled.get(camera.name)!==false;input.addEventListener('change',()=>{viewer.enabled.set(camera.name,input.checked);viewer.upload();viewer.drawSoon();});label.append(input,document.createTextNode(camera.name));root.append(label);}}
function projectionStatus(camera){const card=element('div',undefined,'camera-info');card.append(element('strong',camera.name),element('div',camera.width+' × '+camera.height+' · '+(camera.depth_kind||'depth'),'muted'));for(const [name,label] of [['policy','Policy'],['heatmap','Heatmap'],['measured','Measured TCP']]){const data=camera.projections?.[name],row=element('div',undefined,'projection-row');row.append(element('span',label));const status=data?.status||'unavailable',pixel=data?.pixel;row.append(element('span',status+(pixel?' · '+pixel.map(v=>Number(v).toFixed(1)).join(', '):''),status));card.append(row);}return card;}
let current=null,renderedEntry=null,busy=false,paused=false,generation=0;
$('pause').addEventListener('click',()=>{paused=!paused;generation++;$('pause').textContent=paused?'Resume updates':'Pause updates';$('pause').setAttribute('aria-pressed',String(paused));if(paused)$('status').textContent='Paused · the displayed snapshot is frozen';else refresh();});
async function prepare(item){const result={item,images:[],scene:null,sensor:null,errors:[],sceneError:null,sensorError:item.sensors?.error||null};
 const jobs=Object.entries(item.images||{}).map(async([name,path])=>{try{result.images.push({name,image:await imageAsset(path,name+' prediction heatmaps')});}catch(error){result.images.push({name,error:String(error)});result.errors.push(String(error));}});
 if(item.scene?.url)jobs.push(jsonAsset(item.scene.url).then(scene=>{if(!scene||!scene.points||!Array.isArray(scene.points.positions))throw new Error('Invalid scene geometry');result.scene=scene;}).catch(error=>{result.sceneError=String(error);result.errors.push(String(error));}));
 if(item.sensors?.montage_url)jobs.push(imageAsset(item.sensors.montage_url,'Original sensor images with projected waypoints').then(image=>{result.sensor=image;}).catch(error=>{result.sensorError=String(error);result.errors.push(String(error));}));
 await Promise.all(jobs);return result;
}
function commit(snapshot){const {item,scene}=snapshot;renderedEntry=item.entry_id;$('snapshot').textContent='Episode '+item.episode_id+' · step '+item.step+' · '+item.entry_id.slice(0,8);
 $('metadata').textContent=JSON.stringify({server:location.origin,model:item.model_id,task:item.task,request:item.request_id,episode:item.episode_id,step:item.step,timestamp:item.timestamp,instruction:item.instruction,action:item.predicted_action,policy_tcp:item.policy_tcp_world,heatmap_tcp:item.heatmap_waypoint_world,heatmap_is_auxiliary:item.heatmap_is_auxiliary,sensor_source:item.sensors?.source,entry_id:item.entry_id},null,2);
 const imageRoot=$('images');imageRoot.replaceChildren();for(const name of ['mvt1','mvt2',...snapshot.images.map(image=>image.name).filter(name=>name!=='mvt1'&&name!=='mvt2')]){const part=snapshot.images.find(image=>image.name===name);if(!part)continue;const card=element('div',undefined,'heatmap-card'),box=element('div',undefined,'image-box');card.append(element('h3',name==='mvt1'?'Coarse stage':name==='mvt2'?'Refine stage':name));box.append(part.image||element('p',part.error,'notice'));card.append(box);if(item.stages?.[name]?.arrays)card.append(download(item.stages[name].arrays,'Download stage arrays (.npz)'));imageRoot.append(card);}if(!snapshot.images.length)imageRoot.append(element('p','No virtual-view images in this snapshot.','muted'));
 $('prediction-links').replaceChildren();if(item.prediction_url)$('prediction-links').append(download(item.prediction_url,'Prediction JSON'));
 viewer.setScene(scene);const noScene=!scene||!viewer.gl;$('scene-empty').hidden=!noScene;$('scene-empty').textContent=!viewer.gl?viewer.unavailable:snapshot.sceneError||'No sensor geometry for this snapshot. Prepared observations can still show virtual-view heatmaps.';
 $('scene-source').textContent=scene?sourceLabel(scene.source)+' · '+(scene.coordinate_frame||'world')+' · '+(scene.units||'metres'):'Geometry unavailable for this snapshot';$('scene-error').textContent=snapshot.sceneError||'';
 $('scene-stats').textContent=scene?((scene.points.positions.length).toLocaleString()+' points\n'+(scene.cameras||[]).length+' cameras\n'+(scene.markers||[]).map(marker=>marker.name+': '+marker.position.map(v=>Number(v).toFixed(3)).join(', ')).join('\n')):'No scene loaded';cameraControls(scene?.cameras||[]);
 $('sensor-source').textContent=sourceLabel(item.sensors?.source);const sensorBox=$('sensor-image');sensorBox.replaceChildren();sensorBox.append(snapshot.sensor||element('div',snapshot.sensorError||'Original sensor frames were not included in this inference snapshot.','empty'));
 $('sensor-error').textContent=snapshot.sensorError||'';$('sensor-projections').replaceChildren(...(item.sensors?.cameras||scene?.cameras||[]).map(projectionStatus));$('sensor-links').replaceChildren();if(item.sensors?.arrays_url)$('sensor-links').append(download(item.sensors.arrays_url,'Original RGB-D and calibration (.npz)'));if(item.scene?.url)$('sensor-links').append(download(item.scene.url,'3D scene (.json)'));
 current=snapshot.errors.length?null:item.entry_id;
}
async function refresh(){if(busy||paused)return;busy=true;const token=generation;try{
 const response=await fetch('/visualization/latest',{cache:'no-store'});if(!response.ok)throw new Error('HTTP '+response.status);const state=await response.json();if(paused||token!==generation)return;
 $('status').textContent=state.enabled?'Writer: '+state.writer_status+' | dropped: '+state.dropped+(state.error?' | '+state.error:'')+(state.capture_error?' | capture: '+state.capture_error:''):'Visualization is disabled'+(state.reason?' | '+state.reason:'');
 const item=state.latest;if(!item||item.entry_id===current)return;const snapshot=await prepare(item);if(paused||token!==generation)return;commit(snapshot);if(snapshot.errors.length)$('status').textContent+=' | Some snapshot assets unavailable; retrying automatically';
 }catch(error){if(!paused&&token===generation)$('status').textContent='Update failed: '+String(error)+(renderedEntry?' · keeping previous snapshot':'')+' · retrying automatically';}finally{busy=false;}}
refresh();setInterval(refresh,1000);
</script></body></html>'''

"""Update the supplied weekly report and package its local evidence/media."""
import base64
import json
import mimetypes
import re
import shutil
import zipfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path('/home/yiwei/Downloads/Weekly_Report_2026-11-23.html')
DEST = ROOT / 'reports/Weekly_Report_2026-11-23'
if not SOURCE.is_file():
    SOURCE = DEST / 'Weekly_Report_2026-11-23.html'
DEST.mkdir(parents=True, exist_ok=True)
for folder in ('media', 'evidence'):
    (DEST / folder).mkdir(exist_ok=True)

files = {
    'media/episode_31_prediction_errors.png': 'logs/oht-epoch30-full-episodes-17-31-20261120-ep000031.png',
    'evidence/prediction_analysis.json': 'logs/oht-epoch30-full-episodes-17-31-20261120.analysis.json',
    'evidence/gt_tracking_comparison.json': 'logs/oht-gt-chunk-ep0-velocity-recorded-20261120/velocity_fix_comparison.json',
    'evidence/recorded_tcp_feedback.json': 'logs/oht-epoch30-recorded-feedback-ep31-20261120/result.json',
    'evidence/prediction_video_manifest.json': 'logs/oht-full-prediction-videos-20261123/manifest.json',
}
for ep in (31,):
    stem = f'logs/oht-full-prediction-videos-20261123/episode_{ep:06d}_full_model_prediction'
    files.update({f'media/episode_{ep}_full_prediction.gif': stem + '_preview.gif'})
for target, source in files.items():
    if (ROOT / source).is_file():
        shutil.copy2(ROOT / source, DEST / target)
    elif not (DEST / target).is_file():
        raise FileNotFoundError(f'Missing report asset: {target}')
# Remove only superseded assets produced by the previous version of this report.
for filename in ('episode_17_prediction_errors.png', 'episode_17_full_prediction.mp4',
                 'episode_17_full_prediction.gif', 'episode_17_prediction_poster.jpg',
                 'gt_velocity_tracking_comparison.png', 'recorded_tcp_feedback_diagnostic.png'):
    (DEST / 'media' / filename).unlink(missing_ok=True)
manifest_path = DEST / 'evidence/prediction_video_manifest.json'
video_manifest = [entry for entry in json.loads(manifest_path.read_text()) if entry['episode'] == 31]
for entry in video_manifest:
    entry.pop('mp4', None)
    entry['gif'] = 'media/episode_31_full_prediction.gif'
manifest_path.write_text(json.dumps(video_manifest, indent=2))

style = '''
table { width: 100%; border-collapse: collapse; margin: 16px 0 20px; font-size: 15px; }
th, td { padding: 10px 12px; text-align: left; border: 1px solid #d9e1e7; vertical-align: top; }
th { background: #edf3f7; color: #16354b; font-weight: 600; }
tbody tr:nth-child(even) { background: #fafcfd; }
.metrics th:not(:first-child), .metrics td:not(:first-child) { text-align: right; white-space: nowrap; }
figure { margin: 20px 0 28px; }
figure img, video { display: block; width: 100%; height: auto; border: 1px solid #d9e1e7; border-radius: 4px; box-sizing: border-box; }
video { background: #111827; aspect-ratio: 16 / 11; }
figcaption { color: #52616c; font-size: 14px; margin-top: 8px; line-height: 1.5; }
a { color: #175f8a; text-underline-offset: 2px; }
.note { background: #f2f6f9; border-left: 3px solid #91adbf; padding: 10px 14px; font-size: 14px; color: #425361; }
.media-heading { margin: 25px 0 8px; font-size: 16px; color: #16354b; }
.media-links { margin: 8px 0 12px; font-size: 14px; }
details { margin: 12px 0 22px; }
summary { cursor: pointer; color: #175f8a; font-size: 14px; }
details figure { margin-top: 12px; }
.evidence { font-size: 13px; color: #52616c; }
@media (max-width: 600px) { body { margin-top: 24px; padding: 0 16px 28px; } th, td { padding: 8px; font-size: 13px; } }
@media print { table, figure { break-inside: avoid; } video, .media-links, details { display: none; } .print-poster { display: block !important; } a { color: inherit; } }
'''

oht_results = '''<h4 id="bkmrk-oht-results">2. Completed OHT Prediction Evaluation and Tracking Diagnostics</h4>
<p><strong>Completed full-episode prediction evaluation:</strong> Validated <code>model_epoch_30.pth</code> on Episodes 17 and 31 across all <strong>405 cached observations</strong>, including per-step errors, horizon-wise analysis, and a stationary-baseline comparison. The model outperformed the baseline on both episodes. Errors over the first four predicted steps were:</p>
<table class="metrics" aria-label="Model position prediction errors over the first four future steps">
<thead><tr><th scope="col">Episode</th><th scope="col">Mean Error</th><th scope="col">P95 Error</th><th scope="col">Maximum Error</th></tr></thead>
<tbody><tr><th scope="row">17</th><td>0.91 mm</td><td>2.57 mm</td><td>24.25 mm</td></tr>
<tr><th scope="row">31</th><td>1.58 mm</td><td>7.42 mm</td><td>44.66 mm</td></tr></tbody>
</table>
<ul>
<li><strong>Implemented and validated velocity feedforward:</strong> Added target-velocity feedforward and completed the tested GT replay trajectory with reduced tracking errors.</li>
<li><strong>Completed feedback diagnostics:</strong> Tested recursive TCP feedback with recorded images, identifying accumulated drift separately from independent prediction errors.</li>
<li><strong>Produced full-episode visualizations:</strong> Generated trajectory/error plots and a complete Episode 31 GIF covering all 191 cached observations, with model/GT overlays and local displacement views.</li>
</ul>
<p class="note">These recorded-input forecasts and feedback diagnostics do not establish closed-loop task success.</p>

<p class="media-heading"><strong>Prediction evaluation output — Episode 31</strong></p>
<figure>
<img src="media/episode_31_prediction_errors.png" alt="Episode 31 recorded-input model prediction trajectories and position errors over the full evaluated episode" loading="lazy" width="1950" height="1650">
<figcaption>Figure 2. Episode 31 prediction trajectories and error analysis. XYZ curves show step 4; the lower panel compares errors over steps 1, 1–4, and 1–16.</figcaption>
</figure>

<p class="media-heading"><strong>Full model-prediction visualization — Episode 31</strong></p>
<p>Orange: model forecast; green: recorded future GT. Thick lines highlight the first four steps. The animation shows independent forecasts over recorded images, without executing predicted motion.</p>
'''
for ep, count, duration in ((31, 191, '31.7'),):
    oht_results += f'''
<figure id="episode-{ep}-gif">
<img src="media/episode_{ep}_full_prediction.gif" alt="Full Episode {ep} model prediction animation with GT overlays and local displacement panels" loading="lazy" width="640" height="440">
<figcaption>GIF 1. Episode {ep}: approximately {duration} seconds, covering all {count} cached observations.</figcaption>
<p class="media-links"><a href="media/episode_{ep}_full_prediction.gif" target="_blank" rel="noopener">Open full GIF animation</a></p>
</figure>
'''
oht_results += '''
<p class="evidence">Evidence: <a href="evidence/prediction_analysis.json" download>prediction evaluation</a> · <a href="evidence/gt_tracking_comparison.json" download>GT tracking comparison</a> · <a href="evidence/recorded_tcp_feedback.json" download>TCP feedback diagnostic</a> · <a href="evidence/prediction_video_manifest.json" download>video coverage and provenance</a>.</p>
'''

html = SOURCE.read_text(encoding='utf-8')
if '.metrics th:not(:first-child)' not in html:
    html = html.replace('</style>', style + '\n</style>', 1)
if 'id="sam3-report-style"' not in html:
    html = html.replace('</head>', '''<style id="sam3-report-style">
.table-scroll { overflow-x: auto; margin: 16px 0 20px; }
.table-scroll table { min-width: 660px; margin: 0; }
@media (max-width: 600px) {
  .metrics th:not(:first-child), .metrics td:not(:first-child) { white-space: normal; }
  th, td { overflow-wrap: anywhere; }
}
</style>\n</head>''', 1)
html = html.replace('Distinguish insufficient predicted motion from controller tracking lag when executing continuous chunk targets.',
                    'Separate model prediction errors, accumulated feedback drift, and controller tracking lag during continuous chunk execution.')
html = html.replace('<li><strong>Extended OHT trajectory tracking and diagnosis:</strong> Added target-velocity feedforward, GT-chunk replay, recorded-state feedback diagnostics, and prediction-error analysis by future step and executed horizon.</li>',
                    '<li><strong>Evaluated OHT prediction and tracking:</strong> Completed full-episode prediction evaluation on 405 cached observations and produced complete prediction visualizations. Velocity feedforward improved GT trajectory tracking; local prediction peaks and feedback drift remain priorities.</li>')
start = html.index('<h4 id="bkmrk-oht-results">')
end = html.index('<h3 id="bkmrk-method">', start)
html = html[:start] + oht_results + '\n' + html[end:]
html = html.replace('<p>Use GT-chunk replay and recorded-state feedback to compare commanded targets with measured execution. Separately evaluate model predictions against future GT, both per step and over the executed prefix <em>H</em>. Compare displacement with GT motion and a stationary baseline to detect under-motion.</p>',
                    '<p>Use GT-chunk replay to isolate controller tracking and compare velocity feedforward under the same trajectory and execution settings. Evaluate model predictions independently against future GT, both per step and over the executed prefix <em>H</em>, with a stationary baseline for comparison. Separately test feedback sensitivity by feeding predicted TCP endpoints into subsequent queries while retaining recorded images and other states. Visualize saved predictions against future GT in camera views and local displacement panels.</p>')
html = html.replace('<li><strong>OHT:</strong> Compare tracking with and without velocity feedforward using GT replay and model rollouts. Analyze horizon-wise prediction errors, tracking lag, and displacement to locate the dominant source of under-motion.</li>',
                    '<li><strong>OHT:</strong> Extend paired tracking comparisons to more episodes and fast-motion phases. Investigate late-trajectory prediction peaks and feedback drift, then evaluate model rollouts with synchronized live observations and explicit task-success scoring.</li>')
html = re.sub(r'<li><strong>Evaluated OHT prediction and tracking:</strong>.*?</li>',
              '<li><strong>Completed OHT evaluation and diagnostics:</strong> Evaluated 405 cached observations, validated velocity feedforward in GT replay, completed TCP feedback diagnostics, and produced full-episode prediction visualizations.</li>', html)
html = re.sub(r'<li><strong>Completed OHT evaluation and diagnostics:</strong>.*?</li>',
              '<li><strong>Completed OHT evaluation and diagnostics:</strong> Evaluated 405 cached observations, validated velocity feedforward in GT replay, completed TCP feedback diagnostics, and produced full-episode prediction visualizations.</li>', html)
html = re.sub(r'(<h4 id="bkmrk-method-oht">.*?</h4>)\s*<p>.*?</p>',
              r'\1\n<p>Compare predictions with future GT and a stationary baseline. Use GT replay to isolate controller tracking, and recursive TCP feedback with recorded images to assess drift.</p>', html, flags=re.S)
html = re.sub(r'<li><strong>OHT:</strong>.*?</li>',
              '<li><strong>OHT:</strong> Investigate late-trajectory errors and feedback drift, then validate closed-loop success with synchronized live observations.</li>', html)

# Keep the supplied Stack blocks comparison distinct from OHT prediction errors.
# Prefer a downloaded copy when available; retain the supplied image URLs otherwise.
image_url = 'http://109.105.34.113/uploads/images/gallery/2026-11/qDuimage.png'
image_preview_url = 'http://109.105.34.113/uploads/images/gallery/2026-11/scaled-1680-/qDuimage.png'
local_image = DEST / 'media/sam3_bbox_conditioning.png'
image_src = 'media/sam3_bbox_conditioning.png' if local_image.is_file() else image_preview_url
image_access_note = '' if local_image.is_file() else (
    '<p class="evidence">The illustration is hosted externally and requires network access. '
    f'<a href="{image_url}" target="_blank" rel="noopener">Open the source image</a>.</p>'
)
sam3_results = f'''<!-- sam3-evaluation-start -->
<p class="media-heading"><strong>Stack blocks: task success comparison</strong></p>
<div class="table-scroll" role="region" aria-label="Scrollable task success comparison" tabindex="0">
<table id="bkmrk-oracle-success-rate" aria-label="Stack blocks task success rates in percent">
<thead><tr>
<th scope="col">Task / Success rate (%)</th>
<th scope="col">Reported official model (a)</th>
<th scope="col">Aligned Oracle object representation (d)</th>
<th scope="col">Predicted object (e)</th>
<th scope="col">Predicted object with SAM3 (f)</th>
</tr></thead>
<tbody><tr><th scope="row">Stack blocks</th><td>76.8</td><td><strong>84.0</strong></td><td>80.0</td><td><span style="text-decoration: underline;">80.0</span></td></tr></tbody>
</table>
</div>
<p>The aligned Oracle object representation achieved 84.0% success, compared with 80.0% for both predicted object variants. Adding SAM3 did not improve the observed success rate in this comparison; the gap to the aligned Oracle was 4.0 percentage points. Both predicted object variants were 3.2 percentage points above the reported official model (76.8%).</p>
<p class="media-heading" id="bkmrk-sam3-bbox-issue"><strong>Issue 1: SAM3 depends on accurate input bounding boxes</strong></p>
<p>The supplied example illustrates SAM3's sensitivity to bounding-box conditioning in the current pipeline. Inaccurate input boxes can alter the selected object or mask extent, affecting the extracted 3D object surface and downstream Relation/Anchor reasoning. This dependence on upstream localization introduces variability into the overall method and limits its robustness to bounding-box prediction errors.</p>
<figure id="sam3-bbox-conditioning">
<a href="{image_url}" target="_blank" rel="noopener"><img src="{image_src}" alt="User-supplied example illustrating SAM3 dependence on accurate input bounding-box conditioning" loading="lazy" onerror="this.style.display='none'"></a>
<figcaption>Figure 1. Example of SAM3 sensitivity to input bounding-box conditioning.</figcaption>
</figure>
{image_access_note}
<p class="note"><strong>Interpretation:</strong> The example provides qualitative evidence of prompt sensitivity. The success-rate comparison shows no observed gain from SAM3, but does not by itself establish bounding-box errors as the cause. Controlled box-perturbation experiments are needed to quantify the effect.</p>
<!-- sam3-evaluation-end -->
'''
html = re.sub(r'<!-- sam3-evaluation-start -->.*?<!-- sam3-evaluation-end -->\s*', '', html, flags=re.S)
html = html.replace('<h4 id="bkmrk-oht-results">', sam3_results + '\n<h4 id="bkmrk-oht-results">', 1)
html = html.replace('Figure 1. Episode 31:', 'Figure 2. Episode 31:')
sam3_summary = '<li id="bkmrk-sam3-summary"><strong>SAM3 evaluation:</strong> Stack blocks success was 80.0% with or without SAM3, versus 84.0% with the aligned Oracle object representation. Sensitivity to input bounding-box accuracy remains a robustness concern.</li>'
html = re.sub(r'<li id="bkmrk-sam3-summary">.*?</li>\s*', '', html)
summary_end = html.index('</ul>', html.index('<h2 id="bkmrk-summary">'))
html = html[:summary_end] + sam3_summary + '\n' + html[summary_end:]
html = re.sub(r'<li><strong>SAM3:</strong>.*?</li>',
              '<li><strong>SAM3:</strong> Quantify sensitivity to input-box translation and scale perturbations, measuring mask quality, fallback frequency, and closed-loop task success under matched settings. Use these results to investigate the remaining 4.0-point gap to the aligned Oracle object representation.</li>', html)

class AssetReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []
        self.headings = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for name in ('src', 'poster', 'href'):
            value = attrs.get(name)
            if value and not value.startswith(('#', 'data:', 'http:', 'https:')):
                self.references.append(value)
        if tag in ('h2', 'h3', 'h4'):
            self.headings.append((tag, attrs.get('id')))


check, original = AssetReferences(), AssetReferences()
check.feed(html)
original.feed(SOURCE.read_text(encoding='utf-8'))
assert check.headings == original.headings, 'Original section structure changed'
assert all((DEST / x).is_file() for x in check.references), 'Missing media/evidence reference'
assert '<video ' not in html and '.mp4' not in html
assert html.count('id="episode-31-gif"') == 1 and '405 cached observations' in html
assert not any('episode_17' in reference for reference in check.references)
regular = DEST / 'Weekly_Report_2026-11-23.html'
assert html.count('id="bkmrk-oracle-success-rate"') == 1
assert html.count('id="bkmrk-sam3-summary"') == 1
regular.write_text(html, encoding='utf-8')

# A standalone variant keeps images, GIF animation and evidence portable.
cache = {}
def inline(match):
    attribute, reference = match.group(1), match.group(2)
    if reference not in cache:
        path = DEST / reference
        mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        if path.suffix == '.json':
            mime = 'application/json'
        cache[reference] = f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
    return f'{attribute}="{cache[reference]}"'
standalone_html = re.sub(r'<p class="media-links">.*?</p>', '', html)
standalone = re.sub(r'(src|poster|href)="((?:media|evidence)/[^\"]+)"', inline, standalone_html)
standalone_path = ROOT / 'reports/Weekly_Report_2026-11-23.html'
standalone_path.write_text(standalone, encoding='utf-8')

archive = ROOT / 'reports/Weekly_Report_2026-11-23_with_media.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted({regular, *(DEST / reference for reference in check.references)}):
        z.write(p, p.relative_to(DEST.parent))
with zipfile.ZipFile(archive, 'r') as z:
    bad_member = z.testzip()
    assert bad_member is None, f'Corrupt archive member: {bad_member}'

for p in (regular, standalone_path, archive):
    print(p, f'{p.stat().st_size / 1e6:.2f} MB')
print('Verified original headings,', len(check.references), 'media/evidence references and ZIP integrity.')

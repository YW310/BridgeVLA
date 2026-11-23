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
    files.update({f'media/episode_{ep}_full_prediction.mp4': stem + '.mp4',
                  f'media/episode_{ep}_full_prediction.gif': stem + '_preview.gif',
                  f'media/episode_{ep}_prediction_poster.jpg': stem + '.jpg'})
for target, source in files.items():
    shutil.copy2(ROOT / source, DEST / target)
# Remove only superseded assets produced by the previous version of this report.
for filename in ('episode_17_prediction_errors.png', 'episode_17_full_prediction.mp4',
                 'episode_17_full_prediction.gif', 'episode_17_prediction_poster.jpg',
                 'gt_velocity_tracking_comparison.png', 'recorded_tcp_feedback_diagnostic.png'):
    (DEST / 'media' / filename).unlink(missing_ok=True)
manifest_path = DEST / 'evidence/prediction_video_manifest.json'
video_manifest = [entry for entry in json.loads(manifest_path.read_text()) if entry['episode'] == 31]
for entry in video_manifest:
    entry['mp4'] = 'media/episode_31_full_prediction.mp4'
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

oht_results = '''<h4 id="bkmrk-oht-results">2. Evaluated OHT Prediction Accuracy and Trajectory Tracking</h4>
<p><strong>Model prediction accuracy:</strong> Evaluated <code>model_epoch_30.pth</code> across two complete validation episodes, covering all <strong>405 cached observations</strong>. Position errors over the first four predicted steps were:</p>
<table class="metrics" aria-label="Model position prediction errors over the first four future steps">
<thead><tr><th scope="col">Episode</th><th scope="col">Mean Error</th><th scope="col">P95 Error</th><th scope="col">Maximum Error</th></tr></thead>
<tbody><tr><th scope="row">17</th><td>0.91 mm</td><td>2.57 mm</td><td>24.25 mm</td></tr>
<tr><th scope="row">31</th><td>1.58 mm</td><td>7.42 mm</td><td>44.66 mm</td></tr></tbody>
</table>
<p>Most predictions were accurate under recorded inputs, with larger deviations concentrated in late trajectory segments. The evaluated model outperformed a stationary baseline on both episodes, while local peaks remained significant.</p>
<table aria-label="OHT tracking and feedback findings">
<thead><tr><th scope="col">Experiment</th><th scope="col">Key Result</th><th scope="col">Implication</th></tr></thead>
<tbody><tr><th scope="row">GT replay with velocity feedforward</th><td>Completed the tested trajectory and reduced tracking errors; fast-motion deviations remained.</td><td>Demonstrated improved controller tracking on the tested trajectory.</td></tr>
<tr><th scope="row">Recursive TCP feedback</th><td>Prediction errors accumulated into substantial drift when predicted TCP states were fed back.</td><td>Revealed feedback sensitivity requiring validation with consistent live observations.</td></tr></tbody>
</table>
<p class="note"><strong>Evaluation scope:</strong> The prediction table uses independent recorded-input forecasts, without accumulated prediction feedback. Coverage includes every cached observation in the two selected episodes, primarily sampled every 10 raw frames. It does not establish closed-loop task success. The recursive TCP diagnostic keeps images and other robot states recorded, so it is also distinct from a simulator rollout.</p>

<p class="media-heading"><strong>Representative prediction analysis — Episode 31</strong></p>
<p>The table above summarizes both evaluated episodes. The figure and video below use Episode 31 as one representative example.</p>
<figure>
<img src="media/episode_31_prediction_errors.png" alt="Episode 31 recorded-input model prediction trajectories and position errors over the full evaluated episode" loading="lazy" width="1950" height="1650">
<figcaption>Figure 1. Episode 31: local prediction-error peaks remain despite a low overall mean.</figcaption>
</figure>
<p class="note"><strong>Reading the curves:</strong> The XYZ panels show the fourth predicted step in meters; the lower panel shows mean position errors over steps 1, 1–4, and 1–16 in millimeters. The table above reports statistics across individual points in the first four steps. Longer-horizon error peaks therefore need not match the fourth-step XYZ curves or the table maximum.</p>

<p class="media-heading"><strong>Full model-prediction visualization — Episode 31</strong></p>
<p>Orange shows the model forecast; green shows the recorded future GT. Thick lines highlight the first four predicted steps, and squares mark step 4. Local XY/XZ panels enlarge the displacement, while the blue cursor tracks episode time on the error chart.</p>
<p class="note">This animation visualizes saved <code>model_epoch_30.pth</code> predictions over recorded camera images. Each observation produces a separate forecast; predicted motion is not executed. It is neither a GT replay recording nor a learned-policy closed-loop execution video.</p>
'''
for ep, count, duration in ((31, 191, '31.7'),):
    oht_results += f'''
<figure id="episode-{ep}-video">
<video controls playsinline preload="metadata" poster="media/episode_{ep}_prediction_poster.jpg" aria-label="Episode {ep} complete model prediction visualization">
<source src="media/episode_{ep}_full_prediction.mp4" type="video/mp4">
Your browser does not support this video. Use the MP4 or GIF links below.
</video>
<img class="print-poster" style="display:none" src="media/episode_{ep}_prediction_poster.jpg" alt="Episode {ep} prediction visualization at its principal error peak">
<figcaption>Video 1. Episode {ep}: approximately {duration} seconds, covering all {count} cached observations.</figcaption>
<p class="media-links"><a href="media/episode_{ep}_full_prediction.mp4" download="Episode_{ep}_Full_Model_Prediction.mp4">Download MP4</a> · <a href="media/episode_{ep}_full_prediction.gif" target="_blank" rel="noopener">Open full GIF animation</a></p>
<details><summary>Show GIF preview if video playback is unavailable</summary>
<img src="media/episode_{ep}_full_prediction.gif" alt="Full Episode {ep} model prediction animation with GT overlays and local displacement panels" loading="lazy" width="640" height="440">
</details>
</figure>
'''
oht_results += '''
<p class="evidence">Evidence: <a href="evidence/prediction_analysis.json" download>prediction evaluation</a> · <a href="evidence/gt_tracking_comparison.json" download>GT tracking comparison</a> · <a href="evidence/recorded_tcp_feedback.json" download>TCP feedback diagnostic</a> · <a href="evidence/prediction_video_manifest.json" download>video coverage and provenance</a>.</p>
'''

html = SOURCE.read_text()
html = html.replace('</style>', style + '\n</style>', 1)
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
original.feed(SOURCE.read_text())
assert check.headings == original.headings, 'Original section structure changed'
assert all((DEST / x).is_file() for x in check.references), 'Missing media/evidence reference'
assert html.count('<video ') == 1 and '405 cached observations' in html
assert not any('episode_17' in reference for reference in check.references)
regular = DEST / 'Weekly_Report_2026-11-23.html'
regular.write_text(html)

# A standalone variant keeps images, video, GIF fallbacks and evidence portable.
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
standalone_html = re.sub(r'<p class="media-links">.*?</p>',
                        '<p class="media-links">Use the video controls to play or save the MP4. Expand the GIF preview below if playback is unavailable.</p>', html)
standalone = re.sub(r'(src|poster|href)="((?:media|evidence)/[^\"]+)"', inline, standalone_html)
standalone_path = ROOT / 'reports/Weekly_Report_2026-11-23.html'
standalone_path.write_text(standalone)

archive = ROOT / 'reports/Weekly_Report_2026-11-23_with_media.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(DEST.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(DEST.parent))
    assert z.testzip() is None

for p in (regular, standalone_path, archive):
    print(p, f'{p.stat().st_size / 1e6:.2f} MB')
print('Verified original headings,', len(check.references), 'media/evidence references and ZIP integrity.')

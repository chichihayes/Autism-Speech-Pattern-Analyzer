// Video Behavior Check: browser client.
// A video file is sampled locally with on-device MediaPipe models (gaze direction from face
// blendshapes, wrist position from pose landmarks). Nothing is uploaded — only the resulting
// numeric report is sent to /api/analyze-video for an AI-written interpretation.

import { FilesetResolver, FaceLandmarker, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';

const CFG = {
  maxSeconds: 60,
  maxFileBytes: 500 * 1024 * 1024,
  sampleStepSec: 0.15, // ~6.7 samples/sec: enough to catch a flapping/rocking rhythm, light enough to finish in reasonable time
  gazeAwayThreshold: 0.35, // blendshape score above which gaze counts as "away" rather than centered
  minGazeAwaySegment: 0.6, // seconds - ignore blinks/glances shorter than this
  stimWindowSec: 1.5, // sliding window used to detect oscillation
  stimReversalsPerSec: 2.2, // direction reversals/sec within a window to call it "repetitive movement"
  minStimEpisodeSec: 1.0,
};

const $ = (sel) => document.querySelector(sel);
const views = { upload: $('#view-upload'), progress: $('#view-progress'), results: $('#view-results') };

let selectedFile = null;
let previewUrl = null;
let playerUrl = null;
let faceLandmarker = null;
let poseLandmarker = null;

/* ---------- helpers ---------- */

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

const fmtBytes = (n) => (n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);

function show(name) {
  for (const [key, el] of Object.entries(views)) el.hidden = key !== name;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showError(msg) {
  const el = $('#error');
  el.textContent = msg;
  el.hidden = !msg;
}

function setStep(name) {
  const items = [...document.querySelectorAll('#steps li')];
  const idx = items.findIndex((li) => li.dataset.step === name);
  items.forEach((li, i) => {
    li.classList.toggle('done', i < idx);
    li.classList.toggle('active', i === idx);
  });
}

function setScanProgress(frac) {
  const bar = $('#scan-progress');
  bar.hidden = false;
  $('#scan-bar').style.width = `${Math.min(100, Math.max(0, frac * 100))}%`;
}

/* ---------- on-device models ---------- */

async function loadModels() {
  if (faceLandmarker && poseLandmarker) return;

  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm'
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU',
    },
    outputFaceBlendshapes: true,
    runningMode: 'VIDEO',
    numFaces: 1,
  });

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numPoses: 1,
  });
}

function seekTo(video, t) {
  return new Promise((resolve) => {
    const done = () => { video.removeEventListener('seeked', done); resolve(); };
    video.addEventListener('seeked', done);
    video.currentTime = t;
    // Some browsers don't fire `seeked` for a sub-frame seek to the same decoded frame.
    setTimeout(done, 300);
  });
}

function blendshapeScore(categories, name) {
  return categories?.find((c) => c.categoryName === name)?.score || 0;
}

async function scanVideo(video, onProgress) {
  const duration = Math.min(video.duration, CFG.maxSeconds);
  const samples = []; // { t, gazeAway: bool|null, leftY, rightY }

  let t = 0;
  while (t < duration) {
    await seekTo(video, t);
    const tsMs = Math.round(t * 1000);

    let gazeAway = null;
    try {
      const faceResult = faceLandmarker.detectForVideo(video, tsMs);
      const categories = faceResult?.faceBlendshapes?.[0]?.categories;
      if (categories) {
        const away = Math.max(
          blendshapeScore(categories, 'eyeLookInLeft'),
          blendshapeScore(categories, 'eyeLookOutLeft'),
          blendshapeScore(categories, 'eyeLookUpLeft'),
          blendshapeScore(categories, 'eyeLookDownLeft'),
          blendshapeScore(categories, 'eyeLookInRight'),
          blendshapeScore(categories, 'eyeLookOutRight'),
          blendshapeScore(categories, 'eyeLookUpRight'),
          blendshapeScore(categories, 'eyeLookDownRight')
        );
        gazeAway = away > CFG.gazeAwayThreshold;
      }
    } catch { /* face model hiccup on this frame - leave as "no face" */ }

    let leftY = null, rightY = null;
    try {
      const poseResult = poseLandmarker.detectForVideo(video, tsMs);
      const lm = poseResult?.landmarks?.[0];
      if (lm) { leftY = lm[15].y; rightY = lm[16].y; }
    } catch { /* pose model hiccup on this frame */ }

    samples.push({ t, gazeAway, leftY, rightY });
    t += CFG.sampleStepSec;
    onProgress(t / duration);
  }

  return { samples, duration };
}

/* ---------- metrics ---------- */

function mergeSegments(flags, minLen) {
  // flags: [{t, active}] sorted by t, roughly evenly spaced - merges consecutive active samples into [start, end] runs
  const segments = [];
  let start = null;
  let last = null;
  for (const f of flags) {
    if (f.active) {
      if (start === null) start = f.t;
      last = f.t;
    } else if (start !== null) {
      if (last - start >= minLen) segments.push([start, last]);
      start = null;
    }
  }
  if (start !== null && last - start >= minLen) segments.push([start, last]);
  return segments;
}

function computeGazeMetrics(samples, duration) {
  const withFace = samples.filter((s) => s.gazeAway !== null);
  const facePct = samples.length ? (withFace.length / samples.length) * 100 : 0;
  const awayPct = withFace.length ? (withFace.filter((s) => s.gazeAway).length / withFace.length) * 100 : null;
  const onCameraPct = awayPct === null ? null : 100 - awayPct;

  const awaySegments = mergeSegments(
    samples.map((s) => ({ t: s.t, active: s.gazeAway === true })),
    CFG.minGazeAwaySegment
  );

  // Longest run of consecutive "on camera" samples, in seconds.
  let longestOnCamera = 0, run = 0, prevT = null;
  for (const s of samples) {
    if (s.gazeAway === false) {
      run += prevT === null ? 0 : s.t - prevT;
      longestOnCamera = Math.max(longestOnCamera, run);
    } else if (s.gazeAway === true) {
      run = 0;
    }
    prevT = s.t;
  }

  return { facePct, onCameraPct, awaySegments, longestOnCamera, duration };
}

function countReversals(values) {
  let reversals = 0;
  let dir = 0;
  for (let i = 1; i < values.length; i++) {
    const d = values[i] - values[i - 1];
    if (Math.abs(d) < 0.003) continue; // ignore sub-pixel jitter
    const nd = d > 0 ? 1 : -1;
    if (dir !== 0 && nd !== dir) reversals++;
    dir = nd;
  }
  return reversals;
}

function computeStimMetrics(samples, duration) {
  const windowSamples = Math.max(3, Math.round(CFG.stimWindowSec / CFG.sampleStepSec));
  const flags = samples.map((s, i) => {
    const slice = samples.slice(Math.max(0, i - windowSamples), i + 1);
    const left = slice.map((x) => x.leftY).filter((v) => v !== null);
    const right = slice.map((x) => x.rightY).filter((v) => v !== null);
    const span = slice.length ? slice[slice.length - 1].t - slice[0].t : 0;
    if (span < CFG.stimWindowSec * 0.6) return { t: s.t, active: false };
    const reversals = countReversals(left) + countReversals(right);
    const rate = span > 0 ? reversals / span : 0;
    return { t: s.t, active: rate >= CFG.stimReversalsPerSec };
  });

  const episodes = mergeSegments(flags, CFG.minStimEpisodeSec);
  const totalStimSec = episodes.reduce((sum, [a, b]) => sum + (b - a), 0);
  const avgEpisodeSec = episodes.length ? totalStimSec / episodes.length : 0;

  return { episodes, episodeCount: episodes.length, totalStimSec, avgEpisodeSec, duration };
}

/* ---------- API ---------- */

async function apiError(res, fallback) {
  try {
    const data = await res.json();
    return new Error(data.error || fallback);
  } catch {
    return new Error(fallback);
  }
}

async function requestAnalysis(report) {
  const res = await fetch('/api/analyze-video', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ report }),
  });
  if (!res.ok) throw await apiError(res, 'AI analysis failed.');
  return (await res.json()).analysis;
}

/* ---------- rendering ---------- */

function renderMetrics(gaze, stim) {
  const cards = [
    {
      label: 'Gaze toward camera', value: gaze.onCameraPct, dec: 0, unit: '%',
      typical: 'typically the majority of the time', warn: gaze.onCameraPct !== null && gaze.onCameraPct < 50,
      skip: gaze.onCameraPct === null,
    },
    {
      label: 'Longest gaze streak', value: gaze.longestOnCamera, dec: 1, unit: 's',
    },
    { label: 'Face detected', value: gaze.facePct, dec: 0, unit: '% of frames' },
    {
      label: 'Repetitive-movement episodes', value: stim.episodeCount, dec: 0, unit: '',
      typical: 'typically rare or brief', warn: stim.episodeCount >= 3,
    },
    { label: 'Total repetitive-movement time', value: stim.totalStimSec, dec: 1, unit: 's' },
    { label: 'Video analyzed', value: gaze.duration, dec: 0, unit: 's' },
  ].filter((c) => !c.skip);

  $('#metrics').innerHTML = cards
    .map((c, i) => {
      const state = c.typical ? (c.warn ? 'warn' : 'ok') : '';
      return `<div class="metric ${state}" style="animation-delay:${i * 60}ms">
        <div class="label">${c.label}</div>
        <div class="value"><span data-to="${c.value}" data-dec="${c.dec}">0</span><small>${c.unit}</small></div>
        ${c.typical ? `<div class="typical"><span class="dot"></span>${c.typical}</div>` : ''}
      </div>`;
    })
    .join('');

  document.querySelectorAll('#metrics [data-to]').forEach((el) => {
    const to = Number(el.dataset.to);
    const dec = Number(el.dataset.dec);
    const t0 = performance.now();
    const tick = (now) => {
      const p = Math.min(1, (now - t0) / 900);
      el.textContent = (to * (1 - Math.pow(1 - p, 3))).toFixed(dec);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}

function analyzePatterns(gaze, stim) {
  const flags = [];
  if (gaze.onCameraPct === null) {
    flags.push({ level: 'note', text: 'No face was reliably detected - make sure the person is facing the camera and well lit.' });
  } else if (gaze.onCameraPct < 50) {
    flags.push({ level: 'warn', text: `Gaze was toward the camera only ${gaze.onCameraPct.toFixed(0)}% of the time, with ${gaze.awaySegments.length} distinct away-glances.` });
  } else {
    flags.push({ level: 'ok', text: `Gaze stayed toward the camera ${gaze.onCameraPct.toFixed(0)}% of the time.` });
  }

  if (stim.episodeCount >= 3) {
    flags.push({ level: 'warn', text: `${stim.episodeCount} repetitive-movement episodes detected, averaging ${stim.avgEpisodeSec.toFixed(1)}s each.` });
  } else if (stim.episodeCount > 0) {
    flags.push({ level: 'note', text: `${stim.episodeCount} brief repetitive-movement episode${stim.episodeCount > 1 ? 's' : ''} detected.` });
  } else {
    flags.push({ level: 'ok', text: 'No sustained repetitive hand movement detected.' });
  }

  return flags;
}

function renderFlags(flags) {
  const icons = { warn: '!', note: 'i', ok: '✓' };
  $('#flags').innerHTML = flags
    .map((f) => `<div class="flag ${f.level}"><span class="ic">${icons[f.level]}</span><span>${esc(f.text)}</span></div>`)
    .join('');
}

function renderTimeline(gaze, stim, duration) {
  const pct = (t) => `${(t / duration) * 100}%`;
  let html = '<div class="tl-track">';
  gaze.awaySegments.forEach(([a, b]) => {
    html += `<div class="tl-seg gaze" style="left:${pct(a)};width:${pct(Math.max(b - a, 0.05))}" title="Gaze away ${(b - a).toFixed(1)}s"></div>`;
  });
  stim.episodes.forEach(([a, b]) => {
    html += `<div class="tl-seg stim" style="left:${pct(a)};width:${pct(Math.max(b - a, 0.05))}" title="Repetitive movement ${(b - a).toFixed(1)}s"></div>`;
  });
  html += '<div class="tl-head" id="tl-head"></div></div><div class="tl-axis">';
  const step = duration > 40 ? 10 : duration > 20 ? 5 : 2;
  for (let t = 0; t <= duration; t += step) html += `<span style="left:${pct(t)}">${t}s</span>`;
  html += '</div>';
  $('#timeline').innerHTML = html;
}

function bindPlaybackHead(duration) {
  const player = $('#player');
  const head = $('#tl-head');
  let raf = null;
  const update = () => { head.style.left = `${Math.min(100, (player.currentTime / duration) * 100)}%`; };
  const loop = () => { update(); raf = requestAnimationFrame(loop); };
  player.onplay = () => { cancelAnimationFrame(raf); loop(); };
  player.onpause = player.onended = () => { cancelAnimationFrame(raf); update(); };
  player.onseeked = update;
  $('#timeline').onclick = (e) => {
    const track = e.target.closest('.tl-track');
    if (!track) return;
    const rect = track.getBoundingClientRect();
    player.currentTime = ((e.clientX - rect.left) / rect.width) * duration;
    player.play();
  };
}

// Minimal, HTML-safe Markdown: headings, bullet/numbered lists, bold and paragraphs.
function renderMarkdown(src) {
  const inline = (s) =>
    esc(s)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*\s][^*]*?)\*(?!\*)/g, '$1<em>$2</em>')
      .replace(/`([^`]+)`/g, '<code>$1</code>');
  const out = [];
  let list = null;
  const close = () => { if (list) out.push(`</${list}>`); list = null; };

  for (const raw of src.split('\n')) {
    const line = raw.trimEnd();
    let m;
    if (!line.trim()) {
      close();
    } else if ((m = line.match(/^\s*#{1,6}\s+(.*)/))) {
      close();
      out.push(`<h4>${inline(m[1])}</h4>`);
    } else if ((m = line.match(/^\s*[-*•]\s+(.*)/))) {
      if (list !== 'ul') { close(); out.push('<ul>'); list = 'ul'; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) {
      if (list !== 'ol') { close(); out.push('<ol>'); list = 'ol'; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if (/^\s*---+\s*$/.test(line)) {
      close();
    } else {
      close();
      out.push(`<p>${inline(line)}</p>`);
    }
  }
  close();
  return out.join('');
}

/* ---------- flow ---------- */

function pickFile(file) {
  showError('');
  if (!file) return;
  if (!file.type.startsWith('video/') && !/\.(mp4|mov|webm|m4v)$/i.test(file.name)) {
    return showError('Please choose a video file (MP4, MOV or WEBM).');
  }
  if (file.size > CFG.maxFileBytes) {
    return showError('That file is too large. Please choose one under 500 MB.');
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $('#preview').src = previewUrl;
  $('#picked-name').textContent = file.name;
  $('#picked-meta').textContent = `${fmtBytes(file.size)} · ${file.type || 'video'}`;
  $('#chooser').hidden = true;
  $('#picked').hidden = false;
}

function clearFile() {
  selectedFile = null;
  $('#file').value = '';
  $('#preview').removeAttribute('src');
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  $('#picked').hidden = true;
  $('#chooser').hidden = false;
  showError('');
}

function loadVideoElement(file) {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';
    video.src = URL.createObjectURL(file);
    video.onloadedmetadata = () => resolve(video);
    video.onerror = () => reject(new Error("This video couldn't be loaded. Try an MP4 or MOV file."));
  });
}

async function run() {
  if (!selectedFile) return;
  showError('');
  show('progress');
  $('#scan-progress').hidden = true;
  $('#scan-bar').style.width = '0%';

  try {
    setStep('prepare');
    const video = await loadVideoElement(selectedFile);
    const trimmed = video.duration > CFG.maxSeconds;
    const fullDuration = video.duration;

    setStep('models');
    await loadModels();

    setStep('scan');
    const { samples, duration } = await scanVideo(video, setScanProgress);

    setStep('metrics');
    const gaze = computeGazeMetrics(samples, duration);
    const stim = computeStimMetrics(samples, duration);
    const flags = analyzePatterns(gaze, stim);

    $('#result-file').textContent = selectedFile.name;
    const note = $('#trim-note');
    note.hidden = !trimmed;
    if (trimmed) note.textContent = `Your video is ${Math.round(fullDuration)}s long. Only the first ${CFG.maxSeconds}s were analyzed.`;

    const noFace = $('#no-face-note');
    noFace.hidden = gaze.facePct > 40;
    if (!noFace.hidden) noFace.textContent = `A face was only detected in ${gaze.facePct.toFixed(0)}% of sampled frames - gaze results may be unreliable.`;

    renderMetrics(gaze, stim);
    renderFlags(flags);
    renderTimeline(gaze, stim, duration);

    if (playerUrl) URL.revokeObjectURL(playerUrl);
    playerUrl = URL.createObjectURL(selectedFile);
    $('#player').src = playerUrl;
    bindPlaybackHead(duration);
    show('results');

    const ai = $('#ai');
    ai.className = 'ai loading';
    ai.innerHTML = '<span class="mini"></span>Generating analysis…';
    try {
      const text = await requestAnalysis({
        durationSec: Math.round(duration),
        gazeOnCameraPct: gaze.onCameraPct === null ? null : Math.round(gaze.onCameraPct),
        facePct: Math.round(gaze.facePct),
        longestGazeStreakSec: Number(gaze.longestOnCamera.toFixed(1)),
        gazeAwayEpisodeCount: gaze.awaySegments.length,
        stimEpisodeCount: stim.episodeCount,
        totalStimSec: Number(stim.totalStimSec.toFixed(1)),
        avgStimEpisodeSec: Number(stim.avgEpisodeSec.toFixed(1)),
      });
      ai.className = 'ai';
      ai.innerHTML = renderMarkdown(text);
    } catch (err) {
      ai.className = 'ai failed';
      ai.textContent = `AI analysis is unavailable right now. ${err.message}`;
    }
  } catch (err) {
    show('upload');
    showError(err.message || 'Something went wrong. Please try again.');
  }
}

/* ---------- events ---------- */

const dz = $('#dropzone');
$('#file').addEventListener('change', (e) => pickFile(e.target.files[0]));
dz.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#file').click(); }
});
['dragenter', 'dragover'].forEach((t) =>
  dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
['dragleave', 'drop'].forEach((t) =>
  dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
dz.addEventListener('drop', (e) => pickFile(e.dataTransfer.files[0]));

$('#clear').addEventListener('click', clearFile);
$('#analyze').addEventListener('click', run);
$('#again').addEventListener('click', () => {
  $('#player').pause();
  clearFile();
  show('upload');
});

// Video Behavior Check: browser client.
// A live camera session is sampled in real time with on-device MediaPipe models: gaze
// direction from face blendshapes, and hand shape/position from hand landmarks (which gives
// both arm-level movement like flapping/rocking, and hand-shape movement like squeezing).
// Nothing is uploaded - only the resulting numbers are sent to /api/analyze-video.

import { FilesetResolver, FaceLandmarker, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';

const CFG = {
  maxSeconds: 60,
  sampleStepSec: 0.15, // ~6.7 samples/sec: enough to catch a flapping/rocking/squeezing rhythm
  gazeAwayThreshold: 0.35, // blendshape score above which gaze counts as "away" rather than centered
  minGazeAwaySegment: 0.6, // seconds - ignore blinks/glances shorter than this
  movementWindowSec: 1.5, // sliding window for arm/hand position oscillation (flapping, rocking)
  movementReversalsPerSec: 2.2,
  minMovementEpisodeSec: 1.0,
  squeezeWindowSec: 1.2, // sliding window for hand-openness oscillation (squeezing)
  squeezeReversalsPerSec: 1.8,
  minSqueezeEpisodeSec: 0.8,
};

// Standard 21-point MediaPipe hand topology: wrist (0), thumb (1-4), index (5-8),
// middle (9-12), ring (13-16), pinky (17-20).
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];
const FINGERTIPS = [4, 8, 12, 16, 20];

const $ = (sel) => document.querySelector(sel);
const views = {
  start: $('#v-view-start'),
  loading: $('#v-view-loading'),
  live: $('#v-view-live'),
  progress: $('#v-view-progress'),
  results: $('#v-view-results'),
};

let faceLandmarker = null;
let handLandmarker = null;
let playerUrl = null;

// Live-session state
const live = {
  stream: null,
  recorder: null,
  chunks: [],
  samples: [],
  startedAt: 0,
  timer: null,
  sampleLoopId: null,
  hitCap: false,
  cancelled: false,
  gazeAwayCount: 0,
  movementCount: 0,
  squeezeCount: 0,
  prevGazeAway: null,
  prevMovementActive: false,
  prevSqueezeActive: false,
};

/* ---------- helpers ---------- */

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

const fmtClock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

function show(name) {
  for (const [key, el] of Object.entries(views)) el.hidden = key !== name;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showError(msg) {
  const el = $('#v-error');
  el.textContent = msg;
  el.hidden = !msg;
}

function setStep(name) {
  const items = [...document.querySelectorAll('#v-steps li')];
  const idx = items.findIndex((li) => li.dataset.step === name);
  items.forEach((li, i) => {
    li.classList.toggle('done', i < idx);
    li.classList.toggle('active', i === idx);
  });
}

/* ---------- on-device models ---------- */

let modelsPromise = null;

// Downloads the WASM runtime + two models. Kicked off eagerly as soon as this script runs
// (see the bottom of the file), so by the time someone clicks "Start live check" it's
// usually already loaded. Cached by the browser after the first visit.
function loadModels() {
  if (modelsPromise) return modelsPromise;

  modelsPromise = (async () => {
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

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 1,
    });
  })();

  modelsPromise.catch(() => { modelsPromise = null; }); // let a future call retry after a failure

  return modelsPromise;
}

// Start the download in the background right away, regardless of which tab is active.
loadModels();

function blendshapeScore(categories, name) {
  return categories?.find((c) => c.categoryName === name)?.score || 0;
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// Pulls one sample from the live preview video at this instant. Runs on an interval while
// the live session is active - `performance.now()` is strictly increasing, which is what
// detectForVideo requires.
function sampleLiveFrame(video, startedAt) {
  const t = (performance.now() - startedAt) / 1000;
  const tsMs = Math.round(performance.now());

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

  let wristX = null, wristY = null, openness = null, handLm = null;
  try {
    const handResult = handLandmarker.detectForVideo(video, tsMs);
    const lm = handResult?.landmarks?.[0];
    if (lm) {
      handLm = lm;
      wristX = lm[0].x;
      wristY = lm[0].y;
      const scale = dist(lm[0], lm[9]) || 0.0001; // wrist-to-middle-knuckle: a stable per-hand size reference
      openness = FINGERTIPS.reduce((sum, i) => sum + dist(lm[0], lm[i]), 0) / FINGERTIPS.length / scale;
    }
  } catch { /* hand model hiccup on this frame */ }

  return { t, gazeAway, wristX, wristY, openness, handLm };
}

// Draws the live detection feedback: the full hand skeleton (green, like a landmark-tracking
// overlay), the wrist point enlarging/reddening when arm-level movement is active, the whole
// hand flashing red when a squeezing (open/close) rhythm is active, and the gaze badge.
function drawLiveOverlay(video, canvas, sample, movementActive, squeezeActive) {
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
  }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (sample.handLm) {
    const lm = sample.handLm;
    const px = (p) => [p.x * canvas.width, p.y * canvas.height];
    const lineColor = squeezeActive ? '#f87171' : '#4ade80';

    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2;
    HAND_CONNECTIONS.forEach(([a, b]) => {
      const [x1, y1] = px(lm[a]);
      const [x2, y2] = px(lm[b]);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    });

    lm.forEach((p, i) => {
      const [x, y] = px(p);
      const isWrist = i === 0;
      const highlightWrist = isWrist && movementActive;
      ctx.beginPath();
      ctx.arc(x, y, highlightWrist ? 9 : 3.5, 0, Math.PI * 2);
      ctx.fillStyle = highlightWrist ? '#f87171' : lineColor;
      ctx.shadowColor = ctx.fillStyle;
      ctx.shadowBlur = highlightWrist ? 14 : 4;
      ctx.fill();
    });
  }

  const badge = $('#v-gaze-badge');
  if (sample.gazeAway === null) {
    badge.textContent = 'Gaze: no face';
    badge.className = 'gaze-badge';
  } else if (sample.gazeAway) {
    badge.textContent = 'Gaze: away';
    badge.className = 'gaze-badge away';
  } else {
    badge.textContent = 'Gaze: on camera';
    badge.className = 'gaze-badge on';
  }
}

function updateLiveCounters() {
  $('#v-live-counts').textContent =
    `Gaze deviations: ${live.gazeAwayCount}   Movement: ${live.movementCount}   Squeeze: ${live.squeezeCount}`;
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

// Generic oscillation check: are the given value(s) reversing direction fast enough, over
// this slice of samples, to call it repetitive motion? Used for both arm/hand position
// (flapping, rocking) and hand openness (squeezing) - just with different keys/thresholds.
function isActiveWindow(slice, valueKeys, windowSec, reversalsPerSec) {
  const span = slice.length ? slice[slice.length - 1].t - slice[0].t : 0;
  if (span < windowSec * 0.6) return false;
  let reversals = 0;
  for (const key of valueKeys) {
    reversals += countReversals(slice.map((s) => s[key]).filter((v) => v !== null));
  }
  const rate = span > 0 ? reversals / span : 0;
  return rate >= reversalsPerSec;
}

const windowSizeFor = (windowSec) => Math.max(3, Math.round(windowSec / CFG.sampleStepSec));

function computeEpisodes(samples, valueKeys, windowSec, reversalsPerSec, minEpisodeSec) {
  const windowSamples = windowSizeFor(windowSec);
  const flags = samples.map((s, i) => ({
    t: s.t,
    active: isActiveWindow(samples.slice(Math.max(0, i - windowSamples), i + 1), valueKeys, windowSec, reversalsPerSec),
  }));

  const episodes = mergeSegments(flags, minEpisodeSec);
  const totalSec = episodes.reduce((sum, [a, b]) => sum + (b - a), 0);
  const avgSec = episodes.length ? totalSec / episodes.length : 0;

  return { episodes, episodeCount: episodes.length, totalSec, avgSec };
}

// Cheap live check used only for the on-screen overlay/counters while recording.
function isActiveNow(samples, valueKeys, windowSec, reversalsPerSec) {
  const windowSamples = windowSizeFor(windowSec);
  return isActiveWindow(samples.slice(-windowSamples - 1), valueKeys, windowSec, reversalsPerSec);
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

function renderMetrics(gaze, movement, squeeze) {
  const cards = [
    {
      label: 'Gaze toward camera', value: gaze.onCameraPct, dec: 0, unit: '%',
      typical: 'typically the majority of the time', warn: gaze.onCameraPct !== null && gaze.onCameraPct < 50,
      skip: gaze.onCameraPct === null,
    },
    { label: 'Longest gaze streak', value: gaze.longestOnCamera, dec: 1, unit: 's' },
    { label: 'Gaze-away glances', value: gaze.awaySegments.length, dec: 0, unit: '' },
    {
      label: 'Arm/hand movement episodes', value: movement.episodeCount, dec: 0, unit: '',
      typical: 'typically rare or brief', warn: movement.episodeCount >= 3,
    },
    { label: 'Total movement time', value: movement.totalSec, dec: 1, unit: 's' },
    {
      label: 'Hand-squeeze episodes', value: squeeze.episodeCount, dec: 0, unit: '',
      typical: 'typically rare or brief', warn: squeeze.episodeCount >= 3,
    },
    { label: 'Total squeeze time', value: squeeze.totalSec, dec: 1, unit: 's' },
    { label: 'Video analyzed', value: gaze.duration, dec: 0, unit: 's' },
  ].filter((c) => !c.skip);

  $('#v-metrics').innerHTML = cards
    .map((c, i) => {
      const state = c.typical ? (c.warn ? 'warn' : 'ok') : '';
      return `<div class="metric ${state}" style="animation-delay:${i * 60}ms">
        <div class="label">${c.label}</div>
        <div class="value"><span data-to="${c.value}" data-dec="${c.dec}">0</span><small>${c.unit}</small></div>
        ${c.typical ? `<div class="typical"><span class="dot"></span>${c.typical}</div>` : ''}
      </div>`;
    })
    .join('');

  document.querySelectorAll('#v-metrics [data-to]').forEach((el) => {
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

function analyzePatterns(gaze, movement, squeeze) {
  const flags = [];
  if (gaze.onCameraPct === null) {
    flags.push({ level: 'note', text: 'No face was reliably detected - make sure the person is facing the camera and well lit.' });
  } else if (gaze.onCameraPct < 50) {
    flags.push({ level: 'warn', text: `Gaze was toward the camera only ${gaze.onCameraPct.toFixed(0)}% of the time, with ${gaze.awaySegments.length} distinct away-glances.` });
  } else {
    flags.push({ level: 'ok', text: `Gaze stayed toward the camera ${gaze.onCameraPct.toFixed(0)}% of the time.` });
  }

  if (movement.episodeCount >= 3) {
    flags.push({ level: 'warn', text: `${movement.episodeCount} arm/hand movement episodes detected, averaging ${movement.avgSec.toFixed(1)}s each.` });
  } else if (movement.episodeCount > 0) {
    flags.push({ level: 'note', text: `${movement.episodeCount} brief movement episode${movement.episodeCount > 1 ? 's' : ''} detected.` });
  } else {
    flags.push({ level: 'ok', text: 'No sustained repetitive arm/hand movement detected.' });
  }

  if (squeeze.episodeCount >= 3) {
    flags.push({ level: 'warn', text: `${squeeze.episodeCount} hand-squeezing episodes detected, averaging ${squeeze.avgSec.toFixed(1)}s each.` });
  } else if (squeeze.episodeCount > 0) {
    flags.push({ level: 'note', text: `${squeeze.episodeCount} brief hand-squeezing episode${squeeze.episodeCount > 1 ? 's' : ''} detected.` });
  } else {
    flags.push({ level: 'ok', text: 'No repeated hand-squeezing motion detected.' });
  }

  return flags;
}

function renderFlags(flags) {
  const icons = { warn: '!', note: 'i', ok: '✓' };
  $('#v-flags').innerHTML = flags
    .map((f) => `<div class="flag ${f.level}"><span class="ic">${icons[f.level]}</span><span>${esc(f.text)}</span></div>`)
    .join('');
}

function renderTimeline(gaze, movement, squeeze, duration) {
  const pct = (t) => `${(t / duration) * 100}%`;
  let html = '<div class="tl-track">';
  gaze.awaySegments.forEach(([a, b]) => {
    html += `<div class="tl-seg gaze" style="left:${pct(a)};width:${pct(Math.max(b - a, 0.05))}" title="Gaze away ${(b - a).toFixed(1)}s"></div>`;
  });
  movement.episodes.forEach(([a, b]) => {
    html += `<div class="tl-seg stim" style="left:${pct(a)};width:${pct(Math.max(b - a, 0.05))}" title="Movement ${(b - a).toFixed(1)}s"></div>`;
  });
  squeeze.episodes.forEach(([a, b]) => {
    html += `<div class="tl-seg squeeze" style="left:${pct(a)};width:${pct(Math.max(b - a, 0.05))}" title="Squeeze ${(b - a).toFixed(1)}s"></div>`;
  });
  html += '<div class="tl-head" id="v-tl-head"></div></div><div class="tl-axis">';
  const step = duration > 40 ? 10 : duration > 20 ? 5 : 2;
  for (let t = 0; t <= duration; t += step) html += `<span style="left:${pct(t)}">${t}s</span>`;
  html += '</div>';
  $('#v-timeline').innerHTML = html;
}

function bindPlaybackHead(duration) {
  const player = $('#v-player');
  const head = $('#v-tl-head');
  let raf = null;
  const update = () => { head.style.left = `${Math.min(100, (player.currentTime / duration) * 100)}%`; };
  const loop = () => { update(); raf = requestAnimationFrame(loop); };
  player.onplay = () => { cancelAnimationFrame(raf); loop(); };
  player.onpause = player.onended = () => { cancelAnimationFrame(raf); update(); };
  player.onseeked = update;
  $('#v-timeline').onclick = (e) => {
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

function stopTracks() {
  live.stream?.getTracks().forEach((t) => t.stop());
  live.stream = null;
}

function resetLive() {
  clearInterval(live.timer);
  clearTimeout(live.sampleLoopId);
  stopTracks();
  Object.assign(live, {
    recorder: null, chunks: [], samples: [], startedAt: 0, timer: null, sampleLoopId: null, hitCap: false, cancelled: false,
    gazeAwayCount: 0, movementCount: 0, squeezeCount: 0,
    prevGazeAway: null, prevMovementActive: false, prevSqueezeActive: false,
  });
  const overlay = $('#v-overlay');
  overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
  const badge = $('#v-gaze-badge');
  badge.textContent = 'Gaze: --';
  badge.className = 'gaze-badge';
  updateLiveCounters();
}

async function startLive() {
  showError('');
  if (!navigator.mediaDevices?.getUserMedia) {
    return showError('Camera access is not supported in this browser.');
  }

  show('loading');
  setStep('camera');

  try {
    live.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  } catch {
    show('start');
    return showError('Camera access was blocked. Allow the camera in your browser and try again.');
  }

  const preview = $('#v-live-preview');
  const overlay = $('#v-overlay');
  preview.srcObject = live.stream;

  try {
    setStep('models');
    await loadModels();
  } catch {
    stopTracks();
    show('start');
    return showError("Couldn't load the on-device detection models. Check your connection and try again.");
  }

  live.cancelled = false;
  live.hitCap = false;
  live.chunks = [];
  live.samples = [];
  live.gazeAwayCount = 0;
  live.movementCount = 0;
  live.squeezeCount = 0;
  live.prevGazeAway = null;
  live.prevMovementActive = false;
  live.prevSqueezeActive = false;
  live.recorder = new MediaRecorder(live.stream);
  live.recorder.ondataavailable = (e) => e.data.size && live.chunks.push(e.data);

  show('live');
  $('#v-rec-time').textContent = '0:00';
  updateLiveCounters();
  live.startedAt = performance.now();
  live.recorder.start(250);

  live.timer = setInterval(() => {
    const secs = (performance.now() - live.startedAt) / 1000;
    $('#v-rec-time').textContent = fmtClock(secs);
    if (secs >= CFG.maxSeconds) { live.hitCap = true; stopLive(); }
  }, 250);

  const loop = () => {
    const sample = sampleLiveFrame(preview, live.startedAt);
    live.samples.push(sample);

    const movementActive = isActiveNow(live.samples, ['wristX', 'wristY'], CFG.movementWindowSec, CFG.movementReversalsPerSec);
    const squeezeActive = isActiveNow(live.samples, ['openness'], CFG.squeezeWindowSec, CFG.squeezeReversalsPerSec);

    if (sample.gazeAway === true && live.prevGazeAway !== true) live.gazeAwayCount++;
    live.prevGazeAway = sample.gazeAway;
    if (movementActive && !live.prevMovementActive) live.movementCount++;
    live.prevMovementActive = movementActive;
    if (squeezeActive && !live.prevSqueezeActive) live.squeezeCount++;
    live.prevSqueezeActive = squeezeActive;
    updateLiveCounters();

    drawLiveOverlay(preview, overlay, sample, movementActive, squeezeActive);
    live.sampleLoopId = setTimeout(loop, CFG.sampleStepSec * 1000);
  };
  loop();
}

function cancelLive() {
  live.cancelled = true;
  if (live.recorder?.state === 'recording') live.recorder.stop();
  resetLive();
  show('start');
}

async function stopLive() {
  if (live.cancelled || !live.recorder) return;
  clearInterval(live.timer);
  clearTimeout(live.sampleLoopId);

  const samples = live.samples;
  const duration = Math.min((performance.now() - live.startedAt) / 1000, CFG.maxSeconds);
  const hitCap = live.hitCap;

  const recordedBlob = await new Promise((resolve) => {
    live.recorder.onstop = () => resolve(new Blob(live.chunks, { type: live.recorder.mimeType || 'video/webm' }));
    if (live.recorder.state === 'recording') live.recorder.stop();
    else resolve(new Blob(live.chunks, { type: live.recorder.mimeType || 'video/webm' }));
  });
  stopTracks();

  show('progress');

  try {
    const gaze = computeGazeMetrics(samples, duration);
    const movement = computeEpisodes(samples, ['wristX', 'wristY'], CFG.movementWindowSec, CFG.movementReversalsPerSec, CFG.minMovementEpisodeSec);
    const squeeze = computeEpisodes(samples, ['openness'], CFG.squeezeWindowSec, CFG.squeezeReversalsPerSec, CFG.minSqueezeEpisodeSec);
    const flags = analyzePatterns(gaze, movement, squeeze);

    const stamp = new Date().toLocaleString();
    $('#v-result-file').textContent = `Session recorded ${stamp}`;
    const note = $('#v-trim-note');
    note.hidden = !hitCap;
    if (hitCap) note.textContent = `Recording reached the ${CFG.maxSeconds}s cap and stopped automatically.`;

    const noFace = $('#v-no-face-note');
    noFace.hidden = gaze.facePct > 40;
    if (!noFace.hidden) noFace.textContent = `A face was only detected in ${gaze.facePct.toFixed(0)}% of sampled frames - gaze results may be unreliable.`;

    renderMetrics(gaze, movement, squeeze);
    renderFlags(flags);
    renderTimeline(gaze, movement, squeeze, duration);

    if (playerUrl) URL.revokeObjectURL(playerUrl);
    playerUrl = URL.createObjectURL(recordedBlob);
    $('#v-player').src = playerUrl;
    bindPlaybackHead(duration);
    show('results');

    const ai = $('#v-ai');
    ai.className = 'ai loading';
    ai.innerHTML = '<span class="mini"></span>Generating analysis…';
    try {
      const text = await requestAnalysis({
        durationSec: Math.round(duration),
        gazeOnCameraPct: gaze.onCameraPct === null ? null : Math.round(gaze.onCameraPct),
        facePct: Math.round(gaze.facePct),
        longestGazeStreakSec: Number(gaze.longestOnCamera.toFixed(1)),
        gazeAwayEpisodeCount: gaze.awaySegments.length,
        movementEpisodeCount: movement.episodeCount,
        totalMovementSec: Number(movement.totalSec.toFixed(1)),
        avgMovementEpisodeSec: Number(movement.avgSec.toFixed(1)),
        squeezeEpisodeCount: squeeze.episodeCount,
        totalSqueezeSec: Number(squeeze.totalSec.toFixed(1)),
        avgSqueezeEpisodeSec: Number(squeeze.avgSec.toFixed(1)),
      });
      ai.className = 'ai';
      ai.innerHTML = renderMarkdown(text);
    } catch (err) {
      ai.className = 'ai failed';
      ai.textContent = `AI analysis is unavailable right now. ${err.message}`;
    }
  } catch (err) {
    show('start');
    showError(err.message || 'Something went wrong. Please try again.');
  } finally {
    resetLive();
  }
}

/* ---------- events ---------- */

$('#v-start').addEventListener('click', startLive);
$('#v-rec-stop').addEventListener('click', stopLive);
$('#v-rec-cancel').addEventListener('click', cancelLive);
$('#v-again').addEventListener('click', () => {
  $('#v-player').pause();
  show('start');
});

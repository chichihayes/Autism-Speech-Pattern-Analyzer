// Speech Pattern Analyzer: browser client.
// Audio is decoded, trimmed and resampled in the browser, transcribed through /api/transcribe,
// measured locally, then explained through /api/analyze.

const CFG = {
  sampleRate: 16000,
  maxSeconds: 120, // 120 s of 16 kHz mono 16-bit WAV is about 3.8 MB, under Vercel's 4.5 MB body limit
  maxFileBytes: 200 * 1024 * 1024,
  initialResponseThreshold: 2.0,
  minSpeechRate: 80,
  maxAvgWordDuration: 1.5,
  longPause: 1.0,
};

const $ = (sel) => document.querySelector(sel);
const views = { upload: $('#view-upload'), progress: $('#view-progress'), results: $('#view-results') };

let selectedFile = null;
let previewUrl = null;
let playerUrl = null;
let rafId = null;

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

/* ---------- audio preparation ---------- */

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, text) => [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));

  write(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, 'WAVE');
  write(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

async function prepareAudio(file) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(await file.arrayBuffer());
  } catch {
    throw new Error("This file couldn't be decoded. Try a WAV or MP3 recording.");
  } finally {
    ctx.close();
  }

  const seconds = Math.min(decoded.duration, CFG.maxSeconds);
  const offline = new OfflineAudioContext(1, Math.ceil(seconds * CFG.sampleRate), CFG.sampleRate);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination); // multi-channel input is mixed down to mono
  source.start(0);
  const rendered = await offline.startRendering();

  return {
    wav: encodeWav(rendered.getChannelData(0), CFG.sampleRate),
    trimmed: decoded.duration > CFG.maxSeconds,
    fullDuration: decoded.duration,
  };
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

async function transcribe(wav) {
  const res = await fetch('/api/transcribe', {
    method: 'POST',
    headers: { 'Content-Type': 'audio/wav' },
    body: wav,
  });
  if (!res.ok) throw await apiError(res, 'Transcription failed.');
  return res.json();
}

async function requestAnalysis(metrics, flags, transcript) {
  const res = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ metrics, flags, transcript }),
  });
  if (!res.ok) throw await apiError(res, 'AI analysis failed.');
  return (await res.json()).analysis;
}

/* ---------- metrics ---------- */

function calculateMetrics(rawWords) {
  const words = rawWords
    .filter((w) => w.word && Number.isFinite(w.start))
    .map((w) => ({ word: w.word, start: w.start, end: w.end > w.start ? w.end : w.start + 0.5 }));
  if (!words.length) return null;

  const first = words[0];
  const last = words[words.length - 1];
  const totalDuration = last.end - first.start;
  const durations = words.map((w) => w.end - w.start);

  const norm = words.map((w) => w.word.toLowerCase().replace(/[^\p{L}\p{N}']/gu, ''));
  const bigrams = {};
  for (let i = 0; i < norm.length - 1; i++) {
    if (!norm[i] || !norm[i + 1]) continue;
    const key = `${norm[i]} ${norm[i + 1]}`;
    bigrams[key] = (bigrams[key] || 0) + 1;
  }
  const repeatedPhrases = Object.entries(bigrams)
    .filter(([, n]) => n > 1)
    .sort((a, b) => b[1] - a[1])
    .map(([phrase]) => phrase);

  return {
    words,
    metrics: {
      initialResponseTime: first.start,
      wordsPerMinute: totalDuration > 0 ? (words.length / totalDuration) * 60 : 0,
      totalDuration,
      averageWordDuration: durations.reduce((a, b) => a + b, 0) / durations.length,
      maxWordDuration: Math.max(...durations),
      wordCount: words.length,
      repeatedPhrases,
    },
  };
}

function analyzeDelays(m) {
  const flags = [];
  if (m.initialResponseTime > CFG.initialResponseThreshold) {
    flags.push({ level: 'warn', text: `Delayed initial response: ${m.initialResponseTime.toFixed(2)}s (typical: <${CFG.initialResponseThreshold}s)` });
  }
  if (m.wordsPerMinute < CFG.minSpeechRate) {
    flags.push({ level: 'warn', text: `Reduced speech rate: ${m.wordsPerMinute.toFixed(1)} wpm (typical: 100-130 wpm)` });
  }
  if (m.averageWordDuration > CFG.maxAvgWordDuration) {
    flags.push({ level: 'warn', text: `Extended word durations: ${m.averageWordDuration.toFixed(2)}s avg (typical: <${CFG.maxAvgWordDuration}s)` });
  }
  if (m.repeatedPhrases.length) {
    flags.push({ level: 'note', text: `Repeated phrases detected (possible echolalia): ${m.repeatedPhrases.slice(0, 3).join(', ')}` });
  }
  if (!flags.length) flags.push({ level: 'ok', text: 'No significant delay patterns detected' });
  return flags;
}

/* ---------- rendering ---------- */

function renderMetrics(m) {
  const cards = [
    { label: 'Initial response', value: m.initialResponseTime, dec: 2, unit: 's', typical: `typical < ${CFG.initialResponseThreshold}s`, warn: m.initialResponseTime > CFG.initialResponseThreshold },
    { label: 'Speech rate', value: m.wordsPerMinute, dec: 0, unit: 'wpm', typical: 'typical 100-130 wpm', warn: m.wordsPerMinute < CFG.minSpeechRate },
    { label: 'Avg word duration', value: m.averageWordDuration, dec: 2, unit: 's', typical: `typical < ${CFG.maxAvgWordDuration}s`, warn: m.averageWordDuration > CFG.maxAvgWordDuration },
    { label: 'Max word duration', value: m.maxWordDuration, dec: 2, unit: 's' },
    { label: 'Total duration', value: m.totalDuration, dec: 1, unit: 's' },
    { label: 'Word count', value: m.wordCount, dec: 0, unit: '' },
  ];

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

function renderFlags(flags) {
  const icons = { warn: '!', note: 'i', ok: '✓' };
  $('#flags').innerHTML = flags
    .map((f) => `<div class="flag ${f.level}"><span class="ic">${icons[f.level]}</span><span>${esc(f.text)}</span></div>`)
    .join('');
}

function renderTimeline(words) {
  const end = words[words.length - 1].end;
  const pct = (t) => `${(t / end) * 100}%`;
  let html = '<div class="tl-track">';

  words.forEach((w, i) => {
    if (i > 0) {
      const gap = w.start - words[i - 1].end;
      if (gap > CFG.longPause) {
        html += `<div class="tl-gap" style="left:${pct(words[i - 1].end)};width:${pct(gap)}" title="Pause ${gap.toFixed(2)}s"></div>`;
      }
    }
  });
  if (words[0].start > CFG.longPause) {
    html += `<div class="tl-gap" style="left:0;width:${pct(words[0].start)}" title="Silence before speech ${words[0].start.toFixed(2)}s"></div>`;
  }

  words.forEach((w, i) => {
    const dur = w.end - w.start;
    html += `<div class="tl-word ${dur > CFG.maxAvgWordDuration ? 'long' : ''}" data-i="${i}" style="left:${pct(w.start)};width:${(dur / end) * 100}%" title="${esc(w.word)} (${dur.toFixed(2)}s)"></div>`;
  });
  html += '<div class="tl-head" id="tl-head"></div></div><div class="tl-axis">';

  const step = end > 60 ? 15 : end > 25 ? 5 : end > 10 ? 2 : 1;
  for (let t = 0; t <= end; t += step) html += `<span style="left:${pct(t)}">${t}s</span>`;
  html += '</div>';
  $('#timeline').innerHTML = html;
}

function renderTranscript(words) {
  $('#transcript').innerHTML = words
    .map((w, i) => {
      const long = w.end - w.start > CFG.maxAvgWordDuration ? ' long' : '';
      return `<span class="w${long}" data-i="${i}">${esc(w.word)}</span>`;
    })
    .join(' ');
}

function renderWordTable(words) {
  $('#word-table').innerHTML =
    '<table><thead><tr><th>Start</th><th>End</th><th>Duration</th><th>Word</th></tr></thead><tbody>' +
    words
      .map((w) => `<tr><td>${w.start.toFixed(2)}s</td><td>${w.end.toFixed(2)}s</td><td>${(w.end - w.start).toFixed(2)}s</td><td>${esc(w.word)}</td></tr>`)
      .join('') +
    '</tbody></table>';
}

// Minimal, HTML-safe Markdown: headings, bullet/numbered lists, bold and paragraphs.
function renderMarkdown(src) {
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`([^`]+)`/g, '<code>$1</code>');
  const out = [];
  let list = null;
  const close = () => {
    if (list) out.push(`</${list}>`);
    list = null;
  };

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

/* ---------- playback sync ---------- */

function bindPlayback(words) {
  const player = $('#player');
  const head = $('#tl-head');
  const end = words[words.length - 1].end;
  let activeIdx = -1;

  const update = () => {
    const t = player.currentTime;
    head.style.left = `${Math.min(100, (t / end) * 100)}%`;

    let idx = -1;
    for (let i = 0; i < words.length; i++) {
      if (words[i].start <= t) idx = i;
      else break;
    }
    if (idx !== -1 && t > words[idx].end + 0.3) idx = -1; // between words or during a pause
    if (idx === activeIdx) return;

    document.querySelectorAll('.w.active, .tl-word.active').forEach((el) => el.classList.remove('active'));
    activeIdx = idx;
    if (idx !== -1) {
      document.querySelector(`.w[data-i="${idx}"]`)?.classList.add('active');
      document.querySelector(`.tl-word[data-i="${idx}"]`)?.classList.add('active');
    }
  };

  const loop = () => {
    update();
    rafId = requestAnimationFrame(loop);
  };
  player.onplay = () => { cancelAnimationFrame(rafId); loop(); };
  player.onpause = player.onended = () => { cancelAnimationFrame(rafId); update(); };
  player.onseeked = update;

  $('#transcript').onclick = (e) => {
    const el = e.target.closest('.w');
    if (!el) return;
    player.currentTime = words[Number(el.dataset.i)].start;
    player.play();
  };
  $('#timeline').onclick = (e) => {
    const track = e.target.closest('.tl-track');
    if (!track) return;
    const rect = track.getBoundingClientRect();
    player.currentTime = ((e.clientX - rect.left) / rect.width) * end;
    player.play();
  };
}

/* ---------- flow ---------- */

function pickFile(file) {
  showError('');
  if (!file) return;
  if (!file.type.startsWith('audio/') && !/\.(wav|mp3|m4a|flac|ogg)$/i.test(file.name)) {
    return showError('Please choose an audio file (WAV, MP3, M4A, FLAC or OGG).');
  }
  if (file.size > CFG.maxFileBytes) {
    return showError('That file is too large. Please choose one under 200 MB.');
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $('#preview').src = previewUrl;
  $('#picked-name').textContent = file.name;
  $('#picked-meta').textContent = `${fmtBytes(file.size)} · ${file.type || 'audio'}`;
  $('#dropzone').hidden = true;
  $('#picked').hidden = false;
}

function clearFile() {
  selectedFile = null;
  $('#file').value = '';
  $('#preview').removeAttribute('src');
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  $('#picked').hidden = true;
  $('#dropzone').hidden = false;
  showError('');
}

async function run() {
  if (!selectedFile) return;
  showError('');
  show('progress');

  try {
    setStep('prepare');
    const prepared = await prepareAudio(selectedFile);

    setStep('transcribe');
    const transcription = await transcribe(prepared.wav);

    setStep('metrics');
    const analysis = calculateMetrics(transcription.words || []);
    if (!analysis) {
      throw new Error('No speech could be measured. Make sure the audio contains clear speech (5-10 seconds minimum).');
    }
    const { words, metrics } = analysis;
    const flags = analyzeDelays(metrics);

    // Results
    $('#result-file').textContent = selectedFile.name;
    const note = $('#trim-note');
    note.hidden = !prepared.trimmed;
    if (prepared.trimmed) {
      note.textContent = `Your recording is ${Math.round(prepared.fullDuration)}s long. Only the first ${CFG.maxSeconds}s were analyzed.`;
    }

    renderMetrics(metrics);
    renderFlags(flags);
    renderTimeline(words);
    renderTranscript(words);
    renderWordTable(words);

    if (playerUrl) URL.revokeObjectURL(playerUrl);
    playerUrl = URL.createObjectURL(selectedFile);
    $('#player').src = playerUrl;
    bindPlayback(words);
    show('results');

    // AI interpretation loads after the results are already on screen
    const ai = $('#ai');
    ai.className = 'ai loading';
    ai.innerHTML = '<span class="mini"></span>Generating analysis…';
    const flagText = flags.map((f) => `${{ warn: 'WARNING', note: 'NOTE', ok: 'RESULT' }[f.level]}: ${f.text}`);
    try {
      const text = await requestAnalysis(metrics, flagText, transcription.text);
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

// Receives a 16 kHz mono WAV body and returns word-level timestamps from Groq Whisper.
// Vercel limits request bodies to 4.5 MB, so the browser trims and resamples before sending.

export const config = { api: { bodyParser: false } };

const WHISPER_MODEL = 'whisper-large-v3';
const MAX_BYTES = 4.5 * 1024 * 1024;

async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BYTES) throw new Error('too-large');
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed.' });
  }

  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'GROQ_API_KEY is not configured on the server.' });
  }

  let audio;
  try {
    audio = await readBody(req);
  } catch (err) {
    if (err.message === 'too-large') {
      return res.status(413).json({ error: 'Audio is too large. Try a shorter recording.' });
    }
    return res.status(400).json({ error: 'Could not read the uploaded audio.' });
  }
  if (audio.length < 1000) {
    return res.status(400).json({ error: 'No audio received.' });
  }

  const form = new FormData();
  form.append('file', new Blob([audio], { type: 'audio/wav' }), 'audio.wav');
  form.append('model', WHISPER_MODEL);
  form.append('response_format', 'verbose_json');
  form.append('timestamp_granularities[]', 'word');
  form.append('language', 'en');

  try {
    const upstream = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}` },
      body: form,
    });

    if (!upstream.ok) {
      const detail = await upstream.text();
      console.error('Groq error', upstream.status, detail);
      return res.status(502).json({ error: `Transcription failed (${upstream.status}).` });
    }

    const data = await upstream.json();
    const words = (data.words || []).map((w) => ({
      word: String(w.word ?? '').trim(),
      start: Number(w.start ?? 0),
      end: Number(w.end ?? 0),
    }));
    return res.status(200).json({ text: data.text || '', words });
  } catch (err) {
    console.error(err);
    return res.status(502).json({ error: 'Could not reach the transcription service.' });
  }
}

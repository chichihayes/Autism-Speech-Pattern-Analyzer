// Builds the interpretation prompt from measured video metrics (gaze + repetitive movement)
// and asks OpenRouter for an explanation. Mirrors api/analyze.js.

// Paid model first (fast and reliable); free models are frequently rate-limited, so they only serve as backups.
const FALLBACK_MODELS = [
  'anthropic/claude-haiku-4.5',
  'nvidia/nemotron-3-super-120b-a12b:free',
  'google/gemma-4-31b-it:free',
  'z-ai/glm-5.2:free',
  'qwen/qwen3.8-27b:free',
  'google/gemma-4-26b-a4b-it:free',
];

const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed.' });
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'OPENROUTER_API_KEY is not configured on the server.' });
  }

  const r = (req.body || {}).report || {};
  const gazePct = r.gazeOnCameraPct === null || r.gazeOnCameraPct === undefined ? null : num(r.gazeOnCameraPct);

  const prompt = `You are a behavioral-video pattern analyst. Analyze the following measurements from a short video and provide insights:

MEASUREMENTS:
- Video analyzed: ${num(r.durationSec)} seconds
- A face was detected in: ${num(r.facePct)}% of sampled frames
- Gaze toward the camera: ${gazePct === null ? 'not reliably measurable (face not consistently detected)' : `${gazePct}% of the time`}
- Longest unbroken stretch of gaze toward the camera: ${num(r.longestGazeStreakSec).toFixed(1)} seconds
- Distinct gaze-away glances: ${num(r.gazeAwayEpisodeCount)}
- Repetitive hand/arm movement episodes (e.g. flapping, rocking motion): ${num(r.stimEpisodeCount)}
- Total time spent in repetitive movement: ${num(r.totalStimSec).toFixed(1)} seconds
- Average length of a repetitive-movement episode: ${num(r.avgStimEpisodeSec).toFixed(1)} seconds

These numbers come from lightweight on-device computer vision (face gaze-direction blendshapes and body-pose wrist tracking), not a clinical instrument.

Please provide:
1. What these numbers indicate about attention and movement during the clip
2. How gaze and repetitive-movement patterns like these are generally discussed in the research literature
3. Notable observations - anything that stands out as unusually high, low, or typical
4. Clear caveats about what a single short clip can and can't tell you

Be analytical and educational, not diagnostic. Never suggest a diagnosis or clinical conclusion - this is a single short sample of behavior, not an assessment.
Format the answer in Markdown with short headings and bullet points.`;

  const models = [process.env.OPENROUTER_MODEL, ...FALLBACK_MODELS].filter(Boolean);

  const deadline = Date.now() + 85000;

  const attempts = [];
  for (let round = 0; round < 3; round++) attempts.push(...models);

  for (const [i, model] of attempts.entries()) {
    if (deadline - Date.now() < 5000) break;
    if (i > 0 && i % models.length === 0) await new Promise((r2) => setTimeout(r2, 3000));
    try {
      const upstream = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
          'X-Title': 'Video Behavior Check',
        },
        body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }] }),
        signal: AbortSignal.timeout(Math.min(45000, deadline - Date.now())),
      });

      if (!upstream.ok) {
        console.error('OpenRouter error', model, upstream.status, await upstream.text());
        continue;
      }

      const data = await upstream.json();
      let text = data?.choices?.[0]?.message?.content || '';
      text = text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
      if (text) return res.status(200).json({ analysis: text });
    } catch (err) {
      console.error('OpenRouter request failed', model, err.message);
    }
  }

  return res.status(502).json({ error: 'The AI models are busy right now. Please try again in a minute.' });
}

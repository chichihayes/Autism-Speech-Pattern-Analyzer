// Builds the interpretation prompt from measured video metrics (gaze, arm/hand movement,
// and hand-squeezing) and asks OpenRouter for an explanation. Mirrors api/analyze.js.

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

  const prompt = `Report on exactly three things measured from a short video clip: gaze, arm/hand movement (flapping or rocking-type motion), and hand-squeezing (repeated opening/closing of the hand). Nothing else - no literature review, no extra topics.

GAZE
- Toward the camera: ${gazePct === null ? 'not reliably measurable (face not consistently detected)' : `${gazePct}% of the time`}
- Longest unbroken gaze streak: ${num(r.longestGazeStreakSec).toFixed(1)}s
- Distinct gaze-away glances: ${num(r.gazeAwayEpisodeCount)}

ARM/HAND MOVEMENT
- Episodes detected: ${num(r.movementEpisodeCount)}
- Total time: ${num(r.totalMovementSec).toFixed(1)}s
- Average episode length: ${num(r.avgMovementEpisodeSec).toFixed(1)}s

HAND-SQUEEZING
- Episodes detected: ${num(r.squeezeEpisodeCount)}
- Total time: ${num(r.totalSqueezeSec).toFixed(1)}s
- Average episode length: ${num(r.avgSqueezeEpisodeSec).toFixed(1)}s

Clip length: ${num(r.durationSec)}s

For each of the three things above, write exactly 3 to 4 plain sentences, no headings, no bullet points, no markdown formatting: state what the numbers show, then say plainly and directly what a pattern at that frequency is generally associated with - don't hedge everything into vagueness, be concrete. End the whole report with exactly one closing sentence noting this is a single short clip, not a diagnosis. No introduction, no "in summary," nothing else.`;

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

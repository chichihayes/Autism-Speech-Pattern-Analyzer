// Builds the interpretation prompt from measured metrics and asks OpenRouter for an explanation.

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
const str = (v, max) => String(v ?? '').slice(0, max);

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed.' });
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'OPENROUTER_API_KEY is not configured on the server.' });
  }

  const body = req.body || {};
  const m = body.metrics || {};
  const flags = Array.isArray(body.flags) ? body.flags.slice(0, 10).map((f) => str(f, 300)) : [];
  const phrases = Array.isArray(m.repeatedPhrases)
    ? m.repeatedPhrases.slice(0, 10).map((p) => str(p, 80))
    : [];
  const transcript = str(body.transcript, 8000);

  const prompt = `You are a speech pattern analyst. Analyze the following speech metrics and provide insights:

TRANSCRIPT: ${transcript}

METRICS:
- Initial Response Time: ${num(m.initialResponseTime).toFixed(2)} seconds
- Speech Rate: ${num(m.wordsPerMinute).toFixed(0)} words per minute
- Average Word Duration: ${num(m.averageWordDuration).toFixed(2)} seconds
- Maximum Word Duration: ${num(m.maxWordDuration).toFixed(2)} seconds
- Total Duration: ${num(m.totalDuration).toFixed(1)} seconds
- Word Count: ${num(m.wordCount)}

PATTERN ANALYSIS:
${flags.join('\n')}

REPEATED PHRASES: ${phrases.length ? phrases.join(', ') : 'None'}

Please provide:
1. What these numbers indicate about the speech pattern
2. How the metrics relate to typical speech development
3. Notable observations about timing, pacing, and fluency
4. Any patterns that stand out

Be analytical and educational, not diagnostic. Focus on explaining what the data shows.
Format the answer in Markdown with short headings and bullet points.`;

  const models = [process.env.OPENROUTER_MODEL, ...FALLBACK_MODELS].filter(Boolean);

  const deadline = Date.now() + 85000;

  const attempts = [];
  for (let round = 0; round < 3; round++) attempts.push(...models);

  for (const [i, model] of attempts.entries()) {
    if (deadline - Date.now() < 5000) break;
    if (i > 0 && i % models.length === 0) await new Promise((r) => setTimeout(r, 3000));
    try {
      const upstream = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
          'X-Title': 'Speech Pattern Analyzer',
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

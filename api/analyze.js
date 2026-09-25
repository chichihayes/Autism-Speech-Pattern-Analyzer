// Builds the interpretation prompt from measured metrics and asks OpenRouter for an explanation.

const DEFAULT_MODEL = 'deepseek/deepseek-r1-0528:free';

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

  try {
    const upstream = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'X-Title': 'Speech Pattern Analyzer',
      },
      body: JSON.stringify({
        model: process.env.OPENROUTER_MODEL || DEFAULT_MODEL,
        messages: [{ role: 'user', content: prompt }],
      }),
    });

    if (!upstream.ok) {
      console.error('OpenRouter error', upstream.status, await upstream.text());
      return res.status(502).json({ error: `AI analysis failed (${upstream.status}).` });
    }

    const data = await upstream.json();
    let text = data?.choices?.[0]?.message?.content || '';
    text = text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
    if (!text) return res.status(502).json({ error: 'The AI returned an empty response.' });
    return res.status(200).json({ analysis: text });
  } catch (err) {
    console.error(err);
    return res.status(502).json({ error: 'Could not reach the AI service.' });
  }
}

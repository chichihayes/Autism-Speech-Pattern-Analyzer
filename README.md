# Speech Pattern Analyzer

A dark-themed web app that measures speech timing and fluency from an audio recording and explains the results with AI. Plain HTML, CSS and JavaScript on the front end, with two small serverless functions on Vercel. No framework and no TypeScript.

> For educational and research use only. It is not a diagnostic tool.

## What it does

- Transcribes audio with Groq Whisper Large V3, with word-level timestamps
- Measures initial response time, speech rate (wpm), average and maximum word duration, and word count
- Flags delayed responses, reduced speech rate, extended words and repeated phrases (possible echolalia)
- Shows a speech timeline with pauses highlighted, and a transcript that follows the audio as it plays
- Writes an educational interpretation of the numbers through OpenRouter

## How it works

| Piece | Role |
| --- | --- |
| `public/` | Static site: `index.html`, `style.css`, `app.js` |
| `api/transcribe.js` | Sends the audio to Groq Whisper and returns word timings |
| `api/analyze.js` | Builds the prompt from the measured metrics and calls OpenRouter |

The browser decodes the file, trims it to the first 120 seconds, and resamples it to 16 kHz mono WAV before upload. That keeps the request under Vercel's 4.5 MB body limit and replaces the librosa step from the earlier Python version. API keys stay on the server.

## Deploy to Vercel

1. Import this repository at [vercel.com/new](https://vercel.com/new). No build settings are needed.
2. Add these environment variables under Project Settings, then redeploy:

   | Name | Where to get it |
   | --- | --- |
   | `GROQ_API_KEY` | https://console.groq.com/keys |
   | `OPENROUTER_API_KEY` | https://openrouter.ai/keys |
   | `OPENROUTER_MODEL` (optional) | Defaults to `deepseek/deepseek-r1-0528:free` |

## Run locally

```bash
npm i -g vercel
cp .env.example .env   # then fill in your keys
vercel dev
```

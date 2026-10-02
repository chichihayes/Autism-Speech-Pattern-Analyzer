# Speech & Video Pattern Analyzer

A dark-themed web app with two tabs - Speech and Video - that measure behavioral patterns and explain the results with AI. Plain HTML, CSS and JavaScript on the front end, with small serverless functions on Vercel. No framework and no TypeScript.

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
| `public/index.html` | Single page with two tabs (Speech, Video); tab switching is a tiny inline script |
| `public/app.js` | Speech tab's logic |
| `public/video.js` | Video tab's logic (its DOM ids are all prefixed `v-` so the two tabs' scripts never collide) |
| `api/transcribe.js` | Sends the audio to Groq Whisper and returns word timings |
| `api/analyze.js` | Builds the prompt from the measured speech metrics and calls OpenRouter |
| `api/analyze-video.js` | Builds the prompt from the measured video metrics and calls OpenRouter |

The browser decodes the file, trims it to the first 120 seconds, and resamples it to 16 kHz mono WAV before upload. That keeps the request under Vercel's 4.5 MB body limit and replaces the librosa step from the earlier Python version. API keys stay on the server.

## Video tab

Runs a live webcam session and measures three things in real time, entirely on-device:

- Starts the camera with `getUserMedia`, records up to 60 seconds with `MediaRecorder` (for playback afterward), and samples the live feed at roughly 6-7 frames/second while it runs
- Runs MediaPipe FaceLandmarker (gaze-direction blendshapes) and HandLandmarker (21-point hand skeleton) in the browser via WebAssembly - nothing is ever uploaded, detection and recording both happen on-device
- **Live feedback while recording**: the hand skeleton is drawn directly on the camera preview (green, flashing red when a repetitive pattern is active), the wrist point enlarges when arm-level movement is active, a gaze badge shows on-camera/away/no-face live, and a running counter shows gaze-deviation/movement/squeeze counts updating as they happen
- **Gaze**: % of time looking toward the camera vs away, longest unbroken streak, number of distinct away-glances
- **Arm/hand movement** (flapping, rocking): sustained oscillation in wrist position (direction reversals/sec over a sliding window) - episode count and total/average duration
- **Hand-squeezing**: sustained oscillation in hand openness (the average distance from wrist to fingertips, normalized by hand size, so it tracks the hand opening and closing rather than moving) - episode count and total/average duration
- Sends only the resulting numbers (never the video) to `api/analyze-video.js`, which reports on exactly these three things in a short, direct, non-diagnostic explanation

## Deploy to Vercel

1. Import this repository at [vercel.com/new](https://vercel.com/new). No build settings are needed.
2. Add these environment variables under Project Settings, then redeploy:

   | Name | Where to get it |
   | --- | --- |
   | `GROQ_API_KEY` | https://console.groq.com/keys |
   | `OPENROUTER_API_KEY` | https://openrouter.ai/keys |
   | `OPENROUTER_MODEL` (optional) | Tried first if set; otherwise a built-in list of free models is tried in order |

## Run locally

```bash
npm i -g vercel
cp .env.example .env   # then fill in your keys
vercel dev
```

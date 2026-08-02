<div align="center">

<img src="https://raw.githubusercontent.com/sruckh/mossTTS-v1.5-runpod-serverless/main/assets/readme/hero-banner.svg" width="100%" alt="MOSS-TTS RunPod Serverless Hero Banner">

# MOSS-TTS-v1.5 RunPod Serverless API Engine

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8.2-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![RunPod](https://img.shields.io/badge/RunPod-Serverless-7C3AED?style=flat-square&logo=runpod&logoColor=white)](https://www.runpod.io/)
[![FlashAttention](https://img.shields.io/badge/FlashAttention-v2.8.3-06B6D4?style=flat-square)](https://github.com/Dao-AILab/flash-attention)

*Production-grade GPU Serverless deployment for MOSS-TTS-v1.5 text-to-speech synthesis, supporting **Dual Streaming & Non-Streaming Modes**, zero-shot voice cloning, auto-language detection, and sub-5-second cold starts.*

</div>

---

## ⚡ Key Architecture Highlights

- **Dual Execution Modes**: Native support for both **Streaming** (`stream: true` generator yields) and **Non-Streaming** (`stream: false` full Base64 audio payload).
- **Sub-5-Second Cold Starts**: Automated First-Time Bootstrapper downloads weights once to persistent RunPod Network Volume NVMe storage (`/runpod-volume`). Subsequent worker spawns skip downloads and initialize in ~3–5s.
- **Zero-Shot Voice Cloning**: Accepts reference audio via direct **Base64 upload** (`reference_audio_base64`) OR remote **Audio URL** (`reference_url`).
- **Word-Level Timings**: Non-streaming responses optionally carry `word_timings` — transcript-locked forced alignment (MMS_FA, ~50 Hz emission) against the model-normalized text, with **zero extra inference models required by the caller** and graceful omission on any failure. See Response Schema §3A.
- **Language Detection & Selection**: Explicit language tagging (`Chinese`, `English`, `French`) with automatic prompt script detection when omitted.
- **FlashAttention-2 Acceleration**: Pre-compiled binary wheel matching PyTorch 2.8 + CUDA 12 ABI (`flash_attn-2.8.3`) doubling token generation throughput on RTX 4090/A100/A10G GPUs.
- **Robust VRAM & Resource Management**: Unbuffered real-time logging (`PYTHONUNBUFFERED=1`), diagnostic startup header, zero VRAM leakage (`torch.cuda.empty_cache()`), and lightweight health check ping endpoint.

---

## 🤖 AI Agent Guide: Constructing a WebUI Frontend

> **Notice for AI Agents (Cursor, Claude Code, Copilot, ChatGPT)**: Use the technical specifications in this section to generate a complete WebUI frontend (React / Next.js / HTML5 + Tailwind CSS) that connects to this RunPod Serverless API.

### 1. API Endpoint URLs & Routing

RunPod Serverless exposes the following REST endpoints for your Endpoint ID (`YOUR_ENDPOINT_ID`):

| Endpoint Path | HTTP Method | Mode | Purpose |
| :--- | :--- | :--- | :--- |
| `https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/runsync` | `POST` | Non-Streaming | Synchronous execution; blocks until synthesis finishes and returns complete JSON. |
| `https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run` | `POST` | Asynchronous / Stream Init | Initiates an async or streaming job; returns `{ "id": "JOB_ID", "status": "IN_QUEUE" }`. |
| `https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/stream/{JOB_ID}` | `GET` | Stream Polling | Retrieves streamed audio chunks yielded by the worker generator. |
| `https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/health` | `GET` | Monitoring | Returns worker count, queued jobs, and endpoint status. |

---

### 2. Request Schema Specification

#### `POST` Body JSON Structure (`input` object):

```typescript
interface RunPodTTSInput {
  // Target text to synthesize into speech (REQUIRED)
  text: string;

  // Stream toggle: true for generator chunk streaming, false for single full response (DEFAULT: false)
  stream?: boolean;

  // Language tag: "Chinese" | "English" | "French" (DEFAULT: auto-detected from text)
  language?: "Chinese" | "English" | "French";

  // Base64 encoded reference audio string for voice cloning
  reference_audio_base64?: string;

  // Reference audio container format if using Base64 upload (DEFAULT: "wav")
  reference_format?: "wav" | "mp3" | "m4a" | "flac" | "ogg";

  // Remote URL to reference audio file for voice cloning
  reference_url?: string;

  // Optional target duration in tokens
  tokens?: number;

  // Maximum tokens to generate (DEFAULT: 4096)
  max_new_tokens?: number;

  // Health ping flag: true returns instant healthy status without inference
  ping?: boolean;
}
```

---

### 3. Response Schema Specification

#### A. Non-Streaming Response (`stream: false` or `/runsync`)
```json
{
  "delayTime": 320,
  "executionTime": 1850,
  "id": "sync-job-12345",
  "output": {
    "status": "success",
    "audio_base64": "UklGRi...",
    "format": "wav",
    "sample_rate": 24000,
    "detected_language": "English",
    "word_timings": {
      "frame_rate": 50.0,
      "source": "mms_fa_forced_alignment",
      "words": [
        { "w": "Hello,",  "start": 0.02, "end": 0.41 },
        { "w": "welcome", "start": 0.45, "end": 0.83 },
        { "w": "to",      "start": 0.83, "end": 0.94 }
      ]
    }
  },
  "status": "COMPLETED"
}
```

**`word_timings` (optional, additive):** word-level timing produced by forced alignment of the synthesized waveform against the model-normalized transcript (`torchaudio.pipelines.MMS_FA` — transcript-locked, no ASR guesses). Rules for consumers:

- **The key may be absent.** Absent means "fall back" (e.g. to interpolating word positions across the audio duration); present means "trust me". The key is omitted entirely — never empty, never partial — when alignment fails or the text has no alignable words. Alignment failure never fails the job.
- **`words[].w` is the rendered string.** Render the spoken line from this array, not from your own copy of the input text: the model normalizes text before synthesis (numbers expanded, punctuation reflowed), so `w` reflects what was actually spoken.
- **`start`/`end`** are seconds from the start of the returned WAV, floats, monotonically non-decreasing, always `start < end`.
- **`frame_rate`** is the alignment emission rate (≈50 Hz at 16 kHz); informational only.
- **`source`** is currently `"mms_fa_forced_alignment"`. Chinese text is romanized via pinyin for alignment and returned one character per word.

#### B. Streaming Output Chunks (`stream: true` via `/stream/{JOB_ID}`)
```json
{
  "stream": [
    {
      "output": {
        "status": "streaming",
        "chunk_index": 1,
        "audio_chunk_base64": "UklGRi...",
        "format": "wav",
        "sample_rate": 24000,
        "is_final": false
      }
    },
    {
      "output": {
        "status": "streaming",
        "chunk_index": 2,
        "audio_chunk_base64": "UklGRi...",
        "format": "wav",
        "sample_rate": 24000,
        "is_final": true
      }
    }
  ],
  "status": "COMPLETED"
}
```

---

### 4. Client WebAudio & Streaming Implementation (Copy-Paste Code)

#### JavaScript Audio Streaming Player (`WebAudio API`):

```javascript
class MOSSStreamPlayer {
  constructor(apiKey, endpointId) {
    this.apiKey = apiKey;
    this.endpointId = endpointId;
    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    this.nextStartTime = 0;
  }

  // Convert Base64 string to ArrayBuffer
  base64ToArrayBuffer(base64) {
    const binaryString = window.atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }

  // Play audio chunk sequentially without gaps
  async playChunk(base64Wav) {
    const arrayBuffer = this.base64ToArrayBuffer(base64Wav);
    const audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer);
    
    const source = this.audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioCtx.destination);

    const currentTime = this.audioCtx.currentTime;
    if (this.nextStartTime < currentTime) {
      this.nextStartTime = currentTime;
    }

    source.start(this.nextStartTime);
    this.nextStartTime += audioBuffer.duration;
  }

  // Start Streaming Synthesis Job & Poll Chunks
  async synthesizeStream(payload) {
    // 1. Submit Async Job
    const response = await fetch(`https://api.runpod.ai/v2/${this.endpointId}/run`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({ input: { ...payload, stream: true } })
    });
    
    const { id: jobId } = await response.json();
    console.log(`[MOSS-TTS] Streaming Job Initialized: ${jobId}`);

    // 2. Poll /stream/{jobId} until complete
    let isCompleted = false;
    let processedChunks = 0;

    while (!isCompleted) {
      const streamRes = await fetch(`https://api.runpod.ai/v2/${this.endpointId}/stream/${jobId}`, {
        headers: { "Authorization": `Bearer ${this.apiKey}` }
      });
      const streamData = await streamRes.json();

      if (streamData.stream) {
        for (let i = processedChunks; i < streamData.stream.length; i++) {
          const chunkData = streamData.stream[i].output;
          if (chunkData.audio_chunk_base64) {
            await this.playChunk(chunkData.audio_chunk_base64);
          }
          processedChunks++;
        }
      }

      if (streamData.status === "COMPLETED" || streamData.status === "FAILED") {
        isCompleted = true;
      } else {
        await new Promise((r) => setTimeout(r, 250)); // Poll interval
      }
    }
  }
}
```

---

## 🛠️ RunPod Serverless Environment Variables

Configure these environment variables in the **RunPod Console → Serverless Template**:

| Variable Name | Recommended Value | Purpose |
| :--- | :--- | :--- |
| `HF_TOKEN` | `hf_xxxxxxxxxxxxxxxxxxxx` | Hugging Face Access Token for rate limits & model downloads. |
| `RUNPOD_VOLUME_PATH` | `/runpod-volume` | Mount point for the persistent NVMe RunPod Network Volume. |
| `MODEL_REPO` | `OpenMOSS-Team/MOSS-TTS-v1.5` | Target Hugging Face repository ID. |
| `HF_HOME` | `/runpod-volume/huggingface` | Directs Hugging Face cache to Network Volume (prevents root disk full). |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Enables Rust-accelerated `hf_transfer` high-speed downloads. |
| `HF_XET_HIGH_PERFORMANCE` | `1` | Enables multi-part S3 range gets in Hugging Face Hub client. |
| `PYTHONUNBUFFERED` | `1` | Forces unbuffered stdout/stderr real-time console logging. |
| `RUNPOD_SKIP_GPU_CHECK` | `true` | **Required.** Skips the RunPod SDK's native GPU memory allocation fitness test, which runs *after* the model loads into VRAM and false-positives CUDA OOM on otherwise-healthy workers. |
| `RUNPOD_SKIP_AUTO_SYSTEM_CHECKS` | `true` | **Required.** Skips the SDK's auto-registered system fitness checks (memory, disk, network, CUDA init, GPU benchmark) — the CUDA init check also OOMs post-model-load and kills workers before any job runs. |

> **Why the two `RUNPOD_SKIP_*` flags are mandatory:** the SDK's startup fitness checks execute *after* `handler.py` loads the model onto the GPU. Their own probe allocations then fail with `CUDA error: out of memory`, marking the worker UNHEALTHY and cancelling queued jobs — even though the model itself loaded fine. See [runpod-python worker fitness checks](https://github.com/runpod/runpod-python/blob/main/docs/serverless/worker_fitness_checks.md).

---

## 🖥️ GPU Requirements

| Tier | GPU Pools | Status |
| :--- | :--- | :--- |
| **Minimum (recommended)** | `ADA_48_PRO`, `AMPERE_48` (L40S / A40, 48 GB) | ✅ Verified working |
| **Headroom** | `AMPERE_80` (A100 80 GB) | ✅ Works |
| **Fallback** | `ADA_32_PRO` (32 GB) | ⚠️ Works, less headroom |
| **Too small** | `ADA_24`, `AMPERE_24` (L4 / RTX 4090, 24 GB) | ❌ Model loads but inference OOMs |

The model *loads* on 24 GB cards but generation needs several GB of activation/workspace headroom — jobs fail with `CUDA error: out of memory`. Configure pools under **Endpoint → GPU Configuration**.

---

## 🚢 Deployment Notes (GitHub Integration)

- **Rebuilds trigger on releases, not pushes.** RunPod's GitHub integration only builds a new image when you create a **GitHub release** — pushing commits to `main` alone changes nothing on the endpoint. Cut a release (e.g. `gh release create v1.5.1`) to deploy code changes.
- **Handler detection**: the repo scanner requires a top-level `runpod.serverless.start({...})` call in `handler.py` (not wrapped in an `if __name__ == "__main__":` block).
- **Dockerfile is authoritative for the image**: `requirements.txt` is a reference for local dev — the Dockerfile installs deps explicitly, so add new packages in **both** places.
- **Build limits**: `docker build` step capped at 30 min; total build window 160 min; image max 80 GB.

---

## 💻 cURL Testing Snippets

### 1. Non-Streaming Request (`/runsync`)

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "text": "Hello, welcome to MOSS-TTS serverless voice generation.",
      "language": "English",
      "stream": false
    }
  }'
```

### 2. Voice Cloning via Reference Audio URL

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "text": "Bonjour! Ceci est un test de clonage vocal.",
      "reference_url": "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_zh.wav",
      "stream": false
    }
  }'
```

### 3. Health Check Ping Request

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "ping": true
    }
  }'
```

---

## 🚀 Repository Structure

```text
.
├── Dockerfile                  # Multi-stage build (Ubuntu 24.04 + CUDA 12.8.2 + PyTorch 2.8)
├── handler.py                  # Serverless worker handler (Streaming/Non-Streaming/First-time boot)
├── requirements.txt            # Python runtime dependencies
├── .env.example                # Template for RunPod serverless environment variables
└── assets/
    └── readme/
        └── hero-banner.svg     # SVG Header Card graphic
```

---

## 📄 License
This project is licensed under the Apache 2.0 License. MOSS-TTS-v1.5 weights are subject to OpenMOSS Community License terms.

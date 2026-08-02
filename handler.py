import base64
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path

import torch
import torchaudio
import transformers
from huggingface_hub import snapshot_download
import runpod

# Ensure stdout is unbuffered for immediate console output
sys.stdout.reconfigure(line_buffering=True)

FLASH_ATTN_WHEEL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
VOLUME_PATH = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
MODEL_DIR = VOLUME_PATH / "models" / "MOSS-TTS-v1.5"
MODEL_REPO = os.getenv("MODEL_REPO", "OpenMOSS-Team/MOSS-TTS-v1.5")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# MMS_FA forced alignment (word-level timings). torchaudio is already pinned;
# the only extra runtime dep is pypinyin (Chinese romanization), bootstrap-installed
# at provisioning time like the FlashAttention wheel (keeps the Dockerfile untouched).
FA_SAMPLE_RATE = 16000
TORCH_HUB_DIR = VOLUME_PATH / "torch-hub"
# Route torch.hub downloads (MMS_FA weights) to the persistent network volume so
# cold starts never re-download the ~1.2GB aligner.
TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(TORCH_HUB_DIR))

# =====================================================================
# STARTUP DEBUG DIAGNOSTICS BANNER (Logged to Console)
# =====================================================================
print("=" * 75, flush=True)
print("=== RUNPOD SERVERLESS WORKER DIAGNOSTIC LOG ===", flush=True)
print(f"[DEBUG] Python Version:        {sys.version.split()[0]}", flush=True)
print(f"[DEBUG] PyTorch Version:       {torch.__version__}", flush=True)
print(f"[DEBUG] PyTorch CUDA Version:  {torch.version.cuda}", flush=True)
print(f"[DEBUG] CUDA Available:        {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    print(f"[DEBUG] GPU Device Name:       {props.name}", flush=True)
    print(f"[DEBUG] GPU Total VRAM:        {vram_gb:.2f} GB", flush=True)
    print(f"[DEBUG] GPU Compute Cap:       {props.major}.{props.minor}", flush=True)
else:
    print("[WARNING] CUDA NOT AVAILABLE! Running in CPU Mode.", flush=True)

flash_attn_spec = importlib.util.find_spec("flash_attn")
print(f"[DEBUG] FlashAttention Specs:  {'Found' if flash_attn_spec else 'Not Installed'}", flush=True)
print(f"[DEBUG] Transformers Version:  {transformers.__version__}", flush=True)
print(f"[DEBUG] Torchaudio Version:     {torchaudio.__version__}", flush=True)
print(f"[DEBUG] RunPod SDK Version:    {getattr(runpod, '__version__', 'Installed')}", flush=True)
print(f"[DEBUG] Network Volume Path:   {VOLUME_PATH} (Exists: {VOLUME_PATH.exists()})", flush=True)
print(f"[DEBUG] HF Token Present:      {bool(HF_TOKEN)}", flush=True)
print("=" * 75, flush=True)

# Disable cuDNN SDPA backend due to MOSS-TTS compatibility
if DEVICE == "cuda":
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

def ensure_flash_attention():
    """Ensure FlashAttention-2 is installed from binary wheel if missing."""
    if DEVICE == "cuda" and importlib.util.find_spec("flash_attn") is None:
        print(f"[FIRST-TIME BOOTSTRAP] Installing FlashAttention-2 wheel from {FLASH_ATTN_WHEEL}...", flush=True)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", FLASH_ATTN_WHEEL],
                check=True
            )
            print("[FIRST-TIME BOOTSTRAP] FlashAttention-2 installed successfully!", flush=True)
        except Exception as e:
            print(f"[WARNING] FlashAttention-2 installation failed: {e}. Falling back to SDPA.", flush=True)

def ensure_alignment_provisioned():
    """First-Time Start Routine: cache MMS_FA aligner weights and pypinyin on the volume/venv."""
    if importlib.util.find_spec("pypinyin") is None:
        print("[FIRST-TIME BOOTSTRAP] Installing pypinyin (Chinese romanization for alignment)...", flush=True)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", "pypinyin"],
                check=True
            )
            print("[FIRST-TIME BOOTSTRAP] pypinyin installed successfully!", flush=True)
        except Exception as e:
            print(f"[WARNING] pypinyin installation failed: {e}. Chinese word_timings will be omitted.", flush=True)
    try:
        from torchaudio.pipelines import MMS_FA
        if not (TORCH_HUB_DIR / "checkpoints").exists() or not any((TORCH_HUB_DIR / "checkpoints").glob("mms_fa*")):
            print("[FIRST-TIME START] Downloading MMS_FA alignment weights (~1.2GB) to network volume...", flush=True)
            MMS_FA.get_model()  # downloads into TORCH_HUB_DIR via torch.hub
            print("[FIRST-TIME START] MMS_FA weights cached on network volume!", flush=True)
        else:
            print("[SUBSEQUENT START] Using cached MMS_FA weights from network volume", flush=True)
    except Exception as e:
        print(f"[WARNING] MMS_FA provisioning failed: {e}. word_timings will be omitted.", flush=True)

def ensure_model_provisioned():
    """First-Time Start Routine: Auto-downloads models using accelerated hf tool."""
    ensure_flash_attention()
    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        print(f"[FIRST-TIME START] Model directory {MODEL_DIR} not found.", flush=True)
        print(f"[FIRST-TIME START] Initiating high-speed download for {MODEL_REPO} via 'hf download'...", flush=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                "hf", "download", MODEL_REPO,
                "--local-dir", str(MODEL_DIR),
                "--max-workers", "16"
            ]
            if HF_TOKEN:
                cmd.extend(["--token", HF_TOKEN])
            print(f"[EXEC] Running CLI command: {' '.join(cmd[:4])} ...", flush=True)
            subprocess.run(cmd, check=True)
            print(f"[FIRST-TIME START] 'hf download' completed successfully!", flush=True)
        except Exception as cli_err:
            print(f"[WARNING] 'hf download' CLI failed ({cli_err}). Falling back to snapshot_download API...", flush=True)
            snapshot_download(
                repo_id=MODEL_REPO,
                local_dir=str(MODEL_DIR),
                local_dir_use_symlinks=False,
                max_workers=16,
                token=HF_TOKEN
            )
        print(f"[FIRST-TIME START] Successfully stored model weights in {MODEL_DIR} for fast subsequent starts!", flush=True)
    else:
        print(f"[SUBSEQUENT START] Using existing pre-cached model weights from {MODEL_DIR}", flush=True)

def resolve_attn_implementation(device: str, dtype) -> str:
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    return "sdpa" if device == "cuda" else "eager"

# Run First-Time Bootstrapper & Load Model Globally (RunPod Best Practice #1)
ensure_model_provisioned()
ensure_alignment_provisioned()

print(f"[INIT] Loading MOSS-TTS-v1.5 processor and model on {DEVICE}...", flush=True)
ATTN_IMPL = resolve_attn_implementation(DEVICE, DTYPE)
print(f"[INIT] Attention Implementation Selected: {ATTN_IMPL}", flush=True)

from transformers import AutoModel, AutoProcessor

processor = AutoProcessor.from_pretrained(
    str(MODEL_DIR),
    trust_remote_code=True,
)
processor.audio_tokenizer = processor.audio_tokenizer.to(DEVICE)

model = AutoModel.from_pretrained(
    str(MODEL_DIR),
    trust_remote_code=True,
    attn_implementation=ATTN_IMPL,
    torch_dtype=DTYPE,
).to(DEVICE)
model.eval()
print("[INIT] Model loaded successfully and worker is warm!", flush=True)

# Load the MMS_FA forced aligner globally. Failure here must NEVER kill the worker —
# word_timings is simply omitted when the aligner is unavailable.
FA_MODEL = None
FA_TOKENIZER = None
FA_ALIGNER = None
try:
    from torchaudio.pipelines import MMS_FA
    print(f"[INIT] Loading MMS_FA forced aligner on {DEVICE}...", flush=True)
    FA_MODEL = MMS_FA.get_model().to(DEVICE)
    FA_MODEL.eval()
    FA_TOKENIZER = MMS_FA.get_tokenizer()
    FA_ALIGNER = MMS_FA.get_aligner()
    print("[INIT] MMS_FA aligner loaded successfully — word_timings enabled!", flush=True)
except Exception as fa_err:
    print(f"[WARNING] MMS_FA aligner unavailable: {fa_err}. word_timings will be omitted.", flush=True)
    FA_MODEL = None

def detect_language(text: str) -> str | None:
    if any("一" <= char <= "鿿" for char in text):
        return "Chinese"
    elif any(char in "éèêëàâäôöûüçÉÈÊËÀÂÄÔÖÛÜÇ" for char in text):
        return "French"
    elif any(char.isalpha() for char in text):
        return "English"
    return None

def process_reference_audio(job_input: dict) -> tuple[str | None, bool]:
    ref_audio_b64 = job_input.get("reference_audio_base64")
    ref_url = job_input.get("reference_url")

    if ref_audio_b64:
        audio_data = base64.b64decode(ref_audio_b64)
        fmt = job_input.get("reference_format", "wav")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt}")
        temp_file.write(audio_data)
        temp_file.close()
        return temp_file.name, True

    elif ref_url:
        suffix = Path(ref_url).suffix or ".wav"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        urllib.request.urlretrieve(ref_url, temp_file.name)
        return temp_file.name, True

    return None, False

def audio_tensor_to_base64(audio_tensor, sample_rate: int) -> str:
    buffer = io.BytesIO()
    torchaudio.save(
        buffer,
        audio_tensor.unsqueeze(0).cpu(),
        sample_rate,
        format="wav"
    )
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# =====================================================================
# WORD-LEVEL TIMINGS VIA MMS_FA FORCED ALIGNMENT
# =====================================================================
# The aligner is transcript-locked: it aligns the KNOWN text against the
# generated waveform, so timings cannot hallucinate. The transcript is the
# model-normalized text (same normalize_tts_text the processor applies), and
# words[].w carries the normalized display string — consumers render from this
# array, never from their own copy of the caller's input.

def _is_cjk(char: str) -> bool:
    return "一" <= char <= "鿿"

def _load_text_normalizer():
    """Import normalize_tts_text from the downloaded model snapshot (same file the
    processor uses), so the aligned transcript matches what was actually spoken."""
    normalizer_path = MODEL_DIR / "tts_robust_normalizer_single_script.py"
    spec = importlib.util.spec_from_file_location("tts_robust_normalizer_single_script", normalizer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_tts_text

def _strip_diacritics(text: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )

def _prepare_alignment_units(text: str, language: str | None):
    """Split normalized text into (display, align) units for MMS_FA.

    display: the normalized word/character emitted to consumers.
    align:   the Latin-only string fed to the MMS_FA tokenizer (a-z + apostrophe).

    Chinese characters become one unit each, romanized via pypinyin. Latin-script
    words are lowercased and diacritic-stripped for alignment but displayed as-is.
    Units whose align string is empty (pure punctuation/symbols) are dropped.
    Returns None when no alignable units remain (caller omits word_timings).
    """
    # Segment into whitespace-delimited words, then split mixed words into
    # CJK chars and Latin runs.
    units = []  # (display, align)

    def add_latin(run: str):
        align = "".join(c for c in _strip_diacritics(run.lower()) if c.isascii() and (c.isalpha() or c == "'"))
        if align:
            units.append((run, align))

    for raw_word in text.split():
        if not any(_is_cjk(c) for c in raw_word):
            add_latin(raw_word)
            continue
        # Mixed word: walk char-by-char, flushing Latin runs.
        latin_run = ""
        for char in raw_word:
            if _is_cjk(char):
                if latin_run:
                    add_latin(latin_run)
                    latin_run = ""
                try:
                    from pypinyin import Style, pinyin
                    syllables = pinyin(char, style=Style.NORMAL, strict=False)
                    syllable = syllables[0][0] if syllables and syllables[0] else ""
                except ImportError:
                    raise RuntimeError("pypinyin unavailable — cannot align Chinese text")
                align = "".join(c for c in syllable.lower() if c.isascii() and c.isalpha())
                if align:
                    units.append((char, align))
            else:
                latin_run += char
        if latin_run:
            add_latin(latin_run)

    return units if units else None

def extract_word_timings(audio_tensor, sample_rate: int, text: str, language: str | None):
    """Align the normalized transcript against the generated waveform with MMS_FA.

    Returns the word_timings dict, or None when there is nothing alignable.
    Raises on alignment errors — the caller catches, logs, and omits the key.
    """
    if FA_MODEL is None or FA_TOKENIZER is None or FA_ALIGNER is None:
        return None

    try:
        normalize = _load_text_normalizer()
        normalized_text = normalize(text)
    except Exception as norm_err:
        print(f"[ALIGN] Text normalizer unavailable ({norm_err}); aligning raw input text.", flush=True)
        normalized_text = text

    units = _prepare_alignment_units(normalized_text, language)
    if not units:
        return None
    display_words = [display for display, _ in units]
    align_words = [align for _, align in units]

    waveform = audio_tensor.unsqueeze(0).to(torch.float32).cpu()
    if sample_rate != FA_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=FA_SAMPLE_RATE
        )

    with torch.inference_mode():
        emissions, _ = FA_MODEL(waveform.to(DEVICE))
    emission = emissions[0].cpu()

    tokens = FA_TOKENIZER(align_words)
    token_spans = FA_ALIGNER(emission, tokens)

    # Emission-frame index -> seconds within the returned waveform.
    sec_per_frame = waveform.size(-1) / emission.size(0) / FA_SAMPLE_RATE

    words = []
    prev_start = 0.0
    for display, spans in zip(display_words, token_spans):
        if not spans:
            continue
        start = max(spans[0].start * sec_per_frame, prev_start)
        end = max(spans[-1].end * sec_per_frame, start + 1e-3)
        words.append({"w": display, "start": round(start, 3), "end": round(end, 3)})
        prev_start = start

    if not words:
        return None

    return {
        "frame_rate": round(1.0 / sec_per_frame, 2),
        "source": "mms_fa_forced_alignment",
        "words": words,
    }

def handler(job: dict):
    """RunPod Serverless Handler Function supporting Streaming, Non-Streaming, Health Ping, and VRAM Cleanup."""
    ref_file = None
    is_temp = False
    job_id = job.get("id", "unknown")
    try:
        job_input = job.get("input", {})

        # Health Check Ping Support
        if job_input.get("ping") or job_input.get("type") == "ping":
            print(f"[JOB {job_id}] Health check ping received.", flush=True)
            yield {
                "status": "healthy",
                "worker_warm": True,
                "device": DEVICE,
                "attention_impl": ATTN_IMPL
            }
            return

        text = job_input.get("text")
        if not text:
            print(f"[JOB {job_id}] Validation error: Missing required 'text' parameter.", flush=True)
            yield {"error": "Missing required field 'text' in input."}
            return

        is_streaming = job_input.get("stream", False)
        print(f"[JOB {job_id}] Execution Mode: {'STREAMING' if is_streaming else 'NON-STREAMING'}", flush=True)

        language = job_input.get("language")
        if not language:
            language = detect_language(text)
            print(f"[JOB {job_id}] Auto-detected language: {language}", flush=True)

        tokens = job_input.get("tokens")
        max_new_tokens = job_input.get("max_new_tokens", 4096)

        ref_file, is_temp = process_reference_audio(job_input)
        if ref_file:
            print(f"[JOB {job_id}] Processed reference audio file: {ref_file}", flush=True)

        kwargs = {"text": text}
        if language:
            kwargs["language"] = language
        if ref_file:
            kwargs["reference"] = [ref_file]
        if tokens:
            kwargs["tokens"] = int(tokens)

        user_msg = processor.build_user_message(**kwargs)
        conversations = [[user_msg]]

        with torch.no_grad():
            batch = processor(conversations, mode="generation")
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )

            decoded_messages = processor.decode(outputs)
            audio_tensor = decoded_messages[0].audio_codes_list[0]
            sample_rate = processor.model_config.sampling_rate

            if is_streaming:
                chunk_size = 24000  # ~1 second per chunk at 24kHz
                total_samples = audio_tensor.size(0)
                chunk_idx = 0

                for start_idx in range(0, total_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_samples)
                    chunk_tensor = audio_tensor[start_idx:end_idx]
                    chunk_b64 = audio_tensor_to_base64(chunk_tensor, sample_rate)

                    chunk_idx += 1
                    yield {
                        "status": "streaming",
                        "chunk_index": chunk_idx,
                        "audio_chunk_base64": chunk_b64,
                        "format": "wav",
                        "sample_rate": sample_rate,
                        "is_final": (end_idx == total_samples)
                    }
                return
            else:
                word_timings = None
                try:
                    word_timings = extract_word_timings(audio_tensor, sample_rate, text, language)
                except Exception as align_err:
                    print(f"[JOB {job_id}] word_timings extraction failed: {align_err}", flush=True)
                    traceback.print_exc(file=sys.stdout)

                audio_base64 = audio_tensor_to_base64(audio_tensor, sample_rate)
                payload = {
                    "status": "success",
                    "audio_base64": audio_base64,
                    "format": "wav",
                    "sample_rate": sample_rate,
                    "detected_language": language,
                }
                # Additive + optional: the key is OMITTED entirely (never empty or
                # partial) when alignment fails, so consumers fall back to
                # interpolation. A present key means "trust me".
                if word_timings:
                    payload["word_timings"] = word_timings
                yield payload
                return

    except Exception as e:
        err_msg = f"[ERROR JOB {job_id}] Inference failed: {str(e)}"
        print(err_msg, flush=True)
        traceback.print_exc(file=sys.stdout)
        yield {"error": str(e), "traceback": traceback.format_exc()}
    finally:
        if is_temp and ref_file and os.path.exists(ref_file):
            os.remove(ref_file)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Start Serverless Worker with return_aggregate_stream: True (Official RunPod Spec)
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})

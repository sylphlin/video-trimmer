---
name: video-trimmer
description: AI-Powered Smart Video Trimmer and rough-cut engine adhering to Agent Skills Specification. Features Gemini 3.8 Flash Multimodal Video Understanding (native video, no hallucinated cuts, last-take-wins), Whisper Word-Level Acoustic Ground Truth (Apple Silicon Metal GPU via mlx-whisper), Acoustic Onset Snapping (Smart Gap Shortening), 15ms Audio Equal-Power Micro-Crossfade, and multi-format NLE Project Export (Premiere Pro XML, DaVinci Resolve XML, Final Cut Pro FCPXML).
metadata:
  version: "1.0.0"
  author: "sylphlin"
  repository: "https://github.com/sylphlin/video-trimmer"
  category: "video-editing"
  specification: "https://agentskills.io/specification"
---

# Video Trimmer (`video-trimmer`)

AI-Powered Smart Video Trimmer and rough-cut engine adhering to the open [Agent Skills Specification](https://agentskills.io/specification).

Tailored for talking-head videos, tech explainers, video podcasts, tutorials, and multi-take raw footage. Combines **Gemini 3.8 Flash Multimodal Video Understanding** with local **Whisper Word-Level Acoustic Ground Truth** (with Apple Silicon Metal GPU acceleration via `mlx-whisper`), **Acoustic Onset Snapping (Smart Gap Shortening)**, and **15ms Audio Equal-Power Micro-Crossfades**.

---

## Directory Structure

This project complies with the [Agent Plugins 1.0 Specification](https://agent-plugins.org/) and the [Agent Skills Specification](https://agentskills.io/specification):

```text
video-trimmer/
├── plugin.json                       # Agent Plugins 1.0 specification manifest
├── rules/
│   └── AGENTS.md                     # Strict read-only & fail-fast operational invariants for AI clients
├── skills/
│   └── video-trimmer/
│       └── SKILL.md                  # Skill definition and agent reference manual
├── AGENTS.md                         # Permanent project invariants & developer rules (ASD-STE100 English)
├── README.md                         # Public GitHub README documentation
├── LICENSE                           # MIT License
├── .env.example                      # Environment variables template for Vertex AI & GCS
├── setup.sh                          # 100% Native gcloud GCP provisioning script (Zero Terraform)
├── pyproject.toml                    # Standard Python packaging & CLI console scripts
├── requirements.txt                  # Python runtime dependencies
├── video_trimmer.py                  # Primary CLI entrypoint forwarder
├── scripts/                          # Core implementation modules
│   ├── __init__.py
│   ├── video_trimmer.py              # Master rough-cut orchestrator & acoustic engine
│   ├── constants.py                  # Centrally managed named constants
│   ├── exceptions.py                 # Custom exception hierarchy
│   ├── acoustic.py                   # CPS calculation & text-locked acoustic bounds
│   ├── transcribe.py                 # Whisper transcription, sentence merge, clip alignment
│   ├── gemini_client.py              # Vertex AI (ADC) client & multimodal inference
│   ├── gcs_utils.py                  # Google Cloud Storage upload & ephemeral cleanup
│   ├── exporters.py                  # FCP7 XML / FCPXML / CSV generation
│   └── render.py                     # ffprobe inspection & ffmpeg final render
├── tests/                            # Offline unit tests
└── prompts/                          # Multimodal prompting specifications
    └── video_cut_prompt.md           # 5-rule subtraction & take selection prompt
```

---

## Key Capabilities & Architectural Pillars

1. **Multimodal Native Understanding (Vertex AI Gemini 3.8 Flash)**:
   - Eliminates fragile text-only editing.
   - Stages raw footage directly to Google Cloud Storage (GCS) with automatic ephemeral lifecycle cleanup.
   - Evaluates presenter eye contact, facial expressions, stuttering, and retakes simultaneously via Vertex AI.
2. **"Last Take Wins" Semantic Selection**:
   - Automatically detects presenter mistakes, line rehearsals, or multiple retakes of the same section, keeping strictly the final successful take.
3. **Multimodal Active Speaker Diarization (Gemini 3.8 Flash)**:
   - Evaluates on-camera visual cues (camera gaze, mouth articulatory sync, body language) and audio acoustics (close lavalier mic vs distant room echo).
   - Accurately differentiates the on-screen target host from off-screen crew shouting section cues (e.g. "Action", "CTA-S2", "CDA84") and casual blooper chatter between takes, eliminating cumbersome local diarization models while maintaining flawless role separation.
4. **Word-Level Acoustic Ground Truth Locking**:
   - Whisper (`mlx-whisper` on Apple Silicon Metal or `faster-whisper` on CPU/CUDA) extracts phoneme-aligned word timestamps.
   - Every cut boundary is physically anchored to acoustic reality rather than LLM timestamp approximations.
5. **Acoustic Onset Snapping (Smart Gap Shortening)**:
   - Scans the pre-speech dead air to snap cut-ins precisely **80ms before vocal cord vibration**, eliminating awkward pre-speech dead air and post-slate pauses.
6. **Word Ground Truth Tail & Plosive Defense**:
   - Strictly enforces `true_speech_end >= t_last` with forward-only tracking down to ambient room noise.
   - Accommodates voiceless consonant plosive closures (e.g. `/t/`, `/p/`, `/k/` in words like「台」) and soft trailing nasal vowels without clipping word endings.
7. **15ms Audio Equal-Power Micro-Crossfade**:
   - FFmpeg rendering injects 15ms `afade` micro-fades across all cut boundaries, completely eliminating digital pops, clicks, and background noise stepping.
8. **Production Script Injection (`--script`)**:
   - Ingests production shooting scripts (`.md` / `.txt`) to guide section-by-section matching and prevent skipping intended talking points.
9. **Universal NLE Project Export**:
   - Generates industry-standard **FCP 7 XML** (Adobe Premiere Pro & DaVinci Resolve) and **FCPXML** (Final Cut Pro X), alongside direct high-quality **MP4** renders.

---

## Standard Agent Workflow (Autonomous Pipeline Execution)

When Antigravity, Claude Code, Cursor, or any compatible agent is instructed by the user to rough-cut or trim a raw video, follow this protocol:

### Step 1: Environment Verification & GCP Native Setup
Ensure FFmpeg is installed and Google Cloud resources are provisioned (100% native gcloud, zero Terraform):
```bash
# 1. Verify FFmpeg
ffmpeg -version

# 2. Authenticate once with Google Cloud ADC
gcloud auth application-default login

# 3. One-click automated GCP provisioning (Creates bucket with 2-day auto-cleanup, SA, and .env):
./setup.sh
```

### Step 2: Execute Primary Video Trimmer Pipeline

Always pass `--agentic` by default to enable Gemini Agentic Video Understanding. Omit `--agentic` only when the user explicitly requests static mode (`static`).

```bash
# Standard automatic rough-cut (Agentic Video Understanding enabled by default):
video-trimmer -i "/path/to/raw_footage.mp4" --agentic

# If production shooting script is provided (Highly recommended for structured shows):
video-trimmer -i "raw_footage.mp4" --script "shooting_script.md" --agentic

# Fast-paced YouTube tech / science explainer pacing:
video-trimmer -i "raw_footage.mp4" --pacing compact --agentic

# Static Multimodal fallback (Use ONLY when the user explicitly requests static mode):
video-trimmer -i "raw_footage.mp4"
```

### Step 3: Fast Local Iteration (Cached EDL Workflow)
To adjust pacing, fine-tune margins, or re-render without re-incurring cloud API inference:
```bash
video-trimmer -i "take 1.mp4" --cached-json "take 1_agentic_edl.json" --suffix "fine_tuned"
```

---

## CLI Options Reference

| Option | Flag | Default | Description |
| :--- | :---: | :---: | :--- |
| `--input` | `-i` | *(Required)* | Path to input raw video file (`.mp4`, `.mov`). |
| `--output-dir` | `-o` | Same as video | Directory to save all generated output files. |
| `--model` | `-m` | `gemini-3.8-flash` | Gemini model name (defaults to `$MODEL_NAME` or `gemini-3.8-flash`). |
| `--project` | | `None` | Google Cloud Project ID (defaults to `$GOOGLE_CLOUD_PROJECT` or ADC). |
| `--region` | | `None` | Vertex AI location/region (defaults to `$GOOGLE_CLOUD_LOCATION` or `global`). |
| `--bucket` | | `None` | GCS bucket for video staging (defaults to `$VIDEO_TRIMMER_BUCKET`). |
| `--keep-gcs-upload` | | `False` | Retain ephemeral staged video in GCS instead of deleting after inference. |
| `--script` | `-s` | `None` | Path to production shooting script (`.md` / `.txt`). |
| `--agentic` | | `False` | Enable Agentic Video Understanding dynamic exploration mode. |
| `--pacing` | `-p` | `auto` | Pacing style: `auto` (adaptive CPS), `compact`, `breathing`. |
| `--cached-json`| | `None` | Path to existing EDL JSON to bypass cloud Gemini inference. |
| `--suffix` | | `None` | Custom tag suffix for generated filenames. |
| `--crf` | | `18` | FFmpeg H.264 rendering CRF parameter (18 = visually lossless). |
| `--skip-whisper`| | `False` | Skip local Whisper transcription (use pure energy fallback). |
| `--verbose` | | `False` | Verbose DEBUG-level logging (default is INFO). |

---

## Output Deliverables

For an input file `video.mp4`, the skill generates:
1. `video_<suffix>_trimmed.mp4`: High-bitrate assembled video cut with 15ms micro-fades.
2. `video_<suffix>_edl.xml`: Final Cut Pro 7 XML timeline for **Premiere Pro** & **DaVinci Resolve**.
3. `video_<suffix>_edl.fcpxml`: FCPXML timeline for **Final Cut Pro X**.
4. `video_<suffix>_edl.json`: Structured decision metadata with per-clip CPS and margins.
5. `video_<suffix>_edl.csv`: Spreadsheet table with visual/audio validation notes.
6. `video_whisper_sentences.json`: Word-level semantic sentence transcript cache.

---

## License

[MIT License](LICENSE) © 2026 sylphlin


---

## ☁️ Google Drive Direct Link & GCS Smart Caching (ADC)

`video-trimmer` natively supports passing Google Drive file links (`https://drive.google.com/file/d/.../view` or `gdrive://...`) directly to `-i / --input`:
- Authenticated 100% via Application Default Credentials (`gcloud auth application-default login`).
- Automatically checks remote MD5 (`md5Checksum`) to cache locally in `<output_dir>/gdrive_inputs/` and checks `sha256` / `gdrive_md5` metadata on `gs://${VIDEO_TRIMMER_BUCKET}/raw/` to skip redundant GCS uploads.
- Example:
  ```bash
  python3 video_trimmer.py -i "https://drive.google.com/file/d/YOUR_VIDEO_FILE_ID/view?usp=sharing" --agentic -o output/
  ```

# Video Trimmer (`video-trimmer`)

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

[English (en)](README.md) | [繁體中文 (zh-TW)](README.zh-TW.md) | [简体中文 (zh-CN)](README.zh-CN.md) | [日本語 (ja)](README.ja.md) | [한국어 (ko)](README.ko.md)

---

## Overview

**Video Trimmer** is an automated video rough-cut and trimming engine for talking-head recordings, tutorials, and presentations. It combines **Google Vertex AI Gemini 3.8 Flash** multimodal video reasoning with **Whisper Word-Level Acoustic Ground Truth** (`mlx-whisper`) and **Acoustic Onset Snapping**. Each run removes bad takes, stutters, and dead air, then exports NLE project timelines (`.xml`, `.fcpxml`, `.edl`, `.csv`) and a rendered MP4 video.

---

## Core Capabilities

1. **Multimodal Video Reasoning (`gemini-3.8-flash`)**:
   - Analyzes video and audio directly on Vertex AI via Cloud Storage (`gs://${BUCKET}/raw/`).
   - Evaluates presenter eye contact, facial expressions, stuttering, and retakes simultaneously.
2. **Last-Take-Wins Selection**:
   - Detects repeated attempts at the same script line and keeps only the final complete take.
3. **Multimodal Active Speaker Separation**:
   - Distinguishes the on-camera presenter from off-camera crew cues (`Action`, `Cut`) using visual mouth movement and microphone proximity.
4. **Acoustic Onset Snapping (`tighten_clip_to_speech`)**:
   - Scans the local audio waveform to place cut points 80 ms before vocal cord vibration, removing pre-speech silence without clipping initial phonemes.
5. **Adaptive Noise-Floor & Tail Preservation**:
   - Measures local room noise to preserve soft trailing syllables and natural breathing pauses based on presenter Characters Per Second (CPS).
6. **15 ms Audio Equal-Power Micro-Crossfade**:
   - Applies 15 ms equal-power micro-fades (`afade=t=in:d=0.015:curve=iqsin` and `afade=t=out:d=0.015:curve=oqsin`) at every cut boundary to prevent acoustic pops.
7. **Shooting Script Alignment (`--script`)**:
   - Accepts an optional markdown or text shooting script (`shooting_script.md`) to match takes against intended sections.
8. **Multi-NLE Timeline Interoperability**:
   - Exports frame-accurate **Final Cut Pro 7 XML** (`.xml` for Adobe Premiere Pro and DaVinci Resolve), **Apple Final Cut Pro FCPXML** (`.fcpxml`), **CMX 3600 EDL** (`.edl`), and **CSV** cut lists across standard frame rates (`23.976` to `60` fps).

---

## Project Structure

```text
video-trimmer/
├── plugin.json                 # Agent Plugins 1.0 manifest
├── rules/
│   └── AGENTS.md               # Read-only and fail-fast operational invariants for AI clients
├── skills/
│   └── video-trimmer/
│       └── SKILL.md            # Agent Skill specification and operational manual
├── AGENTS.md                   # Permanent project invariants and developer rules (ASD-STE100)
├── README.md                   # English documentation
├── LICENSE                     # MIT License
├── .env.example                # Environment variables template for Vertex AI and GCS
├── setup.sh                    # Native gcloud provisioning script (Zero Terraform)
├── pyproject.toml              # PEP 621 Python package configuration
├── requirements.txt            # Python dependencies
├── video_trimmer.py            # Primary CLI entrypoint forwarder
├── scripts/                    # Core engine modules
│   ├── video_trimmer.py        # CLI parser and pipeline orchestrator
│   ├── constants.py            # Named constants
│   ├── exceptions.py           # Exception hierarchy
│   ├── acoustic.py             # CPS calculation, onset snapping, and tail margins
│   ├── transcribe.py           # Whisper word-level transcription and sentence alignment
│   ├── gemini_client.py        # Vertex AI (ADC) client and multimodal inference
│   ├── gcs_utils.py            # Cloud Storage staging, Google Drive cache, and CJK filename recovery
│   ├── exporters.py            # FCP7 XML, FCPXML, EDL, and CSV timeline exporters
│   └── render.py               # ffprobe inspection and FFmpeg micro-crossfade rendering
├── prompts/
│   └── video_cut_prompt.md     # Multimodal rough-cut prompt specification
└── tests/                      # Offline unit tests
```

---

## Installation & Google Cloud Setup

### 1. Install FFmpeg

Install [FFmpeg](https://ffmpeg.org/) and verify that `ffmpeg` and `ffprobe` are in `PATH`:

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg
```

### 2. Install as an Antigravity Plugin or Local CLI

#### Option A: Install as an Antigravity Plugin (Recommended)
```bash
# Global Plugin
git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/plugins/video-trimmer

# Or Workspace Plugin
git clone https://github.com/sylphlin/video-trimmer.git .agents/plugins/video-trimmer
```

#### Option B: Install Standalone Python CLI
```bash
git clone https://github.com/sylphlin/video-trimmer.git
cd video-trimmer
pip install -r requirements.txt
pip install mlx-whisper
pip install -e .
```

### 3. Provision Google Cloud Resources (`./setup.sh`)

Run `setup.sh` to configure Vertex AI, Cloud Storage, and Application Default Credentials (ADC) using native `gcloud` commands:

```bash
# Authenticate ADC
gcloud auth application-default login

# Provision GCS bucket, CORS, two-tier lifecycle rules (raw: 2d, deliverables: 15d), IAM, and .env
chmod +x setup.sh
./setup.sh --project YOUR_GCP_PROJECT_ID --region us-central1
```

---

## Command-Line Usage

```bash
# Standard rough-cut execution (with Agentic Video Understanding)
python3 video_trimmer.py -i "raw_footage.mp4" --agentic

# Align takes against a shooting script
python3 video_trimmer.py -i "raw_footage.mp4" --script "shooting_script.md" --agentic

# Apply compact pacing for fast-paced tutorials
python3 video_trimmer.py -i "sample_take.mp4" --pacing compact --suffix "fast" --agentic

# Run in Static Multimodal mode (omit --agentic only when static sampling is desired)
python3 video_trimmer.py -i "raw_footage.mp4"

# Re-render locally from a cached EDL JSON without re-running Gemini inference
python3 video_trimmer.py -i "raw_footage.mp4" --cached-json "raw_footage_agentic_edl.json"
```

---

## CLI Options Reference

| Option | Short Flag | Default | Description |
| :--- | :---: | :---: | :--- |
| `--input` | `-i` | *(Required)* | Input video file path (`.mp4`, `.mov`), Google Drive URL, or `gs://` URI |
| `--output-dir` | `-o` | Same as input | Output directory for generated timelines and rendered video |
| `--model` | `-m` | `gemini-3.8-flash` | Vertex AI model identifier (`MODEL_NAME`) |
| `--project` | | `None` | Google Cloud Project ID (`GOOGLE_CLOUD_PROJECT`) |
| `--region` | | `global` | Vertex AI location (`GOOGLE_CLOUD_LOCATION`) |
| `--bucket` | | `None` | GCS staging bucket (`VIDEO_TRIMMER_BUCKET`) |
| `--keep-gcs-upload` | | `False` | Retain staged video in `gs://<bucket>/raw/` after inference |
| `--script` | `-s` | `None` | Shooting script file path (`shooting_script.md` or `.txt`) |
| `--agentic` | | `False` | Enable Agentic Video Understanding mode |
| `--pacing` | `-p` | `auto` | Pacing mode: `auto` (adaptive CPS), `compact`, or `breathing` |
| `--cached-json` | | `None` | Existing EDL JSON path to skip cloud inference |
| `--suffix` | | `None` | Custom filename suffix tag |
| `--crf` | | `18` | FFmpeg H.264 quality factor (`18` is visually lossless) |
| `--skip-whisper` | | `False` | Skip Whisper transcription and use energy-only onset detection |
| `--verbose` | | `False` | Enable debug logging |

---

## Generated Deliverables

For an input video `raw_footage.mp4`, the tool generates:

1. **`raw_footage_<tag>_trimmed.mp4`**: Rendered rough-cut video with 15 ms equal-power audio crossfades.
2. **`raw_footage_<tag>_edl.xml`**: Final Cut Pro 7 XML timeline for **Adobe Premiere Pro** and **DaVinci Resolve**.
3. **`raw_footage_<tag>_edl.fcpxml`**: Apple FCPXML timeline for **Final Cut Pro**.
4. **`raw_footage_<tag>_edl.json`**: Structured cut list with selected sentences, CPS values, and timestamps.
5. **`raw_footage_<tag>_edl.csv`**: Spreadsheet cut table with editorial notes.
6. **`raw_footage_whisper_sentences.json`**: Cached Whisper word-level transcript.

---

## Import Timelines into NLE Software

- **DaVinci Resolve**:
  1. Open the Media Pool and select **File -> Import -> Timeline...** (`Cmd + Shift + I` or `Ctrl + Shift + I`).
  2. Select `raw_footage_<tag>_edl.xml`.
- **Adobe Premiere Pro**:
  1. Select **File -> Import...** (`Cmd + I` or `Ctrl + I`).
  2. Select `raw_footage_<tag>_edl.xml` and open the imported sequence.
- **Final Cut Pro**:
  1. Select **File -> Import -> XML...**.
  2. Select `raw_footage_<tag>_edl.fcpxml`.

---

## Google Drive Direct Links & Two-Tier GCS Lifecycle Policy

### 1. Supported Google Drive Scenarios (`drive.readonly` ADC)

| Scenario | Input Flag Syntax | Automated Behavior |
| :--- | :--- | :--- |
| **Scenario A: Google Drive Video Rough-Cut** | `-i "https://drive.google.com/file/d/FILE_ID/view"` | Verifies remote `md5Checksum`, recovers UTF-8 CJK filenames, caches in `gdrive_inputs/`, runs local Whisper word-level timing, and stages to GCS `raw/`. |
| **Scenario B: Script-Guided Cloud Rough-Cut** | `-i "https://drive.google.com/file/d/FILE_ID/view" -s shooting_script.md --agentic` | Matches retakes against `shooting_script.md`, keeps the final valid take, tightens speech boundaries, and exports `.mp4`, `.xml`, and `.fcpxml`. |
| **Scenario C: Direct GCS URI Input** | `-i "gs://video-preprocessing-PROJECT_ID/raw/raw_footage.mp4"` | References the existing GCS object in Vertex AI without re-uploading. |

### 2. Two-Tier GCS Bucket Lifecycle Policy (`gs://video-preprocessing-${PROJECT_ID}`)

| GCS Prefix (`matchesPrefix`) | Stored Objects | Retention (`age`) | Purpose |
| :--- | :--- | :--- | :--- |
| **`raw/`** | Staged raw videos (`raw/<filename>.mp4`) | **2 Days (`age: 2`)** | Keeps temporary staged media for 2 days as a safety backstop (normal runs delete staged objects immediately in `finally` blocks). |
| **`output/`**, **`deliverables/`**, **`trimmed/`** | Trimmed videos, XML/FCPXML timelines, and EDL reports | **15 Days (`age: 15`)** | Retains deliverables for 15 days for team review before automatic deletion. |

---

## Unit Testing

Run the offline test suite before committing changes:

```bash
python3 -m unittest discover -s tests -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).

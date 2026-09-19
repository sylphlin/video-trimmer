# Video Trimmer (`video-trimmer`)

> **AI-Powered Smart Video Trimmer & Rough-Cut Engine**
> 
> An intelligent, end-to-end video rough-cut and trimming tool tailored for talking-head videos, video podcasts, tutorials, and speeches.
> Powered by **Gemini 3.8 Flash Multimodal Video Understanding**, **Whisper Word-Level Acoustic Ground Truth** (with Apple Silicon Metal acceleration via `mlx-whisper`), and **Acoustic Onset Snapping (Smart Gap Shortening)**.

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

[English (en)](README.md) | [繁體中文 (zh-TW)](README.zh-TW.md) | [简体中文 (zh-CN)](README.zh-CN.md) | [日本語 (ja)](README.ja.md) | [한국어 (ko)](README.ko.md)

---

## 🌟 Key Highlights & Innovations

1. **Multimodal Native Understanding (Gemini 3.8 Flash)**:
   - Does not rely on brittle ASR text matching alone.
   - Uploads raw footage directly via Gemini Files API / Interactions API to evaluate **presenter eye contact, facial expressions, stuttering, and retakes** simultaneously.
2. **Intelligent "Last Take Wins" Selection**:
   - Automatically detects presenter mistakes, line rehearsals, or multiple retakes of the same section, keeping strictly the final successful take.
3. **Multimodal Active Speaker Diarization (Gemini 3.8 Flash)**:
   - Seamlessly evaluates on-camera visual cues (camera gaze, mouth articulatory sync, body language) and audio acoustics (close lavalier mic vs distant room echo).
   - Accurately differentiates the on-screen target host from off-screen crew shouting section cues (e.g. "Action", "CTA-S2", "CDA84") and casual blooper chatter between takes, eliminating cumbersome local diarization models while maintaining flawless role separation.
4. **Acoustic Onset Snapping (Smart Gap Shortening)**:
   - Whisper timestamps often trigger 0.3s ~ 0.8s before the speaker actually begins vocalizing. `video-trimmer` dynamically scans the waveform to snap cuts precisely **80ms before vocal cord vibration**, eliminating awkward pre-speech dead air.
5. **Adaptive Noise-Floor & Tail Tracking**:
   - Dynamically analyzes localized ambient room noise to preserve subtle nasal endings (e.g. Japanese 「〜ございます」, nasal vowels, fading words) without clipping word tails.
6. **15ms Audio Equal-Power Micro-Crossfade**:
   - Automatically injects 15ms micro-fades at every cut boundary in FFmpeg rendering, completely eliminating digital pops, clicks, and background noise stepping.
7. **Production Script Injection (`--script`)**:
   - Optional support for feeding production shooting scripts or subtitles to guide section-by-section matching and avoid missing intended sections.
8. **One-Click NLE Project Export**:
   - Generates industry-standard **FCP 7 XML** (Adobe Premiere Pro & DaVinci Resolve) and **FCPXML** (Final Cut Pro X), plus direct high-quality **MP4** renders.

---

## 📂 Project Structure

This project complies with the [Agent Plugins 1.0 Specification](https://agent-plugins.org/) and the [Agent Skills Specification](https://agentskills.io/specification):

```text
video-trimmer/
├── plugin.json                 # Agent Plugins 1.0 specification manifest
├── rules/
│   └── AGENTS.md               # Strict read-only & fail-fast operational invariants for AI clients
├── skills/
│   └── video-trimmer/
│       └── SKILL.md            # Standard Agent Skill specification & agent manual
├── AGENTS.md                   # Permanent project invariants & developer rules (ASD-STE100 English)
├── README.md                   # Public GitHub documentation
├── LICENSE                     # MIT License
├── .env.example                # Environment variables template for Vertex AI & GCS
├── setup.sh                    # 100% Native gcloud GCP provisioning script (Zero Terraform)
├── pyproject.toml              # Modern PEP 621 Python packaging & console scripts
├── requirements.txt            # Python dependencies
├── video_trimmer.py            # Primary CLI entrypoint forwarder
├── auto_rough_cut.py           # Backward compatibility wrapper
├── scripts/                    # Core engine modules
│   ├── __init__.py
│   ├── video_trimmer.py        # CLI (argparse) + pipeline orchestration (main())
│   ├── constants.py            # Centrally managed named constants
│   ├── exceptions.py           # Custom exception hierarchy
│   ├── acoustic.py             # CPS, dynamic margins, text-locked acoustic bounds
│   ├── transcribe.py           # Whisper transcription, sentence merge, clip alignment
│   ├── gemini_client.py        # Vertex AI (ADC) client & multimodal inference
│   ├── gcs_utils.py            # Google Cloud Storage upload & ephemeral cleanup
│   ├── exporters.py            # FCP7 XML / FCPXML / CSV generation
│   └── render.py               # ffprobe inspection & ffmpeg final render
├── tests/                      # Offline unit tests
├── prompts/
│   └── video_cut_prompt.md     # Core multimodal prompting specification
└── examples/                   # Sample project outputs (EDL, XML, FCPXML, JSON)
```

---

## 🚀 Quick Start

### 1. System Requirements

Ensure [FFmpeg](https://ffmpeg.org/) is installed and available in your `PATH`:

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg
```

### 2. Installation & Deployment

#### Method A: Google Antigravity & Agent Plugins 1.0 Installation (Recommended for AI Agents)

Install directly into Google Antigravity or any [Agent Plugins 1.0](https://agent-plugins.org/) compatible client as an Agent Plugin/Skill:

1. **Install as an Agent Plugin (Recommended - automatically loads `plugin.json` and strict read-only execution invariants in `rules/AGENTS.md`)**:
   - **Global Plugin** (available across all projects and workspaces, recommended):
     ```bash
     git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/plugins/video-trimmer
     ```
   - **Workspace Plugin** (scoped to current workspace):
     ```bash
     git clone https://github.com/sylphlin/video-trimmer.git .agents/plugins/video-trimmer
     ```

#### Method B: Standalone Python CLI Installation

Clone the repository and install dependencies locally:

```bash
git clone https://github.com/sylphlin/video-trimmer.git
cd video-trimmer

# Install core dependencies
pip install -r requirements.txt

# (Recommended for macOS Apple Silicon) Install mlx-whisper for Metal GPU acceleration
pip install mlx-whisper

# Or install as an editable CLI tool
pip install -e .
```

### 3. Google Cloud Native Setup (100% Native gcloud, Zero Terraform)

This project exclusively uses **Google Cloud Vertex AI** with **Application Default Credentials (ADC)** and **Google Cloud Storage (GCS)** for video staging.

All cloud resources (GCS bucket, CORS, 2-day ephemeral auto-cleanup, dedicated Service Account, and least-privilege IAM bindings) are provisioned natively using `gcloud`—**zero external tool or Terraform dependencies, 100% Cloud Shell ready**.

#### Option A: One-Click Automated Setup (Recommended)

Run the included automated provisioning script:

```bash
# Make script executable (first time only)
chmod +x setup.sh

# Automatic setup (reads existing gcloud project and configures .env):
./setup.sh

# Or explicitly specify project and region:
./setup.sh --project YOUR_PROJECT_ID --region us-central1

# Dry-run preview:
./setup.sh --dry-run
```

The script automatically:
1. Provisions/verifies GCS bucket `gs://video-preprocessing-${PROJECT_ID}` with `--uniform-bucket-level-access` and `--public-access-prevention`.
2. Configures CORS (24-hour cache, `GET`/`HEAD`) for signed URL video playback.
3. Configures Lifecycle Rules on `raw/` for **2-day ephemeral auto-cleanup**, preventing cloud storage clutter.
4. Creates dedicated service account `video-trimmer-sa` with least-privilege permissions (`roles/storage.objectUser`, `roles/aiplatform.user`, `roles/logging.logWriter`).
5. Configures your local `.env` file automatically.

#### Option B: Manual Setup via Native gcloud Commands

If you prefer configuring resources manually in your shell:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export BUCKET_NAME="video-preprocessing-${PROJECT_ID}"
export SA="video-trimmer-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# 1. Create GCS Staging Bucket
gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access \
    --public-access-prevention

# 2. Configure 2-Day Ephemeral Auto-Cleanup for Staged Video
cat << 'EOF' > /tmp/lifecycle.json
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {
        "age": 2,
        "matchesPrefix": ["raw/"]
      }
    }
  ]
}
EOF
gcloud storage buckets update "gs://${BUCKET_NAME}" --lifecycle-file=/tmp/lifecycle.json

# 3. Create Service Account & Grant Least-Privilege IAM Roles
gcloud iam service-accounts create video-trimmer-sa \
    --display-name="Video Trimmer Service Account" \
    --project="${PROJECT_ID}"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA}" \
    --role="roles/aiplatform.user"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
    --member="serviceAccount:${SA}" \
    --role="roles/storage.objectUser"

# 4. Authenticate locally with Application Default Credentials (ADC)
gcloud auth application-default login

# 5. Configure local .env
cp .env.example .env
# Set GOOGLE_CLOUD_PROJECT, VIDEO_TRIMMER_BUCKET, etc. in .env
```

---

## 💻 Usage

Run the trimmer on any raw recording:

```bash
# Using the installed CLI command:
video-trimmer -i "/path/to/raw_footage.mp4"

# Or directly with Python:
python video_trimmer.py -i "/path/to/raw_footage.mp4"
```

### Advanced Examples

```bash
# 1. Supply a production script for guided section alignment
video-trimmer -i "take 1.mp4" --script "script.md"

# 2. Enable Agentic Video Understanding (dynamic multi-turn exploration)
video-trimmer -i "interview.mp4" --agentic

# 3. Compact pacing style for fast-paced YouTube tech / science explainers
video-trimmer -i "news.mp4" --pacing compact --suffix "fast"

# 4. Re-render / re-cut from an existing cached decision JSON (instant local processing)
video-trimmer -i "take 1.mp4" --cached-json "take 1_agentic_edl.json" --suffix "fine_tuned"
```

---

## ⚙️ CLI Options Reference

| Option | Flag | Default | Description |
| :--- | :---: | :---: | :--- |
| `--input` | `-i` | *(Required)* | Path to input raw video file (`.mp4`, `.mov`). |
| `--output-dir` | `-o` | Same as video | Directory to save all output files. |
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

## 📦 Generated Output Files

For an input file `take 1.mp4`, `video-trimmer` produces:

1. **`take 1_<tag>_trimmed.mp4`**: Fully assembled cut video with equal-power audio micro-fades.
2. **`take 1_<tag>_edl.xml`**: Standard FCP 7 XML for **Adobe Premiere Pro** and **DaVinci Resolve**.
3. **`take 1_<tag>_edl.fcpxml`**: Apple FCPXML for **Final Cut Pro X**.
4. **`take 1_<tag>_edl.json`**: Structured decisions including selected sentences, CPS, and timing.
5. **`take 1_<tag>_edl.csv`**: Spreadsheet-ready table with visual/audio validation notes.
6. **`take 1_whisper_sentences.json`**: Full semantic transcript with word-level boundaries.
7. **`usage_log.jsonl`** (in `--output-dir`): One appended JSON line per Gemini API call, recording token usage and call duration for manual cost tracking.

---

## 🎬 Importing into NLE Software

- **DaVinci Resolve**:
  1. Right click in Media Pool -> `Timelines` -> `Import` -> `AAF / EDL / XML...` (`Ctrl+Shift+I` / `Cmd+Shift+I`).
  2. Select the generated `_edl.xml`. The trimmed sequence is instantly created with links to your source video.
- **Adobe Premiere Pro**:
  1. `File` -> `Import...` (`Cmd+I` / `Ctrl+I`).
  2. Select `_edl.xml`. Double-click the resulting sequence in the project bin.
- **Final Cut Pro X**:
  1. `File` -> `Import` -> `XML...`.
  2. Select `_edl.fcpxml`.

---

---

## ☁️ Google Drive Direct Links & GCS Lifecycle Policy (100% ADC Integration)

In real-world workflows, raw single-camera takes are often uploaded directly to **Google Drive (My Drive or Shared Drives)**. `video-trimmer` natively accepts Google Drive share links via `gcloud` ADC (`drive.readonly` scope) with smart MD5/SHA-256 caching:

### 1. One-Click Cloud & Google Drive Setup (`./setup.sh`)
```bash
# Step 1: Authenticate ADC with Google Drive Read-Only scope
gcloud auth application-default login

# Step 2: Provision Vertex AI / GCS / Drive APIs, bucket, and two-tier Lifecycle rules
./setup.sh --project YOUR_GCP_PROJECT_ID
```

### 2. 📌 Supported Google Drive Scenarios & Examples

| Scenario | Input Flag Syntax | Smart Caching & Automated Behavior |
| :--- | :--- | :--- |
| **Scenario A: Direct Google Drive Raw Footage Rough-Cut**<br/>*(Zero manual browser download)* | `-i "https://drive.google.com/file/d/<FILE_ID>/view"`<br/>or `-i "gdrive://<FILE_ID>"` | Verifies remote `md5Checksum` via Drive API v3, caches locally in `<output_dir>/gdrive_inputs/` for local Whisper word-level ground truth & FFmpeg rendering, and stages to GCS `raw/` (skipping GCS upload if `sha256`/`gdrive_md5` already matches). |
| **Scenario B: Script-Guided Cloud Rough-Cut**<br/>*(Last-Take-Wins with shooting script)* | `-i "<GDRIVE_VIDEO_LINK>"`<br/>`-s script.md --agentic` | Matches retakes against your shooting script, keeps the best final take, snaps silences, and exports `.mp4`, `.xml` (Premiere/Resolve), and `.fcpxml` (Final Cut Pro). |
| **Scenario C: Direct GCS URI Input** | `-i "gs://video-preprocessing-proj/raw/take1.mp4"` | Directly references the existing GCS object in Vertex AI without re-uploading. |

#### 💻 Practical CLI Examples:
```bash
# [Scenario A] Trim a raw video directly from a Google Drive share link using Agentic Video mode:
python3 video_trimmer.py \
  -i "https://drive.google.com/file/d/1RawTakeVideoIdxxxxxx/view?usp=sharing" \
  --agentic -o output/

# [Scenario B] Google Drive raw footage + shooting script alignment + compact pacing:
python3 video_trimmer.py \
  -i "https://drive.google.com/file/d/1RawTakeVideoIdxxxxxx/view?usp=sharing" \
  -s shooting_script.md --pacing compact --agentic -o output/
```

#### 💬 Antigravity Agent Conversational Prompt Example:
> *"Trim the bad takes, stutters, and dead air from this raw footage on Google Drive, and export a DaVinci Resolve / Premiere XML timeline plus trimmed MP4: `https://drive.google.com/file/d/1RawTakeVideoIdxxxxxx/view?usp=sharing`"*

### 3. 🗑️ Two-Tier GCS Bucket Lifecycle Policy (`raw/` 2 Days / Deliverables 15 Days)

| GCS Path Prefix (`matchesPrefix`) | Stored Assets | Retention Period (`age`) | Rationale |
| :--- | :--- | :--- | :--- |
| **`raw/`** | Staged raw videos (`raw/<filename>.mp4`) | **2 Days (`age: 2`)** | Retains staged media for 2 days so repeated runs hit the `sha256` / `gdrive_md5` cache instantaneously, then auto-deletes. |
| **`output/`**, **`deliverables/`**, **`trimmed/`** | Trimmed videos, NLE XML/FCPXML timelines, EDL reports | **15 Days (`age: 15`)** | Retains deliverables for 15 days for team review before automatic cleanup. |

---

## 🧪 Development & Testing

Offline unit tests (synthetic audio, handwritten fixtures — no real video/API calls needed):

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## 📄 License

[MIT License](LICENSE) © 2026 sylphlin

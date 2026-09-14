# Video Trimmer (`video-trimmer`)

> **AI-Powered Smart Video Trimmer & Rough-Cut Engine**
> 
> An intelligent, end-to-end video rough-cut and trimming tool tailored for talking-head videos, video podcasts, tutorials, and speeches.
> Powered by **Gemini 3.8 Flash Multimodal Video Understanding**, **Whisper Word-Level Acoustic Ground Truth** (with Apple Silicon Metal acceleration via `mlx-whisper`), and **Acoustic Onset Snapping (Smart Gap Shortening)**.

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

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

This project complies with the [Agent Skills Specification](https://agentskills.io/specification):

```text
video-trimmer/
├── SKILL.md                  # Standard Agent Skill specification & agent manual
├── README.md                 # Public GitHub documentation
├── LICENSE                   # MIT License
├── .env.example              # Environment variables template for Vertex AI & GCS
├── pyproject.toml            # Modern PEP 621 Python packaging & console scripts
├── requirements.txt          # Python dependencies
├── video_trimmer.py          # Primary CLI entrypoint forwarder
├── auto_rough_cut.py         # Backward compatibility wrapper
├── scripts/                  # Core engine modules
│   ├── __init__.py
│   ├── video_trimmer.py      # CLI (argparse) + pipeline orchestration (main())
│   ├── constants.py          # Centrally managed named constants
│   ├── exceptions.py         # Custom exception hierarchy
│   ├── acoustic.py           # CPS, dynamic margins, text-locked acoustic bounds
│   ├── transcribe.py         # Whisper transcription, sentence merge, clip alignment
│   ├── gemini_client.py      # Vertex AI (ADC) client & multimodal inference
│   ├── gcs_utils.py          # Google Cloud Storage upload & ephemeral cleanup
│   ├── exporters.py          # FCP7 XML / FCPXML / CSV generation
│   └── render.py             # ffprobe inspection & ffmpeg final render
├── tests/                    # Offline unit tests
├── prompts/
│   └── video_cut_prompt.md   # Core multimodal prompting specification
└── examples/                 # Sample project outputs (EDL, XML, FCPXML, JSON)
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

### 2. Installation

Clone the repository and install dependencies:

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

### 3. Authentication & Configuration (Vertex AI + ADC)

This project exclusively uses **Google Cloud Vertex AI** with **Application Default Credentials (ADC)** and **Google Cloud Storage (GCS)** for video staging:

1. **Authenticate once with Google Cloud**:
   ```bash
   gcloud auth application-default login
   ```

2. **Configure your environment**:
   Copy `.env.example` to `.env` (or configure `~/.gemini/.env`):
   ```bash
   cp .env.example .env
   ```
   Set your Google Cloud project and staging bucket:
   ```bash
   GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   GOOGLE_CLOUD_LOCATION=global
   VIDEO_TRIMMER_BUCKET=your-gcs-bucket-name
   MODEL_NAME=gemini-3.8-flash
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

## 🧪 Development & Testing

Offline unit tests (synthetic audio, handwritten fixtures — no real video/API calls needed):

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## 📄 License

[MIT License](LICENSE) © 2026 sylphlin

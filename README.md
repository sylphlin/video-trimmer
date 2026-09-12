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
3. **Acoustic Onset Snapping (Smart Gap Shortening)**:
   - Whisper timestamps often trigger 0.3s ~ 0.8s before the speaker actually begins vocalizing. `video-trimmer` dynamically scans the waveform to snap cuts precisely **80ms before vocal cord vibration**, eliminating awkward pre-speech dead air.
4. **Adaptive Noise-Floor & Tail Tracking**:
   - Dynamically analyzes localized ambient room noise to preserve subtle nasal endings (e.g. Japanese 「〜ございます」, nasal vowels, fading words) without clipping word tails.
5. **15ms Audio Equal-Power Micro-Crossfade**:
   - Automatically injects 15ms micro-fades at every cut boundary in FFmpeg rendering, completely eliminating digital pops, clicks, and background noise stepping.
6. **Production Script Injection (`--script`)**:
   - Optional support for feeding production shooting scripts or subtitles to guide section-by-section matching and avoid missing intended sections.
7. **One-Click NLE Project Export**:
   - Generates industry-standard **FCP 7 XML** (Adobe Premiere Pro & DaVinci Resolve) and **FCPXML** (Final Cut Pro X), plus direct high-quality **MP4** renders.

---

## 📂 Project Structure

```text
video-trimmer/
├── pyproject.toml            # Modern Python package configuration
├── requirements.txt          # Python dependencies
├── video_trimmer.py          # Primary CLI trimmer program
├── auto_rough_cut.py         # Backward compatibility wrapper
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

### 3. Configure Gemini API Key

Get an API Key from [Google AI Studio](https://aistudio.google.com/):

```bash
export GEMINI_API_KEY="your_api_key_here"

# Or save to ~/.gemini/.env
echo 'GEMINI_API_KEY="your_api_key_here"' >> ~/.gemini/.env
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
| `--script` | `-s` | `None` | Path to production shooting script (`.md` / `.txt`). |
| `--agentic` | | `False` | Enable Google Interactions API Agentic Video Understanding mode. |
| `--pacing` | `-p` | `auto` | Pacing style: `auto` (adaptive CPS), `compact`, `breathing`. |
| `--cached-json`| | `None` | Path to existing EDL JSON to bypass cloud Gemini inference. |
| `--suffix` | | `None` | Custom tag suffix for generated filenames. |
| `--crf` | | `18` | FFmpeg H.264 rendering CRF parameter (18 = visually lossless). |
| `--skip-whisper`| | `False` | Skip local Whisper transcription (use pure energy fallback). |

---

## 📦 Generated Output Files

For an input file `take 1.mp4`, `video-trimmer` produces:

1. **`take 1_<tag>_trimmed.mp4`**: Fully assembled cut video with 15ms audio micro-fades.
2. **`take 1_<tag>_edl.xml`**: Standard FCP 7 XML for **Adobe Premiere Pro** and **DaVinci Resolve**.
3. **`take 1_<tag>_edl.fcpxml`**: Apple FCPXML for **Final Cut Pro X**.
4. **`take 1_<tag>_edl.json`**: Structured decisions including selected sentences, CPS, and timing.
5. **`take 1_<tag>_edl.csv`**: Spreadsheet-ready table with visual/audio validation notes.
6. **`take 1_whisper_sentences.json`**: Full semantic transcript with word-level boundaries.

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

## 📄 License

[MIT License](LICENSE) © 2026 sylphlin

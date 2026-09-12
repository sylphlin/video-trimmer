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

This skill strictly complies with the [Agent Skills Specification](https://agentskills.io/specification):

```text
video-trimmer/
├── SKILL.md                          # Skill definition and agent reference manual
├── README.md                         # Public GitHub README documentation
├── LICENSE                           # MIT License
├── pyproject.toml                    # Standard Python packaging & CLI console scripts
├── requirements.txt                  # Python runtime dependencies
├── video_trimmer.py                  # Primary CLI entrypoint forwarder
├── auto_rough_cut.py                 # Backward-compatibility CLI wrapper
├── scripts/                          # Core implementation modules
│   ├── __init__.py
│   └── video_trimmer.py              # Master rough-cut orchestrator & acoustic engine
├── prompts/                          # Multimodal prompting specifications
│   └── video_cut_prompt.md           # 5-rule subtraction & take selection prompt
└── examples/                         # Verified project outputs & shooting scripts
    ├── take 1_script.md              # Production shooting script example
    ├── take1_edl.json                # Structured decision JSON
    ├── take1_edl.csv                 # Spreadsheet decision table
    ├── take1_edl.xml                 # FCP 7 XML (Premiere Pro & DaVinci Resolve)
    └── take1_edl.fcpxml              # Final Cut Pro X FCPXML
```

---

## Key Capabilities & Architectural Pillars

1. **Multimodal Native Understanding (Gemini 3.8 Flash)**:
   - Eliminates fragile text-only editing.
   - Uploads raw footage directly via Gemini Files API / Interactions API to evaluate **presenter eye contact, facial expressions, stuttering, and retakes** simultaneously.
2. **"Last Take Wins" Semantic Selection**:
   - Automatically detects presenter mistakes, line rehearsals, or multiple retakes of the same section, keeping strictly the final successful take.
3. **Word-Level Acoustic Ground Truth Locking**:
   - Whisper (`mlx-whisper` on Apple Silicon Metal or `faster-whisper` on CPU/CUDA) extracts phoneme-aligned word timestamps.
   - Every cut boundary is physically anchored to acoustic reality rather than LLM timestamp approximations.
4. **Acoustic Onset Snapping (Smart Gap Shortening)**:
   - Scans the pre-speech dead air to snap cut-ins precisely **80ms before vocal cord vibration**, eliminating awkward pre-speech dead air and post-slate pauses.
5. **Word Ground Truth Tail & Plosive Defense**:
   - Strictly enforces `true_speech_end >= t_last` with forward-only tracking down to ambient room noise.
   - Accommodates voiceless consonant plosive closures (e.g. `/t/`, `/p/`, `/k/` in words like「台」) and soft trailing nasal vowels without clipping word endings.
6. **15ms Audio Equal-Power Micro-Crossfade**:
   - FFmpeg rendering injects 15ms `afade` micro-fades across all cut boundaries, completely eliminating digital pops, clicks, and background noise stepping.
7. **Production Script Injection (`--script`)**:
   - Ingests production shooting scripts (`.md` / `.txt`) to guide section-by-section matching and prevent skipping intended talking points.
8. **Universal NLE Project Export**:
   - Generates industry-standard **FCP 7 XML** (Adobe Premiere Pro & DaVinci Resolve) and **FCPXML** (Final Cut Pro X), alongside direct high-quality **MP4** renders.

---

## Standard Agent Workflow (Autonomous Pipeline Execution)

When Antigravity, Claude Code, Cursor, or any compatible agent is instructed by the user to rough-cut or trim a raw video, follow this protocol:

### Step 1: Environment Verification
Ensure FFmpeg is installed and `GEMINI_API_KEY` is configured:
```bash
ffmpeg -version
export GEMINI_API_KEY="your_api_key_here"
```

### Step 2: Execute Primary Video Trimmer Pipeline

```bash
# Standard automatic rough-cut (Auto adaptive pacing based on presenter CPS)
video-trimmer -i "/path/to/raw_video.mp4"

# If production shooting script is provided (Highly recommended for structured shows):
video-trimmer -i "take 1.mp4" --script "script.md"

# Agentic Video Understanding mode (Dynamic multi-turn exploration for complex takes):
video-trimmer -i "take 1.mp4" --agentic

# Fast-paced YouTube tech / science explainer pacing:
video-trimmer -i "news.mp4" --pacing compact
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
| `--script` | `-s` | `None` | Path to production shooting script (`.md` / `.txt`). |
| `--agentic` | | `False` | Enable Google Interactions API Agentic Video Understanding mode. |
| `--pacing` | `-p` | `auto` | Pacing style: `auto` (adaptive CPS), `compact`, `breathing`. |
| `--cached-json`| | `None` | Path to existing EDL JSON to bypass cloud Gemini inference. |
| `--suffix` | | `None` | Custom tag suffix for generated filenames. |
| `--crf` | | `18` | FFmpeg H.264 rendering CRF parameter (18 = visually lossless). |
| `--skip-whisper`| | `False` | Skip local Whisper transcription (use pure energy fallback). |

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

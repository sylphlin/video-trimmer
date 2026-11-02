---
name: video-trimmer
description: AI-Powered Smart Video Trimmer and rough-cut engine adhering to Agent Skills Specification. Features Gemini 3.8 Flash Multimodal Video Understanding (native video, Last-Take-Wins with Composite Pick-Up Backfill), Dual-Engine Word-Level Acoustic Ground Truth (Google Cloud Speech-to-Text V2 Chirp 3 default; Whisper optional), Acoustic Onset Snapping, 20ms Audio Equal-Power Micro-Crossfade, and multi-format NLE Project Export (Premiere Pro XML, DaVinci Resolve XML, Final Cut Pro FCPXML).
metadata:
  version: "1.0.0"
  author: "sylphlin"
  repository: "https://github.com/sylphlin/video-trimmer"
  category: "video-editing"
  specification: "https://agentskills.io/specification"
---

# Video Trimmer (`video-trimmer`)

AI-Powered Smart Video Trimmer and rough-cut engine adhering to the open [Agent Skills Specification](https://agentskills.io/specification).

Tailored for talking-head videos, tech explainers, video podcasts, tutorials, and multi-take raw footage. Combines **Gemini 3.8 Flash Multimodal Video Understanding**, **Last-Take-Anchored Composite Sub-Span Backfill**, and **Dual-Engine Word-Level Acoustic Ground Truth** (**Google Cloud Speech-to-Text V2 Chirp 3** by default; local `mlx-whisper` / `faster-whisper` optional), with **Acoustic Onset Snapping**, **Composite Loudness Gain Matching**, and **20ms Audio Equal-Power Micro-Crossfades**.

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
│       ├── SKILL.md                  # Skill definition and agent reference manual
│       ├── scripts/                  # Canonical core implementation modules (SSOT)
│       │   ├── __init__.py
│       │   ├── video_trimmer.py      # Master rough-cut orchestrator & acoustic engine
│       │   ├── constants.py          # Centrally managed named constants
│       │   ├── exceptions.py         # Custom exception hierarchy
│       │   ├── chirp3_asr.py         # Cloud Speech-to-Text V2 (Chirp 3) word-timestamp engine
│       │   ├── acoustic.py           # CPS, onset clamping, overlap pivots, gain matching & speech bounds
│       │   ├── transcribe.py         # Dual-engine ASR, CJK intra-token slicing & sub-unit resolution
│       │   ├── edl_auditor.py        # Right-to-Left Span Occlusion, micro-window repair & 8D audit
│       │   ├── gemini_client.py      # Vertex AI (ADC) client & multimodal inference
│       │   ├── gcs_utils.py          # Google Cloud Storage upload & ephemeral cleanup
│       │   ├── exporters.py          # FCP7 XML / FCPXML v1.9 / CSV generation with composite markers
│       │   └── render.py             # ffprobe inspection & ffmpeg render with optional 1.12x punch-in
│       └── prompts/                  # Canonical multimodal prompting specifications (SSOT)
│           └── video_cut_prompt.md   # Dual-mode take arbitration & composite pick-up backfill prompt
├── AGENTS.md                         # Workspace & engineering development rules (ASD-STE100 English)
├── README.md                         # Public GitHub README documentation
├── LICENSE                           # MIT License
├── .env.example                      # Environment variables template for Vertex AI, STT V2 & GCS
├── setup.sh                          # 100% Native gcloud GCP provisioning script (Zero Terraform)
├── pyproject.toml                    # Standard Python packaging & CLI console scripts
├── requirements.txt                  # Python runtime dependencies
└── tests/                            # Offline unit tests
```

---

## Key Capabilities & Architectural Pillars

1. **Multimodal Native Understanding (Vertex AI Gemini 3.8 Flash)**:
   - Stages raw footage directly to Google Cloud Storage (GCS) with automatic ephemeral lifecycle cleanup.
   - Evaluates presenter eye contact, facial expressions, stuttering, and retakes simultaneously via Vertex AI.
2. **Dual-Mode Take Arbitration & Last-Take-Anchored Composite Backfill**:
   - **Mode A (`--script`)**: Monotonic `[Script Block NN]` alignment keeping at most one winning take per script block.
   - **Mode B (No `--script`)**: Unscripted Intent-Window arbitration pruning abandoned fragments while preserving intentional rhetorical repetition.
   - **Partial Tail Pick-Up Backfill (`Right-to-Left Span Occlusion`)**: Keeps 100% `Last-Take-Wins` on full restarts, while automatically backfilling the missing opening prefix (`composite_role: "prefix"`) from the preceding attempt when the presenter only re-records a partial tail pick-up (`composite_role: "continuation"`, minimum `>= 0.55s` and `>= 4` CJK chars / `>= 3` Latin words) or excises a mid-sentence stumble (`A + B_ng + B_ok + C`).
3. **Multimodal Active Speaker Diarization (Gemini 3.8 Flash)**:
   - Evaluates on-camera visual cues (camera gaze, mouth articulatory sync, body language) and audio acoustics (close lavalier mic vs distant room echo).
   - Differentiates the on-screen presenter from off-screen crew cues (`Action`, `Scene 1`, `Take 2`) and between-take chatter.
4. **Dual-Engine Word-Level Acoustic Ground Truth (Chirp 3 Default + Whisper Optional)**:
   - Defaults to **Google Cloud Speech-to-Text V2 `chirp_3`** (`--asr-engine chirp3`, `0.001s` millisecond word offsets, automatic punctuation, 18-minute overlapping chunk stitching, and strict Fail-Fast on Cloud/Auth errors without silent Whisper downgrade).
   - Supports optional local **Whisper** (`--asr-engine whisper` via `mlx-whisper` on Apple Silicon Metal or `faster-whisper` on CPU/CUDA) when offline/local ASR is explicitly requested.
   - Supports CJK multi-character intra-token proportional interpolation and Global Cross-Clip Coalescing (`coalesce_adjacent_sub_units`) across continuous non-composite clips.
5. **Acoustic Onset Snapping, Post-Pause Onset Clamping, & Plosive Tail Defense**:
   - Clamps inflated post-pause word starts (`clamp_post_pause_word_onset`), snaps cut-ins before vocal cord vibration, applies compact `40ms` margins on composite seams, and enforces `next_word_hard_wall` locks so excised stumbles never bleed into retained audio.
6. **Composite Loudness Gain Matching, 20ms Micro-Crossfades, & Optional `1.12x` Punch-In**:
   - Matches voiced RMS loudness across composite pick-up seams (`compute_composite_gain_db`), applies 20ms `afade` equal-power micro-fades (`iqsin`/`qsin`), and optionally applies an even-pixel `1.12x` framing punch-in (`--punch-in-composite`) in both MP4 renders and NLE timelines.
7. **Pre-Render Surgical Micro-Window Repair & 8-Dimension Quality Audit (`edl_auditor.py`)**:
   - Detects missing script blocks, partial script block backfills (`PARTIAL_SCRIPT_BLOCK_BACKFILL`), unanchored clips, and retake collisions before rendering, re-scanning only the affected `15s–90s` slice via `VideoMetadata(start_offset=..., end_offset=...)`.
   - Outputs `<base>_<tag>_edl_report.md` and `<base>_<tag>_edl_report.json` with a top-level `agent_verdict`.
8. **Universal NLE Project Export with Composite Markers**:
   - Generates frame-accurate **FCP 7 XML** (Premiere Pro & DaVinci Resolve) and **FCPXML v1.9** (Final Cut Pro) with `Composite Splice (Pick-up)` markers, alongside CSV and hardware-accelerated MP4 renders.

---

## Standard Agent Workflow (Autonomous Pipeline Execution)

When an AI agent is instructed to rough-cut or trim a raw video, follow this protocol directly.
Resolve `<PLUGIN_ROOT>` as the repository or plugin root located two levels above `skills/video-trimmer/SKILL.md` (`../../`, for example `/Users/sylph/.gemini/config/plugins/video-trimmer`).
Set the command working directory (`Cwd`) to `<PLUGIN_ROOT>` and invoke `python3 skills/video-trimmer/scripts/video_trimmer.py` directly. Do not search the filesystem with `find_by_name` or `list_dir` to locate the CLI entrypoint.

### Step 1: Environment Verification & GCP Native Setup
Run a single pre-flight check in `<PLUGIN_ROOT>` to verify that FFmpeg and `.env` configuration exist:
```bash
# 1. Pre-flight verification (FFmpeg + .env configuration)
ffmpeg -version >/dev/null && test -f .env && echo "Environment Ready"
```
If `.env` is missing or Google Cloud credentials are not configured, run or instruct the user to run the one-click native `gcloud` setup:
```bash
# 2. Authenticate once with Google Cloud ADC
gcloud auth application-default login

# 3. One-click automated GCP provisioning (Enables Vertex AI + Cloud STT V2 + GCS, creates bucket, SA, and .env):
./setup.sh --project YOUR_PROJECT_ID
```

### Step 2: Execute Primary Video Trimmer Pipeline

Default execution uses **Cloud STT V2 Chirp 3 (`--asr-engine chirp3`)** for word timestamps and fast **Static Multimodal mode (`MEDIA_RESOLUTION_LOW`)** for take arbitration, performs automatic surgical micro-window repair (`15s–90s` windows), and isolates all deliverables inside `<input_dir>/output/` (`./output/` for Google Drive links). Pass `--asr-engine whisper` ONLY when the user explicitly requests local Whisper ASR.

```bash
# Standard automatic rough-cut with Chirp 3 default ASR (outputs isolated in <input_dir>/output/, Cwd = <PLUGIN_ROOT>):
python3 skills/video-trimmer/scripts/video_trimmer.py -i "/path/to/raw_footage.mp4"

# If production shooting script is provided (Mode A - Monotonic Script-Anchored Alignment):
python3 skills/video-trimmer/scripts/video_trimmer.py -i "/path/to/raw_footage.mp4" --script "/path/to/shooting_script.md"

# Optional 1.12x framing punch-in on composite pick-up splices:
python3 skills/video-trimmer/scripts/video_trimmer.py -i "/path/to/raw_footage.mp4" --script "/path/to/shooting_script.md" --punch-in-composite

# Explicit local Whisper ASR (ONLY when user explicitly requests Whisper / local ASR):
python3 skills/video-trimmer/scripts/video_trimmer.py -i "/path/to/raw_footage.mp4" --asr-engine whisper --whisper-model small
```

### Step 3: Deliverable Verification, `agent_verdict` Quality Gate, & Cached Local Iteration
Verify that all required deliverables exist in `<input_dir>/output/` and inspect `agent_verdict` in `<base>_<tag>_edl_report.json`:
- If `agent_verdict.pass_quality_gate == true` (`suggested_action == "DELIVER"`), deliver the outputs to the user.
- If `agent_verdict.pass_quality_gate == false` (`suggested_action == "ONE_SHOT_REMEDIATE"`, or exit code `2` when `--strict` is enabled), execute at most **1** remediation run using `remediation_cmd`.

To adjust pacing, toggle `--punch-in-composite`, or re-render locally without re-incurring cloud Gemini inference:
```bash
python3 skills/video-trimmer/scripts/video_trimmer.py -i "/path/to/raw_footage.mp4" --cached-json "/path/to/output/raw_footage_static_edl.json" --punch-in-composite --suffix "fine_tuned"
```

---

## CLI Options Reference

| Option | Flag | Default | Description |
| :--- | :---: | :--- | :--- |
| `--input` | `-i` | *(Required)* | Path to input raw video file (`.mp4`, `.mov`) or Google Drive link. |
| `--output-dir` | `-o` | `<input_dir>/output/` | Directory to save all generated output files (`./output/` for Google Drive links). |
| `--asr-engine` | | `chirp3` | Word-timestamp ASR engine: `chirp3` (Cloud STT V2, default) or `whisper` (local MLX/faster-whisper). |
| `--stt-location` | | `us` | Cloud Speech-to-Text V2 location for Chirp 3 (defaults to `$STT_LOCATION` or `us`). |
| `--whisper-model` | | `small` | Whisper model size when `--asr-engine whisper` is selected. |
| `--language` | | `auto` | Spoken language code (`auto`, `zh-TW`, `zh-CN`, `en`, `ja`, `ko`). |
| `--force` | | `False` | Bypass cached `<basename>_whisper_raw.json` and force re-transcription. |
| `--model` | `-m` | `gemini-3.8-flash` | Gemini model name (defaults to `$MODEL_NAME` or `gemini-3.8-flash`). |
| `--project` | | `None` | Google Cloud Project ID (defaults to `$GOOGLE_CLOUD_PROJECT` or ADC). |
| `--region` | | `None` | Vertex AI location/region (defaults to `$GOOGLE_CLOUD_LOCATION` or `global`). |
| `--bucket` | | `None` | GCS bucket for video and Chirp 3 audio staging (defaults to `$VIDEO_TRIMMER_BUCKET`). |
| `--keep-gcs-upload` | | `False` | Retain ephemeral staged video in GCS instead of deleting after inference. |
| `--script` | `-s` | `None` | Path to production shooting script (`.md` / `.txt`). |
| `--punch-in-composite` | | `False` | Apply a `1.12x` framing punch-in on composite pick-up splices in MP4 and NLE timelines. |
| `--agentic` | | `False` | Enable Agentic Video Understanding dynamic exploration mode. |
| `--pacing` | `-p` | `auto` | Pacing style: `auto` (adaptive CPS), `compact`, `breathing`. |
| `--cached-json`| | `None` | Path to existing EDL JSON to bypass cloud Gemini inference. |
| `--suffix` | | `None` | Custom tag suffix for generated filenames. |
| `--crf` | | `18` | FFmpeg H.264 rendering CRF parameter (18 = visually lossless). |
| `--skip-whisper`| | `False` | Skip word-level ASR transcription and use energy-only fallback. |
| `--strict` | | `False` | Exit with code `2` after writing audit reports if `agent_verdict.pass_quality_gate` is `False`. |
| `--verbose` | | `False` | Verbose DEBUG-level logging (default is INFO). |

---

## Output Deliverables

For an input file `raw_footage.mp4`, the skill isolates all generated files inside `<input_dir>/output/` (or the directory specified via `-o`):
1. `output/raw_footage_<suffix>_trimmed.mp4`: High-bitrate assembled video cut with 20ms micro-fades, composite loudness matching, and optional `1.12x` composite punch-in.
2. `output/raw_footage_<suffix>_edl.xml`: Final Cut Pro 7 XML timeline for **Premiere Pro** & **DaVinci Resolve** (with composite splice markers).
3. `output/raw_footage_<suffix>_edl.fcpxml`: FCPXML v1.9 timeline for **Final Cut Pro** (with DTD-valid `<adjust-transform>` and `<marker>` elements).
4. `output/raw_footage_<suffix>_edl.json`: Structured decision metadata with `asr_engine`, `agent_verdict`, composite splice roles, CPS, and margins.
5. `output/raw_footage_<suffix>_edl.csv`: Spreadsheet table with visual/audio validation notes.
6. `output/raw_footage_<suffix>_edl_report.md`: 8-dimension human-readable rough-cut quality audit report.
7. `output/raw_footage_<suffix>_edl_report.json`: Machine-readable audit report with top-level `agent_verdict` and `composite_splices_count`.
8. `output/raw_footage_whisper_raw.json`: Engine-tagged word-level ground-truth transcript cache (`chirp3` or `whisper`).

---

## License

[MIT License](LICENSE) © 2026 sylphlin

---

## Google Drive Direct Link & GCS Smart Caching (ADC)

`video-trimmer` natively supports passing Google Drive file links (`https://drive.google.com/file/d/.../view` or `gdrive://...`) directly to `-i / --input`:
- Authenticated 100% via Application Default Credentials (`gcloud auth application-default login`).
- Automatically checks remote MD5 (`md5Checksum`) to cache locally in `<output_dir>/gdrive_inputs/` and checks `sha256` / `gdrive_md5` metadata on `gs://${VIDEO_TRIMMER_BUCKET}/raw/` to skip redundant GCS uploads.
- Example:
  ```bash
  python3 skills/video-trimmer/scripts/video_trimmer.py -i "https://drive.google.com/file/d/YOUR_VIDEO_FILE_ID/view?usp=sharing" -o output/
  ```

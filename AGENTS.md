# Developer & Maintenance Operational Rules (AGENTS.md)

This document serves as the project memory and permanent operational guidelines for the **Video Trimmer** codebase. All agents and developers must strictly adhere to these invariant rules across all future tasks and iterations.

---

## 1. Multi-NLE Timeline Interoperability (Core Output Target)

- **Universal NLE Support**: The primary purpose of this tool is to produce frame-accurate rough-cut timelines directly importable into major non-linear editing software (NLEs):
  - Apple Final Cut Pro (`.fcpxml`)
  - Adobe Premiere Pro (`.xml` via Final Cut Pro 7 XML interchange format)
  - DaVinci Resolve (`.xml` / `.fcpxml`)
  - CMX 3600 Edit Decision Lists (`.edl`)
  - Universal CSV cut list (`.csv`)
- **Timecode Accuracy**: Timecode calculations must support standard broadcast and cinema frame rates (23.976, 24, 25, 29.97, 30, 50, 59.94, 60 fps). Always respect non-drop frame / drop frame timecode specifications.
- **Strict Verification**: Any changes to timeline exporters (`scripts/exporters.py`) must pass schema and unit test verifications against all supported NLE targets.

---

## 2. Prompts and Python Code Strictly in ASD-STE100 English

- **ASD-STE100 Standard**: All prompt templates (`prompts/*.md`), docstrings, code comments, and CLI help messages MUST strictly follow **ASD-STE100 (Simplified Technical English)** principles:
  - Use short, direct sentences (keep instructions below 20 words where possible).
  - Use a restricted, controlled vocabulary with clear and unambiguous meanings.
  - Use the imperative mood for instructions (e.g., "Do not cut", "Verify the output", "Calculate timecode").
  - Maintain active voice; avoid passive voice, convoluted clauses, and vague adverbs.
  - Give one instruction per sentence.
- **English for Code Base**: All Python code (`*.py`), including variable names, class/function definitions, docstrings, comments, log output, error messages, and CLI help descriptions, MUST be written in professional, concise English adhering to ASD-STE100 principles.

---

## 3. Strict Generality & Neutrality (Zero Specific Name / Video Hardcoding)

- **Zero Entity Hardcoding**:
  - Never introduce hardcoded logic, special branches, or regex rules tailored to specific video files, specific YouTube channels, or specific individuals.
  - All trimming heuristics (last-take selection, silence stripping, speech-locking) must generalize across any presenter, show, or production style.
- **Completely Generic Documentation**:
  - Documentation (`README.md`, `SKILL.md`), scripts (`scripts/*.py`), test fixtures, and prompt templates must remain completely generic and production-ready.
  - NEVER include test-specific video titles, specific presenter names, or ad-hoc local testing assets in repository files.
- **Standard Placeholders Only**: Always use generic, standard placeholders in documentation and examples:
  - Video files: `raw_footage.mp4`, `sample_take.mp4`
  - Shooting scripts: `shooting_script.md`
  - Output files: `rough_cut.fcpxml`, `rough_cut.xml`, `rough_cut.edl`
  - Cloud storage buckets: `video-preprocessing-PROJECT_ID`

---

## 4. Typography & Plain Text Formatting (Strict Zero-Emoji Policy)

- **No Emojis in Section Titles or Tables**: Under no circumstances should emojis or decorative icons (e.g., 🎬, ✂️, 📌, 💡, ⏱️, 🚀) be used in section headings (`## 1. `, `## 2. `, etc.), sub-headings, table headers, or structured logs.
- **Executive Plain Text**: Maintain clean, professional, enterprise-grade Markdown typography.

---

## 5. Architectural Invariants

- **Runtime Dependencies**: The core engine is 100% Python 3.10+ and standard FFmpeg/ffprobe. Never introduce Node.js, npm, or heavy GUI framework dependencies into the runtime or workflow.
- **Audio Pop Protection (15ms Micro-Crossfade)**:
  - Every cut boundary rendered via FFmpeg must include a 15ms equal-power micro-fade (`afade=t=in:d=0.015:curve=iqsin` and `afade=t=out:d=0.015:curve=oqsin`) to prevent acoustic popping and DC-offset clicks between jump-cuts.
- **Speech Boundary Tightening (`tighten_clip_to_speech`)**:
  - Never truncate spoken phonemes or words.
  - Dynamically calculate lead-in and lead-out margins based on presenter Characters Per Second (CPS).
  - Preserve natural thinking pauses when spoken by the active presenter.
- **Ephemeral Cloud Storage Staging**:
  - Raw video files uploaded to Google Cloud Storage for Gemini multimodal reasoning must be staged under `gs://${BUCKET}/raw/` and deleted immediately after inference.
  - The storage bucket must enforce a 2-day automatic lifecycle deletion rule as a safety backstop.

---

## 6. Commit & Attribution Policy (Human Authorship Only)

- **No AI/Assistant Branding**: Never include any AI assistant name (e.g., "Claude", "Gemini", "Copilot") in branch names, commit messages, PR titles/descriptions, code comments, or file contents.
- **No Co-Authorship Trailers**: Never append `Co-Authored-By`, session links, or any other AI-attribution trailer to commit messages or PR descriptions.
- **Human Authorship Only**: All commits must be authored as the repository owner (`sylphlin <sylph.lin@gmail.com>`), with no secondary author line.

---

## 7. Model Invariants & Single Source of Truth

- **Designated Model Identifier**:
  - The default multimodal model is `gemini-3.8-flash` via Vertex AI.
  - The agent is strictly prohibited from altering, substituting, downgrading, or inventing any model identifiers outside the designated configuration in `.env` / `.env.example`.
- **Dynamic Configuration via Environment Variables**:
  - Models and project settings must always be loaded dynamically:
    - `MODEL_NAME`: Designated multimodal reasoning model (default: `gemini-3.8-flash`)
    - `GOOGLE_CLOUD_PROJECT`: Target Google Cloud Project ID
    - `VIDEO_TRIMMER_BUCKET`: GCS staging bucket (default: `video-preprocessing-${PROJECT_ID}`)
    - `GOOGLE_CLOUD_LOCATION`: Vertex AI location (default: `global`)
- **Native gcloud Setup (`setup.sh`)**:
  - Cloud infrastructure provisioning must remain 100% native `gcloud` via `setup.sh` (zero Terraform dependency, Cloud Shell ready).

---

## 8. Fail-Fast & Explicit Engine Selection (Strict Zero Silent Fallback)

- **Gemini Multimodal Reasoning as Primary Brain**:
  - Gemini 3.8 Flash native video reasoning is the primary editor brain for take evaluation and cut decisions.
  - Whisper acoustic transcription provides frame-accurate speech timestamps and acoustic ground truth.
- **Fail-Fast on External Infrastructure & Auth Errors**:
  - Whenever encountering external authentication (`401`, `RefreshError`), permission denials (`403 AccessDeniedException`), cloud storage, or quota errors, the agent MUST STOP IMMEDIATELY.
  - Zero tolerance on blind retries, probing alternative buckets, or rewriting core logic.
  - The agent must immediately report the blocked error and present the actionable fix to the human user, awaiting user direction.

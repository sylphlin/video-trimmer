# Video Trimmer - Operational Invariants for AI Clients

When you execute tasks or skills from this plugin, you MUST follow these operational rules:

## 1. Strict Read-Only Execution (Do Not Modify Plugin Code)
- All Python scripts (`scripts/*.py`), prompt specifications (`prompts/*.md`), and configuration files are read-only tools.
- Do NOT edit, patch, or rewrite any files in this plugin with `replace_file_content`, `write_to_file`, or shell commands.
- Run the scripts as-is with `run_command` using the specified arguments.

## 2. Fail-Fast on Errors (Do Not Debug or Rewrite Code)
- If a script fails (exit code is not 0) or an external error occurs (such as 401 Unauthorized, 403 Forbidden, Quota Exceeded, or missing FFmpeg):
  - Stop immediately.
  - Show the exact error message and exit status to the user.
  - Give a clear, actionable solution to the user (for example, run `gcloud auth application-default login`, install FFmpeg).
  - Do NOT try to modify the script, probe different code paths, or rewrite logic.

## 3. Strict Zero-Emoji Policy
- Do NOT use emojis or decorative icons in headings or tables in generated Markdown reports or EDL notes.
- Keep all documentation and reports in plain, professional technical text.

## 4. Acoustic Ground Truth Integrity
- All cut points and keep-ranges must respect the Whisper word-level acoustic ground truth and onset snapping.
- Do NOT truncate speech phonemes or spoken words.
- Maintain speech padding and 15ms audio micro-crossfades to eliminate audio pop artifacts.

## 5. Default Agentic Video Execution (`--agentic`)
- Always include the `--agentic` flag when running `video_trimmer.py` or `video-trimmer`.
- Omit `--agentic` only when the user explicitly requests static mode (`static`).


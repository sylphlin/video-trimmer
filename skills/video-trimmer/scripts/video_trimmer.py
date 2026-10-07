#!/usr/bin/env python3
"""
video-trimmer (AI-Powered Smart Video Trimmer)
---------------------------------------------
An end-to-end video rough-cut and trimming engine powered by:
- Whisper word-level acoustic ground truth (Apple Silicon mlx-whisper & faster-whisper)
- Vertex AI Gemini 3.8 Flash multimodal video understanding and dual-mode take selection
- Acoustic onset snapping (80ms pre-vocal onset) and plosive tail defense
- Adaptive noise floor tracking and 20ms equal-power audio micro-crossfades
- Export to Premiere Pro (.xml), DaVinci Resolve (.xml), Final Cut Pro (.fcpxml), CSV (.csv), and MP4.
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import soundfile as sf
    import numpy as np  # noqa: F401  (used transitively by acoustic.py; kept for early dependency check)
    from google.genai import types
except ImportError as e:
    logging.basicConfig(level=logging.ERROR, format="%(message)s", stream=sys.stderr)
    logging.error("Missing required package: %s", e)
    logging.error("Run: pip install -r requirements.txt")
    sys.exit(1)

if __package__ in (None, ""):
    _skill_root = Path(__file__).resolve().parent.parent
    if str(_skill_root) not in sys.path:
        sys.path.insert(0, str(_skill_root))
    __package__ = "scripts"

from .acoustic import refine_speech_bounds_locked
from .constants import ALLOWED_INPUT_EXTENSIONS
from .edl_auditor import (
    audit_edl_quality,
    deduplicate_and_sort_clips,
    generate_edl_audit_markdown,
    repair_edl_micro_windows,
    sanitize_refined_edl,
)
from .exceptions import InvalidInputError, VideoTrimmerError
from .exporters import generate_edl_csv, generate_fcp7_xml, generate_fcpxml
from .gcs_utils import delete_gcs_blob, is_gdrive_source, download_gdrive_file_with_cache
from .gemini_client import (
    build_micro_window_repair_prompt,
    build_prompt,
    get_gemini_client,
    load_env_file,
    merge_chunked_edl,
    run_gemini_inference,
    stage_video_to_gcs,
)
from .render import probe_video, render_cut_video
from .transcribe import (
    calculate_active_script_window,
    coalesce_adjacent_sub_units,
    detect_chunk_boundaries,
    resolve_clip_sub_units,
    transcribe_video_whisper,
)

logger = logging.getLogger(__name__)


def _parse_gemini_json_text(raw_json: str) -> dict:
    """Clean markdown fences and parse JSON returned by Gemini."""
    cleaned = (raw_json or "").strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()
    elif not cleaned.startswith("{") and "{" in cleaned:
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1:
            cleaned = cleaned[start_idx : end_idx + 1].strip()
    try:
        return json.loads(cleaned)
    except Exception as e:
        raise VideoTrimmerError(f"Failed to parse Gemini JSON response: {e}\nRaw output:\n{cleaned}") from e


def _setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)


def _validate_input_extension(video_path):
    if video_path.suffix.lower() not in ALLOWED_INPUT_EXTENSIONS:
        allowed = ", ".join(ALLOWED_INPUT_EXTENSIONS)
        raise InvalidInputError(f"Unsupported video format: {video_path.suffix} (allowed extensions: {allowed})")


def _append_usage_log(out_dir, video_path, model, usage, mode, duration):
    """Append token usage metrics for each Gemini inference run to usage_log.jsonl."""
    usage_log_path = out_dir / "usage_log.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "video": video_path.name,
        "model": model,
        "mode": mode,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "candidates_tokens": usage.get("candidates_tokens", 0),
        "thoughts_tokens": usage.get("thoughts_tokens", 0),
        "tool_use_tokens": usage.get("tool_use_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "duration_seconds": round(duration, 2),
    }
    with open(usage_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _resolve_output_dir(output_dir_arg: str | None, default_parent: Path) -> Path:
    """Resolve the output directory and isolate deliverables in <default_parent>/output by default."""
    if output_dir_arg:
        out_dir = Path(output_dir_arg).expanduser().resolve()
    else:
        out_dir = default_parent.resolve() / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _run(args):
    if is_gdrive_source(args.input):
        out_dir = _resolve_output_dir(args.output_dir, Path.cwd())
        try:
            video_path = download_gdrive_file_with_cache(
                args.input,
                target_dir=out_dir / "gdrive_inputs",
                project_id=args.project,
            )
        except Exception as e:
            raise InvalidInputError(f"Failed to download video from Google Drive: {e}") from e
    else:
        video_path = Path(args.input).expanduser().resolve()
        if not video_path.exists():
            raise InvalidInputError(f"Video file not found: {video_path}")
        out_dir = _resolve_output_dir(args.output_dir, video_path.parent)

    _validate_input_extension(video_path)
    base_name = video_path.stem

    tag = args.suffix if args.suffix else ("agentic" if args.agentic else "static")

    logger.info("==> 1. Probing input video: %s", video_path.name)
    logger.info("    Output directory: %s", out_dir)
    total_dur, width, height, fps = probe_video(video_path)
    logger.info("    Duration: %.1f s (~%.1f min) | Resolution: %dx%d | Frame rate: %.3f fps",
                total_dur, total_dur / 60, width, height, fps)

    # Layer 1: Whisper word-level acoustic transcription and clause segmentation
    whisper_segs = []
    whisper_sentences = []

    if not args.skip_whisper:
        whisper_json = out_dir / f"{base_name}_whisper_raw.json"
        logger.info("==> 2. Running Whisper word-level transcription and clause segmentation...")
        whisper_segs, whisper_sentences = transcribe_video_whisper(
            video_path,
            whisper_json,
            model_name="small",
        )

    whisper_units = whisper_sentences if whisper_sentences else whisper_segs

    # Layer 2: Gemini multimodal video understanding and dual-mode take arbitration
    inference_metrics = {}
    micro_window_repairs = []
    script_path = Path(args.script).resolve() if args.script else None
    script_text = script_path.read_text(encoding="utf-8") if (script_path is not None and script_path.exists()) else None

    if args.cached_json:
        cached_file = Path(args.cached_json).resolve()
        if not cached_file.exists():
            raise InvalidInputError(f"Cached JSON file not found: {cached_file}")
        logger.info("==> 3. Skipping Gemini inference and loading cached EDL: %s", cached_file.name)
        with open(cached_file, "r", encoding="utf-8") as f:
            model_edl = json.load(f)
    else:
        load_env_file()
        client = get_gemini_client(project_id=args.project, location=args.region)

        _resolved_file = Path(__file__).resolve()
        prompt_candidates = [
            Path(__file__).parent / "prompts" / "video_cut_prompt.md",
            Path(__file__).parent.parent / "prompts" / "video_cut_prompt.md",
            Path.cwd() / "prompts" / "video_cut_prompt.md",
            Path(__file__).parent / "video-trimmer" / "prompts" / "video_cut_prompt.md",
            *[p / "prompts" / "video_cut_prompt.md" for p in _resolved_file.parents[:5]],
        ]

        # Step 3a: Locate the active recording window when a reference script is provided on long footage
        win_start, win_end, active_units = 0.0, total_dur, whisper_units
        if script_text is not None and total_dur > 600.0 and whisper_units:
            win_start, win_end, active_units = calculate_active_script_window(
                whisper_units=whisper_units,
                script_text=script_text,
                total_duration=total_dur,
                padding_seconds=20.0,
            )
            if win_start > 0.0 or win_end < total_dur:
                logger.info(
                    "    [Active Script Window] Focused recording window: %.1fs ~ %.1fs (span %.1fs, %d sentences)",
                    win_start,
                    win_end,
                    win_end - win_start,
                    len(active_units),
                )

        # Step 3b: Split long recordings (>660s) at natural pauses (>=2.0s) without splitting retake clusters
        chunks = detect_chunk_boundaries(
            whisper_units=active_units,
            total_duration=win_end,
            target_chunk_duration=480.0,
            min_pause_duration=2.0,
            max_chunk_duration=660.0,
            window_start=win_start,
        )

        resolved_bucket = (
            args.bucket
            or os.environ.get("VIDEO_TRIMMER_BUCKET")
            or os.environ.get("VIDEO_STORAGE_BUCKET")
            or os.environ.get("MEETING_STORAGE_BUCKET")
            or os.environ.get("GCS_BUCKET")
            or ""
        ).removeprefix("gs://") or None

        logger.info("==> 3. Staging video to Cloud Storage...")
        gcs_uri, mime_type, is_ephemeral = stage_video_to_gcs(video_path, resolved_bucket)

        mode_str = "Agentic Video Understanding" if args.agentic else "Static Multimodal"
        logger.info("==> 4. Running %s [Mode: %s] with Whisper sentences for take arbitration...", args.model, mode_str)

        usage = {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0, "tool_use_tokens": 0, "total_tokens": 0}
        duration = 0.0
        chunk_edl_results = []

        try:
            if len(chunks) > 1:
                logger.info("    [Auto-Chunking] Splitting %.1fs active span into %d natural-pause chunks...", win_end - win_start, len(chunks))

            for idx, (c_start, c_end, c_units) in enumerate(chunks, start=1):
                c_prompt = build_prompt(prompt_candidates, script_path=script_path, whisper_units=c_units)
                use_window_meta = (c_start > 0.0) or (c_end < total_dur - 1.0) or (len(chunks) > 1)
                start_offset = f"{int(c_start)}s" if use_window_meta else None
                end_offset = f"{int(math.ceil(c_end))}s" if use_window_meta else None

                if len(chunks) > 1:
                    logger.info(
                        "    ---> Chunk %d/%d: Window %s ~ %s (%.1fs, %d sentences)",
                        idx,
                        len(chunks),
                        start_offset,
                        end_offset,
                        c_end - c_start,
                        len(c_units),
                    )

                c_raw_json, c_usage, c_dur = run_gemini_inference(
                    client,
                    types,
                    args.model,
                    gcs_uri,
                    mime_type,
                    c_prompt,
                    args.agentic,
                    num_sentences=max(1, len(c_units)),
                    video_duration=max(10.0, c_end - c_start),
                    start_offset=start_offset,
                    end_offset=end_offset,
                )
                duration += c_dur
                for k in usage:
                    usage[k] += c_usage.get(k, 0)

                chunk_edl_results.append(_parse_gemini_json_text(c_raw_json))

            if len(chunk_edl_results) == 1:
                model_edl = chunk_edl_results[0]
            else:
                model_edl = merge_chunked_edl(
                    chunk_edl_results,
                    project_title=f"{base_name} AI Video Trimmer ({tag})",
                )

            # Step 4b: Pre-Render Semantic Audit & Surgical Micro-Window Repair (15s-90s windows)
            if whisper_units:
                def _run_window_repair(win_units, win_start, win_end, start_offset, end_offset, anomaly):
                    nonlocal duration
                    w_prompt = build_micro_window_repair_prompt(
                        prompt_file_candidates=prompt_candidates,
                        whisper_units=win_units,
                        anomaly=anomaly,
                        script_path=script_path,
                    )
                    w_raw_json, w_usage, w_dur = run_gemini_inference(
                        client,
                        types,
                        args.model,
                        gcs_uri,
                        mime_type,
                        w_prompt,
                        False,
                        num_sentences=max(1, len(win_units)),
                        video_duration=max(10.0, win_end - win_start),
                        start_offset=start_offset,
                        end_offset=end_offset,
                    )
                    duration += w_dur
                    for k in usage:
                        usage[k] += w_usage.get(k, 0)
                    return _parse_gemini_json_text(w_raw_json)

                model_edl, micro_window_repairs = repair_edl_micro_windows(
                    model_edl=model_edl,
                    whisper_units=whisper_units,
                    total_dur=total_dur,
                    run_window_inference_fn=_run_window_repair,
                    script_text=script_text,
                )
        finally:
            if is_ephemeral and not args.keep_gcs_upload:
                delete_gcs_blob(gcs_uri)

        logger.info("    Model inference complete (%.1f s).", duration)
        logger.info("    Token usage: prompt=%s | candidates=%s | thoughts=%s | tool_use=%s | total=%s",
                    f"{usage['prompt_tokens']:,}", f"{usage['candidates_tokens']:,}",
                    f"{usage['thoughts_tokens']:,}", f"{usage['tool_use_tokens']:,}", f"{usage['total_tokens']:,}")

        mode = "agentic" if args.agentic else "static"
        inference_metrics = {
            "mode": mode,
            "duration_seconds": round(duration, 2),
            "prompt_tokens": usage["prompt_tokens"],
            "candidates_tokens": usage["candidates_tokens"],
            "thoughts_tokens": usage["thoughts_tokens"],
            "tool_use_tokens": usage["tool_use_tokens"],
            "total_tokens": usage["total_tokens"],
            "micro_window_repairs": len(micro_window_repairs),
        }
        _append_usage_log(out_dir, video_path, args.model, usage, mode, duration)

    # Layer 3 & Layer 4: Sub-unit word trimming, cross-clip coalescing, and acoustic onset snapping
    logger.info("==> 5. Running text-locked word boundary trimming and acoustic onset snapping...")
    temp_wav = out_dir / f"temp_{base_name}.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(temp_wav)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audio, sr = sf.read(str(temp_wav))

    if whisper_units and isinstance(model_edl, dict) and model_edl.get("final_edl"):
        model_edl["final_edl"] = deduplicate_and_sort_clips(
            clips=model_edl["final_edl"],
            whisper_units=whisper_units,
            script_text=script_text,
        )

    expanded_units = []
    for c in model_edl.get("final_edl", []):
        sub_units = resolve_clip_sub_units(whisper_units, c, total_dur)
        for idx, su in enumerate(sub_units):
            topic_label = c["topic"] if len(sub_units) == 1 else f"{c['topic']} (#{idx + 1})"
            expanded_units.append({
                "parent_clip": c,
                "topic": topic_label,
                "sentence_ids": su.get("sentence_ids", []),
                "t_first": su["t_first"],
                "t_last": su["t_last"],
                "prev_sentence_end": su["prev_sentence_end"],
                "next_sentence_start": su["next_sentence_start"],
                "transcript": su["transcript"],
                "whisper_transcript": su.get("whisper_transcript", su["transcript"]),
                "trimmed_head": su.get("trimmed_head", False),
                "trimmed_tail": su.get("trimmed_tail", False),
            })

    # Coalesce adjacent continuous sub-units across clip boundaries when gap < 0.40s and no ID is skipped
    filtered_units = coalesce_adjacent_sub_units(expanded_units)

    refined_edl = []
    for seq_id, su in enumerate(filtered_units, start=1):
        c = su["parent_clip"]
        t_first = su["t_first"]
        t_last = su["t_last"]
        transcript = su["transcript"]
        prev_sentence_end = su["prev_sentence_end"]
        next_sentence_start = su["next_sentence_start"]

        tight_in, tight_out, clip_cps, in_m, out_m = refine_speech_bounds_locked(
            audio, sr, t_first, t_last, total_dur, transcript=transcript, pacing=args.pacing,
            prev_sentence_end=prev_sentence_end, next_sentence_start=next_sentence_start
        )
        if tight_out <= tight_in:
            tight_out = tight_in + 1.0
        dur = round(tight_out - tight_in, 2)

        refined_edl.append({
            "clip_id": seq_id,
            "topic": su["topic"],
            "sentence_ids": su.get("sentence_ids", []),
            "t_first": round(t_first, 2),
            "t_last": round(t_last, 2),
            "source_in": tight_in,
            "source_out": tight_out,
            "duration": dur,
            "cps": clip_cps,
            "in_margin": in_m,
            "out_margin": out_m,
            "transcript": transcript,
            "whisper_transcript": su.get("whisper_transcript", transcript),
            "take_selection_reason": c.get("take_selection_reason", ""),
            "visual_check": c.get("visual_check", "Direct eye contact with camera, eyes open."),
            "audio_check": c.get("audio_check", f"Whisper locked (in_margin={in_m:.2f}s, out_margin={out_m:.2f}s)")
        })
        logger.info("    Clip %2d: In=%6.2fs, Out=%6.2fs (%5.2fs) | t_first: %.2fs, t_last: %.2fs | %s",
                    seq_id, tight_in, tight_out, dur, t_first, t_last, su["topic"])

    temp_wav.unlink(missing_ok=True)

    # Layer 5: Deterministic timeline sanitization and 8-dimension rough-cut quality audit
    refined_edl, sanitization_stats = sanitize_refined_edl(refined_edl, total_dur)
    total_out_dur = round(sum(c["duration"] for c in refined_edl), 2)
    avg_cps = round(sum(c["cps"] for c in refined_edl) / max(1, len(refined_edl)), 2)
    logger.info("    Retained clips: %d | Trimmed duration: %.1f s (~%.2f min) | Average CPS: %.2f chars/s",
                len(refined_edl), total_out_dur, total_out_dur / 60, avg_cps)

    audit_report = audit_edl_quality(
        refined_edl=refined_edl,
        whisper_units=whisper_units,
        total_dur=total_dur,
        video_path=video_path,
        script_text=script_text,
        script_path=script_path,
        sanitization_stats=sanitization_stats,
        micro_window_repairs=micro_window_repairs,
    )
    report_json_path = out_dir / f"{base_name}_{tag}_edl_report.json"
    report_md_path = out_dir / f"{base_name}_{tag}_edl_report.md"
    report_json_path.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_edl_audit_markdown(audit_report, report_md_path)
    verdict = audit_report["agent_verdict"]
    logger.info(
        "    [Quality Audit] Grade: %s | Pass Quality Gate: %s | Report: %s",
        verdict["overall_grade"],
        verdict["pass_quality_gate"],
        report_md_path.name,
    )

    json_path = out_dir / f"{base_name}_{tag}_edl.json"
    json_path.write_text(json.dumps({
        "project_title": f"{base_name} AI Video Trimmer ({tag})",
        "pacing_style": "text_based_whisper_grounded",
        "agent_verdict": verdict,
        "inference_metrics": inference_metrics,
        "average_cps": avg_cps,
        "total_duration": total_out_dur,
        "final_edl": refined_edl
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("==> 6. Wrote structured EDL JSON and audit report: %s, %s", json_path.name, report_json_path.name)

    csv_path = out_dir / f"{base_name}_{tag}_edl.csv"
    generate_edl_csv(refined_edl, csv_path)
    logger.info("==> 7. Wrote CSV cut table: %s", csv_path.name)

    xml_path = out_dir / f"{base_name}_{tag}_edl.xml"
    generate_fcp7_xml(refined_edl, video_path, total_dur, xml_path, width, height, fps)
    logger.info("==> 8. Wrote FCP 7 XML timeline: %s", xml_path.name)

    fcpxml_path = out_dir / f"{base_name}_{tag}_edl.fcpxml"
    generate_fcpxml(refined_edl, video_path, total_dur, total_out_dur, fcpxml_path, fps, width, height)
    logger.info("==> 9. Wrote Final Cut Pro FCPXML timeline: %s", fcpxml_path.name)

    out_mp4 = out_dir / f"{base_name}_{tag}_trimmed.mp4"
    logger.info("==> 10. Rendering rough-cut MP4 with FFmpeg: %s ...", out_mp4.name)
    render_cut_video(refined_edl, video_path, out_mp4, crf=args.crf)
    logger.info("==> [Done] Final MP4 rendered (%.1f MB): %s", out_mp4.stat().st_size / (1024 * 1024), out_mp4.name)

    if args.strict and not verdict["pass_quality_gate"]:
        logger.error("==> [Strict Quality Gate Failed] %s", "; ".join(verdict["fatal_violations"]))
        sys.exit(2)


def main():
    load_env_file()
    default_model = os.environ.get("MODEL_NAME") or os.environ.get("TRIMMER_MODEL") or "gemini-3.8-flash"

    parser = argparse.ArgumentParser(description="video-trimmer: AI-Powered Smart Video Trimmer (Whisper Ground-Truth + Vertex AI Gemini)")
    parser.add_argument("--input", "-i", required=True, help="Path to input video file (.mp4/.mov) or Google Drive share link (https://drive.google.com/... / gdrive://...)")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory (defaults to <input_dir>/output/, or ./output/ for Google Drive links)")
    parser.add_argument("--model", "-m", default=default_model, help=f"Gemini model name (default: {default_model})")
    parser.add_argument("--project", default=None, help="Google Cloud Project ID (defaults to GOOGLE_CLOUD_PROJECT/GCP_PROJECT or ADC)")
    parser.add_argument("--region", default=None, help="Vertex AI region/location (defaults to GOOGLE_CLOUD_LOCATION/GCP_REGION or 'global')")
    parser.add_argument("--bucket", default=None, help="GCS bucket for video staging (defaults to VIDEO_TRIMMER_BUCKET)")
    parser.add_argument("--keep-gcs-upload", action="store_true", help="Retain staged video in GCS instead of deleting after inference")
    parser.add_argument("--crf", type=int, default=18, help="FFmpeg H.264 CRF quality parameter (default: 18)")
    parser.add_argument("--pacing", "-p", choices=["auto", "dynamic", "compact", "breathing"], default="auto",
                        help="Pacing style: 'auto'/'dynamic' (adaptive CPS margin, default), 'compact', or 'breathing'")
    parser.add_argument("--agentic", action="store_true", help="Enable Gemini Agentic Video Understanding mode")
    parser.add_argument("--suffix", default=None, help="Custom filename tag suffix (defaults to 'agentic' or 'static')")
    parser.add_argument("--script", "-s", default=None, help="Optional reference shooting script or outline (.txt/.md)")
    parser.add_argument("--cached-json", default=None,
                        help="Path to an existing EDL JSON file to bypass cloud Gemini inference")
    parser.add_argument("--skip-whisper", action="store_true", help="Skip local Whisper transcription and use energy-only fallback")
    parser.add_argument("--strict", action="store_true", help="Exit with code 2 after writing audit reports if agent_verdict.pass_quality_gate is False")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose DEBUG-level logging")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    try:
        _run(args)
    except VideoTrimmerError as e:
        logger.error("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

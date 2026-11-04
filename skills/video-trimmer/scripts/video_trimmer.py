#!/usr/bin/env python3
"""
video-trimmer (AI-Powered Smart Video Trimmer)
---------------------------------------------
An end-to-end video rough-cut and trimming engine powered by:
- Dual-engine word-level acoustic ground truth (Cloud STT V2 Chirp 3 default; Whisper optional)
- Vertex AI Gemini 3.8 Flash multimodal video understanding, dual-mode take selection, and composite pick-up backfill
- Acoustic onset snapping, post-pause onset clamping, and plosive tail defense
- Adaptive noise floor tracking, composite loudness gain matching, and 20ms equal-power audio micro-crossfades
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

from .acoustic import (
    clamp_post_pause_word_onset,
    compute_composite_gain_db,
    find_optimal_overlap_pivot,
    refine_speech_bounds_locked,
)
from .chirp3_asr import join_words_for_display
from .constants import (
    ALLOWED_INPUT_EXTENSIONS,
    DEFAULT_ASR_ENGINE,
    DEFAULT_CHIRP3_MODEL,
    DEFAULT_STT_LOCATION,
    DEFAULT_WHISPER_MODEL,
)
from .edl_auditor import (
    audit_edl_quality,
    generate_edl_audit_markdown,
    sanitize_refined_edl,
)
from .edl_occlusion import (
    deduplicate_and_sort_clips,
    occlude_expanded_sub_units,
)
from .edl_repair import (
    detect_micro_window_anomalies,
    repair_edl_micro_windows,
)
from .exceptions import FFmpegError, InvalidInputError, VideoTrimmerError
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
    is_viable_sub_span,
    normalize_text,
    resolve_clip_sub_units,
    transcribe_video_words,
)

logger = logging.getLogger(__name__)


def _parse_gemini_json_text(raw_json: str) -> dict:
    """Clean markdown fences and parse JSON returned by Gemini."""
    cleaned = (raw_json or "").strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()
    elif not cleaned.startswith("{") and not cleaned.startswith("["):
        brace_idx = cleaned.find("{")
        bracket_idx = cleaned.find("[")
        candidates = [i for i in (brace_idx, bracket_idx) if i != -1]
        if candidates:
            start_idx = min(candidates)
            end_brace = cleaned.rfind("}")
            end_bracket = cleaned.rfind("]")
            end_idx = max(end_brace, end_bracket)
            if end_idx > start_idx:
                cleaned = cleaned[start_idx : end_idx + 1].strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return {"final_edl": parsed}
        return parsed
    except json.JSONDecodeError as e:
        if "Extra data" in str(e) and cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                wrapped = json.loads(f"[{cleaned}]")
                if isinstance(wrapped, list) and all(isinstance(x, dict) for x in wrapped):
                    return {"final_edl": wrapped}
            except Exception:
                pass
        raise VideoTrimmerError(f"Failed to parse Gemini JSON response: {e}\nRaw output:\n{cleaned}") from e
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


def _count_shared_word_overlap(prefix_words: list[dict], cont_words: list[dict], max_words: int = 8) -> int:
    """Count exact normalized token overlap between the suffix of prefix_words and prefix of cont_words."""
    max_k = min(max_words, len(prefix_words) - 1, len(cont_words) - 1)
    if max_k <= 0:
        return 0
    pref_norms = [normalize_text(w.get("word", "")) for w in prefix_words]
    cont_norms = [normalize_text(w.get("word", "")) for w in cont_words]
    for k in range(max_k, 0, -1):
        if all(pref_norms[-k:]) and pref_norms[-k:] == cont_norms[:k]:
            if len("".join(pref_norms[-k:])) >= 2:
                return k
    return 0


def _trim_transcript_tail_to_words(raw_transcript: str, kept_words: list[dict]) -> str:
    """Trim the tail of raw_transcript so its normalized length matches kept_words."""
    target_norm = "".join(normalize_text(w.get("word", "")) for w in kept_words)
    if not target_norm:
        return ""
    if not raw_transcript:
        return join_words_for_display([w.get("word", "") for w in kept_words if w.get("word")])
    seen = 0
    cut_pos = len(raw_transcript)
    for idx, ch in enumerate(raw_transcript):
        if normalize_text(ch):
            seen += 1
            if seen >= len(target_norm):
                cut_pos = idx + 1
                break
    trimmed = raw_transcript[:cut_pos].rstrip(" ，,、；;：:")
    return trimmed or join_words_for_display([w.get("word", "") for w in kept_words if w.get("word")])


def _trim_transcript_head_to_words(raw_transcript: str, kept_words: list[dict]) -> str:
    """Trim the head of raw_transcript so its trailing normalized characters match kept_words."""
    target_norm = "".join(normalize_text(w.get("word", "")) for w in kept_words)
    if not target_norm:
        return ""
    raw_norm = normalize_text(raw_transcript)
    if not raw_transcript or len(raw_norm) <= len(target_norm):
        return raw_transcript or join_words_for_display([w.get("word", "") for w in kept_words if w.get("word")])
    skip_norm = len(raw_norm) - len(target_norm)
    seen = 0
    start_pos = 0
    for idx, ch in enumerate(raw_transcript):
        if normalize_text(ch):
            if seen == skip_norm:
                start_pos = idx
                break
            seen += 1
    trimmed = raw_transcript[start_pos:].lstrip(" ，,、。！？!?；;：:")
    return trimmed or join_words_for_display([w.get("word", "") for w in kept_words if w.get("word")])


def _synchronize_composite_overlap_pivots(filtered_units: list[dict], audio, sr: int) -> None:
    """Synchronize word-level overlap pivots across adjacent composite prefix -> continuation pairs."""
    for idx in range(len(filtered_units) - 1):
        prev_su = filtered_units[idx]
        curr_su = filtered_units[idx + 1]
        if not (prev_su.get("composite_role") == "prefix" and curr_su.get("is_composite_splice")):
            continue
        pref_words = list(prev_su.get("words") or [])
        cont_words = list(curr_su.get("words") or [])
        overlap_k = _count_shared_word_overlap(pref_words, cont_words)
        if overlap_k <= 0:
            continue

        pivot_p = find_optimal_overlap_pivot(audio, sr, pref_words, cont_words, overlap_k)
        pref_keep = len(pref_words) - overlap_k + pivot_p
        cont_start = pivot_p
        if pref_keep <= 0 or cont_start >= len(cont_words):
            continue

        new_pref_words = pref_words[:pref_keep]
        new_cont_words = cont_words[cont_start:]
        new_pref_text = join_words_for_display([w.get("word", "") for w in new_pref_words if w.get("word")])
        new_pref_dur = float(new_pref_words[-1]["end"]) - float(new_pref_words[0]["start"])
        if not is_viable_sub_span(new_pref_text, new_pref_dur):
            continue

        if pref_keep < len(pref_words):
            cut_word_start = round(float(pref_words[pref_keep]["start"]), 3)
            prev_su["words"] = new_pref_words
            prev_su["t_last"] = round(float(new_pref_words[-1]["end"]), 3)
            prev_next = prev_su.get("next_sentence_start")
            prev_su["next_sentence_start"] = min(float(prev_next), cut_word_start) if prev_next is not None else cut_word_start
            prev_su["next_word_hard_wall"] = prev_su["next_sentence_start"]
            prev_su["trimmed_tail"] = True
            prev_su["asr_transcript"] = new_pref_text
            prev_su["transcript"] = _trim_transcript_tail_to_words(prev_su.get("transcript", ""), new_pref_words)

        if cont_start > 0:
            cut_prev_end = round(float(cont_words[cont_start - 1]["end"]), 3)
            new_cont_text = join_words_for_display([w.get("word", "") for w in new_cont_words if w.get("word")])
            curr_su["words"] = new_cont_words
            curr_su["t_first"] = round(float(new_cont_words[0]["start"]), 3)
            curr_prev = curr_su.get("prev_sentence_end")
            curr_su["prev_sentence_end"] = max(float(curr_prev), cut_prev_end) if curr_prev is not None else cut_prev_end
            curr_su["trimmed_head"] = True
            curr_su["asr_transcript"] = new_cont_text
            curr_su["transcript"] = _trim_transcript_head_to_words(curr_su.get("transcript", ""), new_cont_words)


def _expand_and_occlude_edl_clips(
    model_edl: dict,
    whisper_units: list[dict],
    total_dur: float,
    script_text: str | None = None,
) -> list[dict]:
    """
    Execute deterministic Layer 3 & Layer 4 sub-unit word alignment, occlusion, and coalescing.

    ASD-STE100:
    1. Deduplicate and sort LLM clips with composite seam occlusion.
    2. Expand clips to physical word sub-units, occlude residual stumbles, and coalesce continuous spans.
    """
    clips = [dict(c) for c in model_edl.get("final_edl", [])] if isinstance(model_edl, dict) else []
    if whisper_units and clips:
        clips = deduplicate_and_sort_clips(
            clips=clips,
            whisper_units=whisper_units,
            script_text=script_text,
        )

    expanded_units = []
    for c in clips:
        sub_units = resolve_clip_sub_units(whisper_units, c, total_dur)
        base_topic = str(c.get("topic") or f"Clip {c.get('clip_id', 1)}")
        for idx, su in enumerate(sub_units):
            topic_label = base_topic if len(sub_units) == 1 else f"{base_topic} (#{idx + 1})"
            asr_tr = su.get("asr_transcript") or su["transcript"]
            expanded_units.append({
                "parent_clip": c,
                "topic": topic_label,
                "sentence_ids": su.get("sentence_ids", []),
                "t_first": su["t_first"],
                "t_last": su["t_last"],
                "prev_sentence_end": su["prev_sentence_end"],
                "next_sentence_start": su["next_sentence_start"],
                "next_word_hard_wall": su.get("next_word_hard_wall"),
                "words": list(su.get("words") or []),
                "transcript": su["transcript"],
                "asr_transcript": asr_tr,
                "trimmed_head": su.get("trimmed_head", False),
                "trimmed_tail": su.get("trimmed_tail", False),
                "composite_role": su.get("composite_role", "none"),
                "is_composite_splice": bool(su.get("is_composite_splice", False)),
                "llm_verified_window": bool(su.get("llm_verified_window", False)),
            })

    expanded_units = occlude_expanded_sub_units(expanded_units, script_text=script_text)
    filtered_units = coalesce_adjacent_sub_units(expanded_units)
    filtered_units = occlude_expanded_sub_units(filtered_units, script_text=script_text)
    return filtered_units


def _sub_units_to_aligned_edl(model_edl: dict, filtered_units: list[dict]) -> dict:
    """Convert aligned Layer 3/4 sub-units into an EDL dictionary for closed-loop verification."""
    aligned_clips = []
    for seq_id, su in enumerate(filtered_units, start=1):
        parent = su.get("parent_clip") or {}
        sids = list(su.get("sentence_ids") or [])
        aligned_clips.append({
            "clip_id": seq_id,
            "topic": su.get("topic", parent.get("topic", f"Clip {seq_id}")),
            "sentence_ids": sids,
            "start_sentence_id": sids[0] if sids else parent.get("start_sentence_id"),
            "end_sentence_id": sids[-1] if sids else parent.get("end_sentence_id"),
            "t_first": su["t_first"],
            "t_last": su["t_last"],
            "source_in": su["t_first"],
            "source_out": su["t_last"],
            "transcript": su["transcript"],
            "asr_transcript": su.get("asr_transcript") or su["transcript"],
            "trimmed_head": bool(su.get("trimmed_head", False)),
            "trimmed_tail": bool(su.get("trimmed_tail", False)),
            "composite_role": su.get("composite_role", "none"),
            "is_composite_splice": bool(su.get("is_composite_splice", False)),
            "llm_verified_window": bool(su.get("llm_verified_window", False)),
            "take_selection_reason": parent.get("take_selection_reason", ""),
            "visual_check": parent.get("visual_check", "Direct eye contact with camera, eyes open."),
            "audio_check": parent.get("audio_check", ""),
        })
    out_edl = dict(model_edl) if isinstance(model_edl, dict) else {}
    out_edl["final_edl"] = aligned_clips
    return out_edl


def _run(args):
    load_env_file()

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

    script_path = Path(args.script).expanduser().resolve() if args.script else None
    if script_path is not None and not script_path.exists():
        raise InvalidInputError(f"Reference script file not found: {script_path}")
    script_text = script_path.read_text(encoding="utf-8") if script_path is not None else None

    resolved_bucket = (
        args.bucket
        or os.environ.get("VIDEO_TRIMMER_BUCKET")
        or os.environ.get("VIDEO_STORAGE_BUCKET")
        or os.environ.get("MEETING_STORAGE_BUCKET")
        or os.environ.get("GCS_BUCKET")
        or ""
    ).removeprefix("gs://") or None
    resolved_stt_location = args.stt_location or os.environ.get("STT_LOCATION") or DEFAULT_STT_LOCATION
    resolved_transcribe_model = os.environ.get("TRANSCRIBE_MODEL") or DEFAULT_CHIRP3_MODEL

    logger.info("==> 1. Probing input video: %s", video_path.name)
    logger.info("    Output directory: %s", out_dir)
    total_dur, width, height, fps = probe_video(video_path)
    logger.info("    Duration: %.1f s (~%.1f min) | Resolution: %dx%d | Frame rate: %.3f fps",
                total_dur, total_dur / 60, width, height, fps)

    # Layer 1: Dual-engine word-level acoustic transcription (Chirp 3 default; Whisper optional)
    whisper_segs = []
    whisper_sentences = []

    if not args.skip_asr:
        words_json = out_dir / f"{base_name}_words.json"
        asr_label = (
            f"Chirp 3 (Cloud STT V2, {resolved_stt_location})"
            if args.asr_engine == "chirp3"
            else f"Whisper ({args.whisper_model})"
        )
        logger.info("==> 2. Running %s word-level transcription and clause segmentation...", asr_label)
        whisper_segs, whisper_sentences = transcribe_video_words(
            video_path=video_path,
            out_json_path=words_json,
            asr_engine=args.asr_engine,
            whisper_model=args.whisper_model,
            language=args.language,
            text_hint=script_text,
            project=args.project,
            gcs_bucket=resolved_bucket,
            stt_location=resolved_stt_location,
            transcribe_model=resolved_transcribe_model,
            force=args.force,
        )

    whisper_units = whisper_sentences if whisper_sentences else whisper_segs

    # Layer 2: Gemini multimodal video understanding and dual-mode take arbitration
    inference_metrics = {}
    micro_window_repairs = []
    prealigned_units: list[dict] | None = None

    if args.cached_json:
        cached_file = Path(args.cached_json).resolve()
        if not cached_file.exists():
            raise InvalidInputError(f"Cached JSON file not found: {cached_file}")
        logger.info("==> 3. Skipping Gemini inference and loading cached EDL: %s", cached_file.name)
        with open(cached_file, "r", encoding="utf-8") as f:
            model_edl = json.load(f)
    else:
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

        logger.info("==> 3. Staging video to Cloud Storage...")
        gcs_uri, mime_type, is_ephemeral = stage_video_to_gcs(video_path, resolved_bucket)

        mode_str = "Agentic Video Understanding" if args.agentic else "Static Multimodal"
        logger.info("==> 4. Running %s [Mode: %s] with ground-truth sentences for take arbitration...", args.model, mode_str)

        usage = {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0, "tool_use_tokens": 0, "total_tokens": 0}
        duration = 0.0
        chunk_edl_results = []

        try:
            if len(chunks) > 1:
                logger.info("    [Auto-Chunking] Splitting %.1fs active span into %d natural-pause chunks...", win_end - win_start, len(chunks))

            for idx, (c_start, c_end, c_units) in enumerate(chunks, start=1):
                c_prompt = build_prompt(
                    prompt_candidates,
                    script_path=script_path,
                    whisper_units=c_units,
                    script_text=script_text,
                )
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
                        script_text=script_text,
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

                # Step 4c: Closed-Loop Post-Alignment Verification (inspect physical asr_transcript words before GCS cleanup)
                preview_units = _expand_and_occlude_edl_clips(
                    model_edl=model_edl,
                    whisper_units=whisper_units,
                    total_dur=total_dur,
                    script_text=script_text,
                )
                aligned_edl = _sub_units_to_aligned_edl(model_edl, preview_units)
                post_align_anomalies = detect_micro_window_anomalies(
                    model_edl=aligned_edl,
                    whisper_units=whisper_units,
                    total_dur=total_dur,
                    script_text=script_text,
                )
                if post_align_anomalies:
                    logger.info(
                        "    [Closed-Loop Verification] Detected %d post-alignment anomaly window(s); running targeted repair before GCS cleanup...",
                        len(post_align_anomalies),
                    )
                    model_edl, post_repairs = repair_edl_micro_windows(
                        model_edl=aligned_edl,
                        whisper_units=whisper_units,
                        total_dur=total_dur,
                        run_window_inference_fn=_run_window_repair,
                        script_text=script_text,
                    )
                    micro_window_repairs.extend(post_repairs)
                else:
                    prealigned_units = preview_units
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
            "asr_engine": args.asr_engine,
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
    wav_proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(temp_wav)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if wav_proc.returncode != 0 or not temp_wav.exists():
        temp_wav.unlink(missing_ok=True)
        raise FFmpegError(f"Failed to extract 16kHz mono WAV from {video_path.name}:\n{wav_proc.stderr.strip()}")
    audio, sr = sf.read(str(temp_wav))

    try:
        filtered_units = (
            prealigned_units
            if prealigned_units is not None
            else _expand_and_occlude_edl_clips(
                model_edl=model_edl,
                whisper_units=whisper_units,
                total_dur=total_dur,
                script_text=script_text,
            )
        )

        # Synchronize shared overlap pivots across composite prefix -> continuation pairs
        _synchronize_composite_overlap_pivots(filtered_units, audio, sr)

        refined_edl = []
        for seq_id, su in enumerate(filtered_units, start=1):
            c = su["parent_clip"]
            su_words = list(su.get("words") or [])
            prev_sentence_end = su["prev_sentence_end"]
            next_sentence_start = su["next_sentence_start"]

            if su_words:
                w0 = su_words[0]
                gap_before = (
                    float(w0["start"]) - float(prev_sentence_end)
                    if prev_sentence_end is not None
                    else 1.0
                )
                clamped_w0 = clamp_post_pause_word_onset(
                    audio,
                    sr,
                    float(w0["start"]),
                    float(w0["end"]),
                    prev_word_end=prev_sentence_end,
                    gap_before=gap_before,
                    is_trimmed_head=bool(su.get("trimmed_head") or su.get("is_composite_splice")),
                )
                if clamped_w0 > float(w0["start"]):
                    su_words[0] = {**w0, "start": clamped_w0}
                    su["t_first"] = clamped_w0

            t_first = float(su["t_first"])
            t_last = float(su["t_last"])
            transcript = su["transcript"]
            tight_out_seam = bool(su.get("composite_role") == "prefix" or su.get("trimmed_tail"))
            tight_in_seam = bool(su.get("is_composite_splice") or su.get("trimmed_head"))

            tight_in, tight_out, clip_cps, in_m, out_m = refine_speech_bounds_locked(
                audio,
                sr,
                t_first,
                t_last,
                total_dur,
                transcript=transcript,
                pacing=args.pacing,
                prev_sentence_end=prev_sentence_end,
                next_sentence_start=next_sentence_start,
                tight_out_seam=tight_out_seam,
                tight_in_seam=tight_in_seam,
            )
            hard_wall = su.get("next_word_hard_wall")
            if hard_wall is not None and tight_out > float(hard_wall):
                tight_out = round(max(t_last, float(hard_wall)), 3)
            if tight_out <= tight_in:
                tight_out = round(tight_in + 0.50, 3)
            dur = round(tight_out - tight_in, 3)

            refined_edl.append({
                "clip_id": seq_id,
                "topic": su["topic"],
                "sentence_ids": su.get("sentence_ids", []),
                "t_first": round(t_first, 3),
                "t_last": round(t_last, 3),
                "source_in": tight_in,
                "source_out": tight_out,
                "duration": dur,
                "cps": clip_cps,
                "in_margin": in_m,
                "out_margin": out_m,
                "transcript": transcript,
                "asr_transcript": su.get("asr_transcript") or transcript,
                "composite_role": su.get("composite_role", "none"),
                "is_composite_splice": bool(su.get("is_composite_splice", False)),
                "llm_verified_window": bool(su.get("llm_verified_window", False)),
                "trimmed_head": bool(su.get("trimmed_head", False)),
                "trimmed_tail": bool(su.get("trimmed_tail", False)),
                "next_word_hard_wall": hard_wall,
                "gain_db": 0.0,
                "take_selection_reason": c.get("take_selection_reason", ""),
                "visual_check": c.get("visual_check", "Direct eye contact with camera, eyes open."),
                "audio_check": c.get("audio_check", f"{args.asr_engine} locked (in_margin={in_m:.3f}s, out_margin={out_m:.3f}s)")
            })
            logger.info("    Clip %2d: In=%6.3fs, Out=%6.3fs (%5.3fs) | t_first: %.3fs, t_last: %.3fs | %s",
                        seq_id, tight_in, tight_out, dur, t_first, t_last, su["topic"])

        # Layer 5: Deterministic timeline sanitization and composite loudness gain matching
        refined_edl, sanitization_stats = sanitize_refined_edl(refined_edl, total_dur)
        for idx in range(1, len(refined_edl)):
            curr_c = refined_edl[idx]
            if curr_c.get("is_composite_splice"):
                prev_c = refined_edl[idx - 1]
                curr_c["gain_db"] = compute_composite_gain_db(
                    audio,
                    sr,
                    (prev_c["source_in"], prev_c["source_out"]),
                    (curr_c["source_in"], curr_c["source_out"]),
                )
    finally:
        temp_wav.unlink(missing_ok=True)

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
        asr_engine=args.asr_engine,
        language=args.language,
        whisper_model=args.whisper_model,
        punch_in_composite=args.punch_in_composite,
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
        "pacing_style": "text_based_acoustic_grounded",
        "asr_engine": args.asr_engine,
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
    generate_fcp7_xml(
        refined_edl,
        video_path,
        total_dur,
        xml_path,
        width=width,
        height=height,
        fps=fps,
        punch_in_composite=args.punch_in_composite,
    )
    logger.info("==> 8. Wrote FCP 7 XML timeline: %s", xml_path.name)

    fcpxml_path = out_dir / f"{base_name}_{tag}_edl.fcpxml"
    generate_fcpxml(
        refined_edl,
        video_path,
        total_dur,
        total_out_dur,
        fcpxml_path,
        fps=fps,
        width=width,
        height=height,
        punch_in_composite=args.punch_in_composite,
    )
    logger.info("==> 9. Wrote Final Cut Pro FCPXML timeline: %s", fcpxml_path.name)

    out_mp4 = out_dir / f"{base_name}_{tag}_trimmed.mp4"
    logger.info("==> 10. Rendering rough-cut MP4 with FFmpeg: %s ...", out_mp4.name)
    render_cut_video(
        refined_edl,
        video_path,
        out_mp4,
        crf=args.crf,
        punch_in_composite=args.punch_in_composite,
        width=width,
        height=height,
    )
    logger.info("==> [Done] Final MP4 rendered (%.1f MB): %s", out_mp4.stat().st_size / (1024 * 1024), out_mp4.name)

    if args.strict and not verdict["pass_quality_gate"]:
        logger.error("==> [Strict Quality Gate Failed] %s", "; ".join(verdict["fatal_violations"]))
        sys.exit(2)


def main():
    load_env_file()
    default_model = os.environ.get("MODEL_NAME") or os.environ.get("TRIMMER_MODEL") or "gemini-3.8-flash"
    env_asr = (os.environ.get("ASR_ENGINE") or DEFAULT_ASR_ENGINE).strip().lower()
    default_asr = env_asr if env_asr in ("chirp3", "whisper") else DEFAULT_ASR_ENGINE

    parser = argparse.ArgumentParser(
        description="video-trimmer: AI-Powered Smart Video Trimmer (Chirp 3 / Whisper Ground-Truth + Vertex AI Gemini)"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input video file (.mp4/.mov) or Google Drive share link (https://drive.google.com/... / gdrive://...)")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory (defaults to <input_dir>/output/, or ./output/ for Google Drive links)")
    parser.add_argument("--asr-engine", choices=["chirp3", "whisper"], default=default_asr,
                        help=f"Word-timestamp ASR engine: 'chirp3' (Cloud STT V2, default) or 'whisper' (local MLX/faster-whisper) [default: {default_asr}]")
    parser.add_argument("--stt-location", default=None,
                        help="Cloud Speech-to-Text V2 location for Chirp 3 (defaults to STT_LOCATION or 'us')")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL,
                        help=f"Whisper model size when --asr-engine whisper is selected (default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--language", default="auto",
                        help="Spoken language code: auto, zh-TW, zh-CN, en, ja, ko (default: auto)")
    parser.add_argument("--force", action="store_true",
                        help="Bypass cached word timestamps (<basename>_words.json) and force re-transcription")
    parser.add_argument("--model", "-m", default=default_model, help=f"Gemini model name (default: {default_model})")
    parser.add_argument("--project", default=None, help="Google Cloud Project ID (defaults to GOOGLE_CLOUD_PROJECT/GCP_PROJECT or ADC)")
    parser.add_argument("--region", default=None, help="Vertex AI region/location (defaults to GOOGLE_CLOUD_LOCATION/GCP_REGION or 'global')")
    parser.add_argument("--bucket", default=None, help="GCS bucket for video and Chirp 3 audio staging (defaults to VIDEO_TRIMMER_BUCKET)")
    parser.add_argument("--keep-gcs-upload", action="store_true", help="Retain staged video in GCS instead of deleting after inference")
    parser.add_argument("--crf", type=int, default=18, help="FFmpeg H.264 CRF quality parameter (default: 18)")
    parser.add_argument("--pacing", "-p", choices=["auto", "dynamic", "compact", "breathing"], default="auto",
                        help="Pacing style: 'auto'/'dynamic' (adaptive CPS margin, default), 'compact', or 'breathing'")
    parser.add_argument("--punch-in-composite", action="store_true",
                        help="Apply a subtle 1.12x framing punch-in on composite pick-up splices in MP4 and NLE timelines")
    parser.add_argument("--agentic", action="store_true", help="Enable Gemini Agentic Video Understanding mode")
    parser.add_argument("--suffix", default=None, help="Custom filename tag suffix (defaults to 'agentic' or 'static')")
    parser.add_argument("--script", "-s", default=None, help="Optional reference shooting script or outline (.txt/.md)")
    parser.add_argument("--cached-json", default=None,
                        help="Path to an existing EDL JSON file to bypass cloud Gemini inference")
    parser.add_argument("--skip-asr", action="store_true", help="Skip word-level ASR transcription and use energy-only fallback")
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

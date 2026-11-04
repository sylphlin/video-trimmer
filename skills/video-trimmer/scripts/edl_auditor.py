"""
Layer 5 deterministic timeline sanitizer and 8-dimension rough-cut quality auditor.

ASD-STE100:
1. Sanitize physical clip boundaries to eliminate overlaps, restore plosive tails, and merge flash-frame micro-clips (< 0.45s).
2. Enforce strict chronological monotonicity (`source_in < source_out` and `c[i].source_out <= c[i+1].source_in`).
3. Execute an 8-dimension rough-cut quality audit and compute a top-level `agent_verdict`.
4. Generate structured JSON and plain technical Markdown audit reports (`.json` and `.md`).
"""

from pathlib import Path

from .acoustic import calculate_clip_cps
from .chirp3_asr import join_words_for_display
from .edl_occlusion import (
    _clip_dual_texts,
    _detect_intra_clip_repeat,
    _detect_tail_head_overlap,
    _is_opening_retake_pair,
)
from .edl_repair import (
    _detect_script_precision_anomalies,
    _detect_script_take_collisions,
)
from .script_aligner import (
    _is_phrase_legitimately_repeated_in_script,
    _script_block_coverage_score,
    extract_script_blocks,
    normalize_text,
)


def sanitize_refined_edl(
    refined_edl: list[dict],
    total_dur: float,
    min_clip_dur: float = 0.45,
    merge_gap_thresh: float = 0.35,
) -> tuple[list[dict], dict]:
    """
    Sanitize physical clip timings after acoustic boundary locking and before NLE/MP4 export.

    ASD-STE100:
    1. Sort clips chronologically and enforce `source_out >= t_last` without crossing `next_word_hard_wall`.
    2. Prune contained or backward-jumping redundant clips and resolve micro-overlaps at their midpoint.
    3. Merge flash-frame micro-clips (< 0.45s) into adjacent clips only when neither boundary was trimmed
       or marked as a composite splice.
    4. Guarantee strict forward monotonicity (`source_in < source_out` and `c[i].source_out <= c[i+1].source_in`).
    """
    stats = {
        "overlaps_resolved": 0,
        "plosive_tails_restored": 0,
        "micro_clips_merged": 0,
    }
    if not refined_edl:
        return [], stats

    sanitized = [dict(c) for c in refined_edl]

    # 1. Enforce source_bounds and plosive tail protection (source_out >= t_last, capped by next_word_hard_wall)
    for c in sanitized:
        s_in = max(0.0, min(total_dur, float(c.get("source_in", 0.0))))
        s_out = max(0.0, min(total_dur, float(c.get("source_out", s_in + 0.5))))
        t_last = c.get("t_last")
        hard_wall = c.get("next_word_hard_wall")
        if t_last is not None and s_out < float(t_last):
            restored = min(total_dur, round(float(t_last) + 0.04, 3))
            if hard_wall is not None:
                restored = min(restored, max(float(t_last), float(hard_wall)))
            s_out = restored
            stats["plosive_tails_restored"] += 1
        elif hard_wall is not None and s_out > float(hard_wall):
            floor_t = float(t_last) if t_last is not None else s_in + 0.15
            s_out = max(floor_t, float(hard_wall))
        if s_out <= s_in:
            s_out = min(total_dur, round(s_in + 0.50, 3))
        c["source_in"] = round(s_in, 3)
        c["source_out"] = round(s_out, 3)
        c["duration"] = round(c["source_out"] - c["source_in"], 3)

    sanitized.sort(key=lambda c: (float(c["source_in"]), float(c["source_out"])))

    # 2. De-conflict adjacent overlaps and prune nested/contained redundant clips
    deconflicted: list[dict] = []
    for curr_c in sanitized:
        if not deconflicted:
            if curr_c["source_out"] > curr_c["source_in"]:
                deconflicted.append(curr_c)
            continue

        prev_c = deconflicted[-1]
        if curr_c["source_out"] <= prev_c["source_out"] + 0.05:
            stats["overlaps_resolved"] += 1
            continue

        if curr_c["source_in"] < prev_c["source_out"]:
            mid = round((prev_c["source_out"] + curr_c["source_in"]) / 2.0, 3)
            prev_t_last = float(prev_c.get("t_last", prev_c["source_in"] + 0.2))
            curr_t_first = float(curr_c.get("t_first", curr_c["source_in"]))
            boundary = max(prev_t_last, min(curr_t_first, mid))

            min_valid_boundary = prev_c["source_in"] + 0.15
            max_valid_boundary = curr_c["source_out"] - 0.15
            if max_valid_boundary < min_valid_boundary:
                stats["overlaps_resolved"] += 1
                continue

            boundary = round(max(min_valid_boundary, min(max_valid_boundary, boundary)), 3)
            prev_c["source_out"] = boundary
            curr_c["source_in"] = boundary
            prev_c["duration"] = round(prev_c["source_out"] - prev_c["source_in"], 3)
            curr_c["duration"] = round(curr_c["source_out"] - curr_c["source_in"], 3)
            stats["overlaps_resolved"] += 1

        if curr_c["source_out"] - curr_c["source_in"] >= 0.15:
            deconflicted.append(curr_c)

    # 3. Coalesce micro-clips (< min_clip_dur) into adjacent clips only when no stumble/splice boundary exists
    merged: list[dict] = []
    for c in deconflicted:
        can_merge_boundary = (
            bool(merged)
            and not merged[-1].get("trimmed_tail")
            and not c.get("trimmed_head")
            and not c.get("is_composite_splice")
            and merged[-1].get("composite_role") != "prefix"
            and c.get("composite_role") != "continuation"
        )
        if (
            can_merge_boundary
            and (c["duration"] < min_clip_dur or merged[-1]["duration"] < min_clip_dur)
            and c["source_out"] > merged[-1]["source_out"]
            and 0.0 <= (c["source_in"] - merged[-1]["source_out"]) <= merge_gap_thresh
        ):
            prev = merged[-1]
            prev["source_out"] = c["source_out"]
            prev["duration"] = round(prev["source_out"] - prev["source_in"], 3)
            if c.get("t_last") is not None:
                prev["t_last"] = max(float(prev.get("t_last") or 0.0), float(c["t_last"]))
            prev_sids = list(prev.get("sentence_ids") or [])
            for sid in (c.get("sentence_ids") or []):
                if sid not in prev_sids:
                    prev_sids.append(sid)
            prev["sentence_ids"] = prev_sids
            prev["transcript"] = join_words_for_display([prev.get("transcript", ""), c.get("transcript", "")])
            prev_asr_tr = prev.get("asr_transcript") or ""
            curr_asr_tr = c.get("asr_transcript") or ""
            if prev_asr_tr or curr_asr_tr:
                prev["asr_transcript"] = join_words_for_display([prev_asr_tr, curr_asr_tr])
            prev["cps"], _ = calculate_clip_cps(prev["transcript"], prev["duration"])
            prev["llm_verified_window"] = bool(
                prev.get("llm_verified_window") and c.get("llm_verified_window")
            )
            stats["micro_clips_merged"] += 1
        else:
            merged.append(c)

    final_clips: list[dict] = []
    for c in merged:
        dur = round(float(c["source_out"]) - float(c["source_in"]), 3)
        if dur <= 0.0:
            continue
        c["duration"] = dur
        c["clip_id"] = len(final_clips) + 1
        c["cps"], _ = calculate_clip_cps(c.get("transcript", ""), dur)
        final_clips.append(c)

    return final_clips, stats


def audit_edl_quality(
    refined_edl: list[dict],
    whisper_units: list[dict],
    total_dur: float,
    video_path: Path,
    script_text: str | None = None,
    script_path: Path | None = None,
    sanitization_stats: dict | None = None,
    micro_window_repairs: list[dict] | None = None,
    asr_engine: str = "chirp3",
    language: str = "auto",
    whisper_model: str = "small",
    punch_in_composite: bool = False,
) -> dict:
    """
    Execute an 8-dimension rough-cut quality audit and compute the top-level agent_verdict.

    ASD-STE100:
    Evaluate acoustic lock rate, timeline monotonicity, retake hygiene, script coverage,
    flash-frame micro-clips, CPS dead-air anomalies, acoustic margins, and compression ratio.
    """
    clip_count = len(refined_edl)
    fatal_violations: list[str] = []
    warnings: list[str] = []

    if clip_count == 0:
        fatal_violations.append("Final EDL contains 0 clips.")
        return {
            "agent_verdict": {
                "pass_quality_gate": False,
                "overall_grade": "FAIL",
                "fatal_violations": fatal_violations,
                "suggested_action": "ONE_SHOT_REMEDIATE",
                "remediation_reason": "Final EDL is empty.",
                "remediation_cmd": None,
            },
            "summary": {
                "clip_count": 0,
                "source_duration_sec": round(total_dur, 2),
                "asr_engine": asr_engine,
                "composite_splices_count": 0,
            },
        }

    # Dimension 1: Acoustic Word-Level Sentence Lock Rate
    if whisper_units:
        locked_clips = [c for c in refined_edl if c.get("sentence_ids")]
        acoustic_lock_rate_pct = round(len(locked_clips) * 100.0 / clip_count, 1)
    else:
        acoustic_lock_rate_pct = 100.0

    if whisper_units and acoustic_lock_rate_pct < 80.0:
        fatal_violations.append(
            f"Acoustic word-level sentence lock rate {acoustic_lock_rate_pct:.1f}% is below the 80.0% minimum threshold."
        )
    elif whisper_units and acoustic_lock_rate_pct < 90.0:
        warnings.append(f"Acoustic word-level sentence lock rate is {acoustic_lock_rate_pct:.1f}% (< 90.0%).")

    # Dimension 2: Timeline Monotonicity & Overlap Check
    overlap_count = 0
    out_of_order_count = 0
    out_of_bounds_count = 0
    for i, c in enumerate(refined_edl):
        s_in = float(c.get("source_in", 0.0))
        s_out = float(c.get("source_out", 0.0))
        if s_in < -0.01 or s_out > total_dur + 0.05 or s_out <= s_in:
            out_of_bounds_count += 1
        if i > 0:
            prev_in = float(refined_edl[i - 1].get("source_in", 0.0))
            prev_out = float(refined_edl[i - 1].get("source_out", 0.0))
            if s_in < prev_in:
                out_of_order_count += 1
            elif s_in < prev_out - 0.005:
                overlap_count += 1

    if overlap_count > 0:
        fatal_violations.append(f"Detected {overlap_count} overlapping clip boundary pair(s).")
    if out_of_order_count > 0:
        fatal_violations.append(f"Detected {out_of_order_count} non-monotonic out-of-order clip(s).")
    if out_of_bounds_count > 0:
        fatal_violations.append(f"Detected {out_of_bounds_count} clip(s) outside [0, {total_dur:.2f}s].")

    # Dimension 3: Residual Retake, Tail-to-Head Stumble, Intra-Clip Repeat & Off-Screen Crew Check
    script_blocks = extract_script_blocks(script_text)
    unit_by_id = {int(u["id"]): u for u in whisper_units if "id" in u}
    residual_retake_pairs = []
    tail_head_retake_pairs = []
    for i in range(clip_count - 1):
        c_a = refined_edl[i]
        c_b = refined_edl[i + 1]
        pair_verified = bool(c_a.get("llm_verified_window") and c_b.get("llm_verified_window"))
        llm_a, w_a = _clip_dual_texts(c_a, unit_by_id, whisper_units)
        llm_b, w_b = _clip_dual_texts(c_b, unit_by_id, whisper_units)
        a_untrimmed = len(normalize_text(llm_a)) >= len(normalize_text(w_a)) * 0.90
        b_untrimmed = len(normalize_text(llm_b)) >= len(normalize_text(w_b)) * 0.90
        opening_whitelisted = _is_phrase_legitimately_repeated_in_script(
            normalize_text(llm_a or w_a)[:6],
            script_text,
            script_blocks,
        )
        gap_sec = float(c_b["source_in"]) - float(c_a["source_out"])
        if gap_sec <= 25.0:
            if not opening_whitelisted and (
                _is_opening_retake_pair(llm_a, llm_b)
                or (a_untrimmed and b_untrimmed and _is_opening_retake_pair(w_a, w_b))
            ):
                pair_entry = {
                    "type": "OPENING_RETAKE",
                    "clip_a": c_a["clip_id"],
                    "clip_b": c_b["clip_id"],
                    "text_a": (llm_a or w_a)[:40],
                    "text_b": (llm_b or w_b)[:40],
                }
                if pair_verified:
                    warnings.append(
                        f"LLM-verified window retained parallel opening phrasing across Clip {c_a['clip_id']}->{c_b['clip_id']}."
                    )
                else:
                    residual_retake_pairs.append(pair_entry)
            else:
                tail_overlap = _detect_tail_head_overlap(
                    llm_a,
                    llm_b,
                    script_text=script_text,
                    script_blocks=script_blocks,
                )
                if tail_overlap is None and a_untrimmed and b_untrimmed:
                    tail_overlap = _detect_tail_head_overlap(
                        w_a,
                        w_b,
                        script_text=script_text,
                        script_blocks=script_blocks,
                    )
                if tail_overlap:
                    pair_entry = {
                        "type": "ERR_TAIL_HEAD_RETAKE",
                        "clip_a": c_a["clip_id"],
                        "clip_b": c_b["clip_id"],
                        "overlap": tail_overlap,
                        "text_a": (llm_a or w_a)[-40:],
                        "text_b": (llm_b or w_b)[:40],
                    }
                    if pair_verified:
                        warnings.append(
                            f"LLM-verified window retained repeated boundary phrase '{tail_overlap}' across Clip {c_a['clip_id']}->{c_b['clip_id']}."
                        )
                    else:
                        residual_retake_pairs.append(pair_entry)
                        tail_head_retake_pairs.append(pair_entry)

    intra_clip_repeats = []
    off_screen_clip_ids = []
    for c in refined_edl:
        sids = c.get("sentence_ids") or []
        if sids and all(not unit_by_id.get(int(sid), {}).get("is_target_speaker", True) for sid in sids if int(sid) in unit_by_id):
            off_screen_clip_ids.append(c["clip_id"])
        llm_c, w_c = _clip_dual_texts(c, unit_by_id, whisper_units)
        rep_phrase = _detect_intra_clip_repeat(
            llm_c,
            script_text=script_text,
            script_blocks=script_blocks,
        )
        if rep_phrase is None and len(normalize_text(llm_c)) >= len(normalize_text(w_c)) * 0.92:
            rep_phrase = _detect_intra_clip_repeat(
                w_c,
                script_text=script_text,
                script_blocks=script_blocks,
            )
        if rep_phrase:
            if c.get("llm_verified_window"):
                warnings.append(
                    f"LLM-verified window retained repeated phrase '{rep_phrase}' in Clip {c['clip_id']}."
                )
            else:
                intra_clip_repeats.append({
                    "type": "ERR_INTRA_CLIP_REPEAT",
                    "clip_id": c["clip_id"],
                    "repeated_phrase": rep_phrase,
                    "text": (llm_c or w_c)[:60],
                })

    if residual_retake_pairs:
        pair_desc = ", ".join(f"Clip {p['clip_a']}->{p['clip_b']}" for p in residual_retake_pairs[:3])
        err_tag = " [ERR_TAIL_HEAD_RETAKE]" if tail_head_retake_pairs else ""
        fatal_violations.append(f"Detected {len(residual_retake_pairs)} residual retake pair(s) ({pair_desc}){err_tag}.")
    if intra_clip_repeats:
        rep_desc = ", ".join(f"Clip {r['clip_id']} ('{r['repeated_phrase']}')" for r in intra_clip_repeats[:3])
        fatal_violations.append(f"Detected {len(intra_clip_repeats)} intra-clip repeat(s) ({rep_desc}) [ERR_INTRA_CLIP_REPEAT].")
    if off_screen_clip_ids:
        fatal_violations.append(f"Detected off-screen speaker audio in Clip(s) {off_screen_clip_ids}.")

    # Dimension 4: Script Clause Coverage, Single-Winner Collision & Phrase Multiplicity Check (Mode A)
    script_take_collisions = []
    script_phrase_multiplicities = []
    if script_blocks:
        combined_norm = "".join(normalize_text(c.get("transcript", "")) for c in refined_edl)
        matched_blocks = []
        missing_blocks = []
        for blk in script_blocks:
            if _script_block_coverage_score(blk["norm_text"], combined_norm) >= 0.50:
                matched_blocks.append(blk["label"])
            else:
                missing_blocks.append(blk["label"])
        script_coverage_pct = round(len(matched_blocks) * 100.0 / len(script_blocks), 1)
        if script_coverage_pct < 75.0:
            fatal_violations.append(
                f"Mode A script block coverage {script_coverage_pct:.1f}% is below 75.0% (missing: {', '.join(missing_blocks[:4])})."
            )
        elif script_coverage_pct < 85.0:
            warnings.append(f"Mode A script block coverage is {script_coverage_pct:.1f}% (< 85.0%).")

        raw_collisions = _detect_script_take_collisions(refined_edl, whisper_units, script_text, min_coverage=0.45)
        for col in raw_collisions:
            if col.get("llm_verified"):
                warnings.append(
                    f"LLM-verified window retained Clips {col['clip_a']}&{col['clip_b']} on {col['label']}."
                )
            else:
                script_take_collisions.append({
                    "type": "ERR_SCRIPT_TAKE_COLLISION",
                    "clip_a": col["clip_a"],
                    "clip_b": col["clip_b"],
                    "block_id": col["block_id"],
                    "label": col["label"],
                    "score_a": col["score_a"],
                    "score_b": col["score_b"],
                })
        if script_take_collisions:
            col_desc = ", ".join(
                f"Clips {c['clip_a']}&{c['clip_b']} on {c['label']}" for c in script_take_collisions[:3]
            )
            fatal_violations.append(
                f"Detected {len(script_take_collisions)} Mode A script block collision(s) ({col_desc}) [ERR_SCRIPT_TAKE_COLLISION]."
            )

        raw_multiplicities, _ = _detect_script_precision_anomalies(
            clips=refined_edl,
            whisper_units=whisper_units,
            total_dur=total_dur,
            script_text=script_text,
            script_blocks=script_blocks,
        )
        for mult in raw_multiplicities:
            if mult.get("llm_verified"):
                warnings.append(
                    f"LLM-verified window retained duplicate script phrase '{mult['phrase']}' across Clips {mult['clip_a']}&{mult['clip_b']}."
                )
            else:
                script_phrase_multiplicities.append({
                    "type": "ERR_SCRIPT_PHRASE_MULTIPLICITY",
                    "clip_a": mult["clip_a"],
                    "clip_b": mult["clip_b"],
                    "phrase": mult["phrase"],
                    "script_occurrences": mult["script_occurrences"],
                })
        if script_phrase_multiplicities:
            mult_desc = ", ".join(
                f"Clips {m['clip_a']}&{m['clip_b']} ('{m['phrase']}')" for m in script_phrase_multiplicities[:3]
            )
            fatal_violations.append(
                f"Detected {len(script_phrase_multiplicities)} duplicate script phrase(s) ({mult_desc}) [ERR_SCRIPT_PHRASE_MULTIPLICITY]."
            )
    else:
        script_coverage_pct = None
        missing_blocks = []

    # Dimension 5: Flash-Frame Micro-Clip Check (< 0.40s)
    micro_clips = [c["clip_id"] for c in refined_edl if float(c.get("duration", 0.0)) < 0.40]
    if micro_clips:
        warnings.append(f"Detected {len(micro_clips)} micro-clip(s) shorter than 0.40s: {micro_clips}.")

    # Dimension 6: Dead-Air & CPS Anomaly Check
    low_cps_clips = [
        c["clip_id"]
        for c in refined_edl
        if float(c.get("duration", 0.0)) >= 3.0 and float(c.get("cps", 3.0)) < 1.2
    ]
    high_cps_clips = [c["clip_id"] for c in refined_edl if float(c.get("cps", 3.0)) > 10.0]
    if low_cps_clips:
        warnings.append(f"Detected {len(low_cps_clips)} clip(s) with low CPS (< 1.2 chars/s, possible internal dead air): {low_cps_clips}.")

    # Dimension 7: Acoustic Margin & Plosive Tail Protection
    plosive_truncated_clips = [
        c["clip_id"]
        for c in refined_edl
        if c.get("t_last") is not None and float(c["source_out"]) < float(c["t_last"]) - 0.01
    ]
    if plosive_truncated_clips:
        fatal_violations.append(
            f"Detected {len(plosive_truncated_clips)} clip(s) where source_out truncates the final spoken word (t_last): {plosive_truncated_clips}."
        )

    avg_in_margin = round(sum(float(c.get("in_margin", 0.0)) for c in refined_edl) / clip_count, 3)
    avg_out_margin = round(sum(float(c.get("out_margin", 0.0)) for c in refined_edl) / clip_count, 3)

    # Dimension 8: Compression & Pacing Summary
    trimmed_dur = round(sum(float(c.get("duration", 0.0)) for c in refined_edl), 2)
    avg_cps = round(sum(float(c.get("cps", 0.0)) for c in refined_edl) / clip_count, 2)
    trim_ratio_pct = round(max(0.0, (total_dur - trimmed_dur) * 100.0 / max(0.01, total_dur)), 1)
    composite_splices_count = sum(
        1 for c in refined_edl if c.get("is_composite_splice") or c.get("composite_role") == "continuation"
    )

    pass_gate = len(fatal_violations) == 0
    if pass_gate and not warnings:
        overall_grade = "EXCELLENT"
    elif pass_gate:
        overall_grade = "PASS"
    else:
        overall_grade = "FAIL"

    remediation_cmd = None
    remediation_reason = None
    if not pass_gate:
        flags = [f'-i "{video_path}"']
        if script_path:
            flags.append(f'-s "{script_path}"')
        flags.append(f"--asr-engine {asr_engine}")
        if asr_engine == "whisper":
            flags.append(f"--whisper-model {whisper_model}")
        if language and language != "auto":
            flags.append(f"--language {language}")
        if punch_in_composite:
            flags.append("--punch-in-composite")
        flags.append("--force")
        remediation_reason = "; ".join(fatal_violations)
        remediation_cmd = "python3 skills/video-trimmer/scripts/video_trimmer.py " + " ".join(flags)

    return {
        "agent_verdict": {
            "pass_quality_gate": pass_gate,
            "overall_grade": overall_grade,
            "fatal_violations": fatal_violations,
            "warnings": warnings,
            "suggested_action": "DELIVER" if pass_gate else "ONE_SHOT_REMEDIATE",
            "remediation_reason": remediation_reason,
            "remediation_cmd": remediation_cmd,
        },
        "summary": {
            "video_file": video_path.name,
            "asr_engine": asr_engine,
            "mode": "Mode A (Script-Anchored)" if script_blocks else "Mode B (Unscripted Intent-Window)",
            "source_duration_sec": round(total_dur, 2),
            "trimmed_duration_sec": trimmed_dur,
            "trim_ratio_pct": trim_ratio_pct,
            "clip_count": clip_count,
            "composite_splices_count": composite_splices_count,
            "average_cps": avg_cps,
            "micro_window_repairs_applied": len(micro_window_repairs or []),
            "sanitization_stats": sanitization_stats or {},
        },
        "dimensions": {
            "1_acoustic_lock": {
                "acoustic_lock_rate_pct": acoustic_lock_rate_pct,
                "pass": (not whisper_units) or (acoustic_lock_rate_pct >= 80.0),
            },
            "2_timeline_integrity": {
                "overlap_count": overlap_count,
                "out_of_order_count": out_of_order_count,
                "out_of_bounds_count": out_of_bounds_count,
                "pass": overlap_count == 0 and out_of_order_count == 0 and out_of_bounds_count == 0,
            },
            "3_retake_and_speaker_hygiene": {
                "residual_retake_count": len(residual_retake_pairs),
                "residual_retake_pairs": residual_retake_pairs,
                "intra_clip_repeat_count": len(intra_clip_repeats),
                "intra_clip_repeats": intra_clip_repeats,
                "off_screen_speaker_clip_count": len(off_screen_clip_ids),
                "pass": (
                    len(residual_retake_pairs) == 0
                    and len(intra_clip_repeats) == 0
                    and len(off_screen_clip_ids) == 0
                ),
            },
            "4_script_coverage": {
                "enabled": bool(script_blocks),
                "total_script_blocks": len(script_blocks),
                "script_coverage_pct": script_coverage_pct,
                "missing_blocks": missing_blocks,
                "script_take_collision_count": len(script_take_collisions),
                "script_take_collisions": script_take_collisions,
                "script_phrase_multiplicity_count": len(script_phrase_multiplicities),
                "script_phrase_multiplicities": script_phrase_multiplicities,
                "pass": (
                    (not script_blocks)
                    or (
                        script_coverage_pct is not None
                        and script_coverage_pct >= 75.0
                        and len(script_take_collisions) == 0
                        and len(script_phrase_multiplicities) == 0
                    )
                ),
            },
            "5_micro_clip_check": {
                "micro_clip_count": len(micro_clips),
                "micro_clip_ids": micro_clips,
                "pass": len(micro_clips) == 0,
            },
            "6_cps_and_dead_air": {
                "low_cps_dead_air_count": len(low_cps_clips),
                "low_cps_clip_ids": low_cps_clips,
                "high_cps_anomaly_count": len(high_cps_clips),
                "pass": len(low_cps_clips) == 0,
            },
            "7_acoustic_margins": {
                "plosive_truncated_count": len(plosive_truncated_clips),
                "avg_in_margin_sec": avg_in_margin,
                "avg_out_margin_sec": avg_out_margin,
                "pass": len(plosive_truncated_clips) == 0,
            },
            "8_compression_summary": {
                "source_duration_sec": round(total_dur, 2),
                "trimmed_duration_sec": trimmed_dur,
                "trim_ratio_pct": trim_ratio_pct,
                "average_cps": avg_cps,
                "pass": trimmed_dur > 0.0,
            },
        },
        "micro_window_repairs": micro_window_repairs or [],
    }


def generate_edl_audit_markdown(report: dict, output_md_path: Path) -> None:
    """
    Write a plain technical Markdown audit report without decorative emojis.

    ASD-STE100:
    Format the 8-dimension audit results and agent_verdict into a structured Markdown table.
    """
    verdict = report.get("agent_verdict", {})
    summary = report.get("summary", {})
    dims = report.get("dimensions", {})
    repairs = report.get("micro_window_repairs", [])
    san = summary.get("sanitization_stats", {})

    d1 = dims.get("1_acoustic_lock", {})
    d1_rate = d1.get("acoustic_lock_rate_pct")
    d2 = dims.get("2_timeline_integrity", {})
    d3 = dims.get("3_retake_and_speaker_hygiene", {})
    d4 = dims.get("4_script_coverage", {})
    d5 = dims.get("5_micro_clip_check", {})
    d6 = dims.get("6_cps_and_dead_air", {})
    d7 = dims.get("7_acoustic_margins", {})

    script_cov_str = (
        f"{d4.get('script_coverage_pct')}% ({d4.get('total_script_blocks')} blocks, "
        f"{d4.get('script_take_collision_count', 0)} collisions, "
        f"{d4.get('script_phrase_multiplicity_count', 0)} multiplicities)"
        if d4.get("enabled")
        else "N/A (Mode B Unscripted)"
    )

    lines = [
        f"# Video Trimmer Quality Audit Report - {summary.get('video_file', '')}",
        "",
        "## 1. Executive Summary & Agent Verdict",
        "",
        "| Field | Value |",
        "| :--- | :--- |",
        f"| **Pass Quality Gate** | `{verdict.get('pass_quality_gate')}` |",
        f"| **Overall Grade** | `{verdict.get('overall_grade')}` |",
        f"| **Suggested Action** | `{verdict.get('suggested_action')}` |",
        f"| **Word Timestamp Engine** | `{summary.get('asr_engine', 'chirp3')}` |",
        f"| **Arbitration Mode** | {summary.get('mode')} |",
        f"| **Source Duration** | {summary.get('source_duration_sec')} s |",
        f"| **Trimmed Duration** | {summary.get('trimmed_duration_sec')} s (Trim Ratio: {summary.get('trim_ratio_pct')}%) |",
        f"| **Retained Clips** | {summary.get('clip_count')} |",
        f"| **Composite Splices (Pick-up)** | {summary.get('composite_splices_count', 0)} |",
        f"| **Average CPS** | {summary.get('average_cps')} chars/s |",
        "",
        "## 2. 8-Dimension Rough-Cut Quality Audit",
        "",
        "| Dimension | Metric | Result | Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **1. Acoustic Word Lock** | Acoustic Lock Rate | {d1_rate}% | {'PASS' if d1.get('pass') else 'FAIL'} |",
        f"| **2. Timeline Monotonicity** | Overlaps / Out-of-Order | {d2.get('overlap_count')} / {d2.get('out_of_order_count')} | {'PASS' if d2.get('pass') else 'FAIL'} |",
        f"| **3. Retake & Speaker Hygiene** | Retakes / Intra Repeats / Crew Clips | {d3.get('residual_retake_count')} / {d3.get('intra_clip_repeat_count', 0)} / {d3.get('off_screen_speaker_clip_count')} | {'PASS' if d3.get('pass') else 'FAIL'} |",
        f"| **4. Script Block Coverage** | Mode A Coverage & Collisions | {script_cov_str} | {'PASS' if d4.get('pass') else 'FAIL'} |",
        f"| **5. Flash-Frame Micro-Clips** | Clips < 0.40s | {d5.get('micro_clip_count')} | {'PASS' if d5.get('pass') else 'WARN'} |",
        f"| **6. Dead-Air & CPS Check** | Low-CPS Clips (< 1.2 CPS) | {d6.get('low_cps_dead_air_count')} | {'PASS' if d6.get('pass') else 'WARN'} |",
        f"| **7. Acoustic Margins** | Mean In / Out Margin | {d7.get('avg_in_margin_sec')}s / {d7.get('avg_out_margin_sec')}s | {'PASS' if d7.get('pass') else 'FAIL'} |",
        f"| **8. Compression Summary** | Trimmed / Source Duration | {summary.get('trimmed_duration_sec')}s / {summary.get('source_duration_sec')}s | PASS |",
        "",
        "## 3. In-Pipeline Self-Healing Summary",
        "",
        f"- **Surgical Micro-Window Re-Scans Applied**: {len(repairs)}",
        f"- **Composite Pick-Up Splices Retained**: {summary.get('composite_splices_count', 0)}",
        f"- **Boundary Overlaps Resolved**: {san.get('overlaps_resolved', 0)}",
        f"- **Plosive Tails Restored (`source_out >= t_last`)**: {san.get('plosive_tails_restored', 0)}",
        f"- **Flash-Frame Micro-Clips Merged**: {san.get('micro_clips_merged', 0)}",
    ]

    if repairs:
        lines.extend([
            "",
            "### Micro-Window Repair Log",
            "",
            "| Type | Window | Repaired Clips | Reason |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for r in repairs:
            lines.append(
                f"| `{r.get('type')}` | `{r.get('window')}` | {r.get('repaired_clip_count')} | {r.get('reason')} |"
            )

    if verdict.get("fatal_violations"):
        lines.extend(["", "## 4. Fatal Violations", ""])
        for v in verdict["fatal_violations"]:
            lines.append(f"- {v}")

    if verdict.get("warnings"):
        lines.extend(["", "## 5. Audit Warnings", ""])
        for w in verdict["warnings"]:
            lines.append(f"- {w}")

    lines.append("")
    Path(output_md_path).write_text("\n".join(lines), encoding="utf-8")

"""
Closed-loop EDL anomaly detection and surgical 15-to-90s micro-window repair.

ASD-STE100:
1. Detect Mode A script block collisions, duplicate script clauses, and unscripted extraneous spans.
2. Coalesce adjacent pre-render anomalies into unified 15-to-90-second micro-windows.
3. Re-scan only affected video slices via Vertex AI VideoMetadata and stamp `llm_verified_window`.
"""

import difflib
import logging
import math

from .chirp3_asr import join_words_for_display
from .edl_occlusion import (
    _clip_dual_texts,
    _clip_time_bounds,
    _detect_intra_clip_repeat,
    _detect_tail_head_overlap,
    _is_complementary_composite_pair,
    _is_opening_retake_pair,
    _resolve_clip_sentence_ids,
    deduplicate_and_sort_clips,
)
from .script_aligner import (
    _count_fuzzy_script_occurrences,
    _is_phrase_legitimately_repeated_in_script,
    _script_block_coverage_score,
    _unit_overlaps_script_block,
    extract_script_blocks,
    normalize_text,
)
from .transcribe import is_viable_sub_span

logger = logging.getLogger(__name__)

MAX_MICRO_WINDOW_REPAIRS = 5
MIN_MICRO_WINDOW_SEC = 15.0
MAX_MICRO_WINDOW_SEC = 90.0


def _detect_script_take_collisions(
    clips: list[dict],
    whisper_units: list[dict],
    script_text: str | None,
    min_coverage: float = 0.45,
) -> list[dict]:
    """
    Detect when multiple clips in Mode A map to the same `[Script Block NN]` (`SCRIPT_TAKE_COLLISION`).

    ASD-STE100:
    Flag pairs of clips that both cover >= 45% of the same script block and share non-whitelisted
    overlapping content, while exempting complementary composite `prefix -> continuation` pairs.
    """
    if not script_text or len(clips) <= 1:
        return []

    script_blocks = extract_script_blocks(script_text)
    if not script_blocks:
        return []

    unit_by_id = {int(u["id"]): u for u in whisper_units if "id" in u}
    block_by_id = {b["block_id"]: b for b in script_blocks}
    clip_matches: list[tuple[int, dict, int, float, str]] = []

    for idx, c in enumerate(clips):
        llm_text, w_text = _clip_dual_texts(c, unit_by_id, whisper_units)
        llm_norm = normalize_text(llm_text)
        w_norm = normalize_text(w_text)
        # When the clip intentionally trimmed a sub-span, evaluate against the trimmed transcript
        eval_texts = (llm_text,) if (llm_norm and w_norm and len(llm_norm) < len(w_norm) * 0.90) else (llm_text, w_text)
        best_blk_id = None
        best_score = 0.0
        best_norm = llm_norm or w_norm
        for cand_text in eval_texts:
            norm = normalize_text(cand_text)
            if not norm:
                continue
            for blk in script_blocks:
                score = _script_block_coverage_score(blk["norm_text"], norm)
                if score >= min_coverage and score > best_score and len(norm) <= len(blk["norm_text"]) * 1.85:
                    best_score = score
                    best_blk_id = blk["block_id"]
                    best_norm = norm
        if best_blk_id is not None:
            clip_matches.append((idx, c, best_blk_id, best_score, best_norm))

    collisions: list[dict] = []
    for i in range(len(clip_matches)):
        idx_a, clip_a, blk_a, score_a, norm_a = clip_matches[i]
        for j in range(i + 1, len(clip_matches)):
            idx_b, clip_b, blk_b, score_b, norm_b = clip_matches[j]
            if blk_a != blk_b:
                continue
            if idx_b == idx_a + 1 and _is_complementary_composite_pair(
                clip_a, clip_b, unit_by_id, whisper_units, script_blocks
            ):
                continue
            matcher = difflib.SequenceMatcher(None, norm_a, norm_b, autojunk=False)
            shared = matcher.find_longest_match(0, len(norm_a), 0, len(norm_b))
            shared_str = norm_a[shared.a : shared.a + shared.size] if shared.size > 0 else ""
            is_script_repeat = (
                shared.size >= 4
                and _is_phrase_legitimately_repeated_in_script(shared_str, script_text, script_blocks)
            )
            if (shared.size >= 4 and not is_script_repeat) or (score_a >= 0.65 and score_b >= 0.65):
                blk = block_by_id[blk_a]
                collisions.append({
                    "idx_a": idx_a,
                    "idx_b": idx_b,
                    "clip_a": clip_a.get("clip_id", idx_a + 1),
                    "clip_b": clip_b.get("clip_id", idx_b + 1),
                    "block_id": blk_a,
                    "label": blk["label"],
                    "script_block": blk,
                    "score_a": round(score_a, 2),
                    "score_b": round(score_b, 2),
                    "llm_verified": bool(
                        clip_a.get("llm_verified_window") and clip_b.get("llm_verified_window")
                    ),
                })
    return collisions


def _detect_script_precision_anomalies(
    clips: list[dict],
    whisper_units: list[dict],
    total_dur: float,
    script_text: str | None,
    script_blocks: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Detect Mode A script phrase multiplicities and unscripted extraneous spans.

    ASD-STE100:
    1. Flag `SCRIPT_PHRASE_MULTIPLICITY` when two clips within 45s repeat a >= 5 CJK char
       (or >= 18 Latin char) clause more times than it occurs in the reference script.
    2. Flag `UNSCRIPTED_EXTRANEOUS_SPAN` when physical `asr_transcript` leaks >= 5 unscripted
       characters beyond `transcript` or retains a >= 6 CJK char unscripted false start/tail.
    """
    blocks = script_blocks if script_blocks is not None else (extract_script_blocks(script_text) if script_text else [])
    if not blocks or not clips:
        return [], []

    script_norm = "".join(b.get("norm_text", "") for b in blocks if b.get("norm_text"))
    if not script_norm:
        return [], []

    unit_by_id = {int(u["id"]): u for u in whisper_units if "id" in u}
    multiplicities: list[dict] = []
    extraneous_spans: list[dict] = []

    bounds: list[tuple[float, float]] = []
    eff_norms: list[str] = []
    for idx, c in enumerate(clips):
        t_in, t_out = _clip_time_bounds(c, unit_by_id, whisper_units)
        bounds.append((t_in, t_out))
        has_aligned_asr = bool((c.get("asr_transcript") or "").strip())
        llm_t, w_t = _clip_dual_texts(c, unit_by_id, whisper_units)
        eff_t = w_t if has_aligned_asr else llm_t
        eff_n = normalize_text(eff_t)
        eff_norms.append(eff_n)

        # Check C-2a: Physical ASR alignment leak beyond LLM-cleaned transcript
        if has_aligned_asr and not c.get("llm_verified_window"):
            llm_n = normalize_text(llm_t)
            asr_n = normalize_text(w_t)
            if llm_n and (len(asr_n) - len(llm_n)) >= 5:
                sm_leak = difflib.SequenceMatcher(None, asr_n, llm_n, autojunk=False)
                for tag, i1, i2, j1, j2 in sm_leak.get_opcodes():
                    if tag == "equal":
                        continue
                    ins_len = i2 - i1
                    tgt_len = j2 - j1
                    if (ins_len - tgt_len) >= 5:
                        leak_str = asr_n[i1:i2]
                        if _count_fuzzy_script_occurrences(leak_str, script_norm) == 0:
                            win_start = max(0.0, t_in - 3.0)
                            win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, t_out + 3.0))
                            extraneous_spans.append({
                                "type": "UNSCRIPTED_EXTRANEOUS_SPAN",
                                "idx": idx,
                                "clip_id": c.get("clip_id", idx + 1),
                                "extraneous_text": leak_str,
                                "replace_range": (idx, idx + 1),
                                "win_start": round(win_start, 2),
                                "win_end": round(win_end, 2),
                                "reason": (
                                    f"Clip {idx + 1} physical ASR alignment retained extraneous unscripted span "
                                    f"'{leak_str}' absent from both the cleaned transcript and reference script."
                                ),
                            })
                            break

    # Check C-1: Script phrase multiplicity across distinct clips within 45s
    for idx_a in range(len(clips)):
        norm_a = eff_norms[idx_a]
        if not norm_a:
            continue
        t_a_in, t_a_out = bounds[idx_a]
        for idx_b in range(idx_a + 1, len(clips)):
            norm_b = eff_norms[idx_b]
            if not norm_b:
                continue
            t_b_in, t_b_out = bounds[idx_b]
            if (t_b_in - t_a_out) > 45.0 or (t_b_out - t_a_in) > MAX_MICRO_WINDOW_SEC:
                continue
            if idx_b == idx_a + 1 and _is_complementary_composite_pair(
                clips[idx_a], clips[idx_b], unit_by_id, whisper_units, blocks
            ):
                continue

            shared = difflib.SequenceMatcher(None, norm_a, norm_b, autojunk=False).find_longest_match(
                0, len(norm_a), 0, len(norm_b)
            )
            if shared.size < 5:
                continue
            shared_str = norm_a[shared.a : shared.a + shared.size]
            cjk_count = sum(1 for ch in shared_str if "\u4e00" <= ch <= "\u9fff")
            if not (cjk_count >= 5 or len(shared_str) >= 18):
                continue
            if not (
                len(shared_str) >= min(len(norm_a), len(norm_b)) * 0.45
                or cjk_count >= 7
                or len(shared_str) >= 24
            ):
                continue

            script_occ = _count_fuzzy_script_occurrences(shared_str, script_norm)
            if script_occ >= 2:
                continue

            pair_verified = bool(
                clips[idx_a].get("llm_verified_window") and clips[idx_b].get("llm_verified_window")
            )
            win_start = max(0.0, t_a_in - 3.0)
            win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, t_b_out + 3.0))
            multiplicities.append({
                "type": "SCRIPT_PHRASE_MULTIPLICITY",
                "idx_a": idx_a,
                "idx_b": idx_b,
                "clip_a": clips[idx_a].get("clip_id", idx_a + 1),
                "clip_b": clips[idx_b].get("clip_id", idx_b + 1),
                "phrase": shared_str,
                "script_occurrences": script_occ,
                "llm_verified": pair_verified,
                "replace_range": (idx_a, idx_b + 1),
                "win_start": round(win_start, 2),
                "win_end": round(win_end, 2),
                "reason": (
                    f"Clips {idx_a + 1} and {idx_b + 1} both speak '{shared_str}', which occurs "
                    f"{script_occ} time(s) in the reference script; keep only the final clean take."
                ),
            })

    # Check C-2b: Global sequence alignment against script_norm for unscripted false starts or tails
    already_flagged_clips = {item["idx"] for item in extraneous_spans}
    clip_char_ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for idx, c_n in enumerate(eff_norms):
        if not c_n:
            continue
        clip_char_ranges.append((cursor, cursor + len(c_n), idx))
        cursor += len(c_n)

    selected_norm = "".join(eff_norms)
    if selected_norm and clip_char_ranges:
        sm_global = difflib.SequenceMatcher(None, selected_norm, script_norm, autojunk=False)
        for tag, i1, i2, j1, j2 in sm_global.get_opcodes():
            if tag == "equal":
                continue
            ins_len = i2 - i1
            scr_len = j2 - j1
            if ins_len < 6:
                continue
            span_str = selected_norm[i1:i2]
            cjk_ins = sum(1 for ch in span_str if "\u4e00" <= ch <= "\u9fff")
            is_significant_unscripted = (cjk_ins >= 6 and (ins_len - scr_len) >= 5) or (
                ins_len >= 22 and (ins_len - scr_len) >= 16
            )
            if not is_significant_unscripted:
                continue
            if _count_fuzzy_script_occurrences(span_str, script_norm) > 0:
                continue

            best_owner = None
            best_ov = 0
            best_bounds = (0, 0)
            for c_start, c_end, c_idx in clip_char_ranges:
                ov_s = max(i1, c_start)
                ov_e = min(i2, c_end)
                if ov_e - ov_s > best_ov:
                    best_ov = ov_e - ov_s
                    best_owner = c_idx
                    best_bounds = (c_start, c_end)

            if best_owner is None or best_owner in already_flagged_clips:
                continue
            if clips[best_owner].get("llm_verified_window"):
                continue

            c_start, c_end = best_bounds
            ov_s = max(i1, c_start)
            ov_e = min(i2, c_end)
            touches_boundary = (ov_s - c_start) <= 2 or (c_end - ov_e) <= 2
            if not touches_boundary and cjk_ins < 8 and ins_len < 28:
                continue

            t_in, t_out = bounds[best_owner]
            win_start = max(0.0, t_in - 3.0)
            win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, t_out + 3.0))
            already_flagged_clips.add(best_owner)
            extraneous_spans.append({
                "type": "UNSCRIPTED_EXTRANEOUS_SPAN",
                "idx": best_owner,
                "clip_id": clips[best_owner].get("clip_id", best_owner + 1),
                "extraneous_text": selected_norm[ov_s:ov_e],
                "replace_range": (best_owner, best_owner + 1),
                "win_start": round(win_start, 2),
                "win_end": round(win_end, 2),
                "reason": (
                    f"Clip {best_owner + 1} retains an unscripted spoken span '{selected_norm[ov_s:ov_e]}' "
                    "not present in the reference script (likely an unpruned false start or off-script remark)."
                ),
            })

    return multiplicities, extraneous_spans


def detect_micro_window_anomalies(
    model_edl: dict,
    whisper_units: list[dict],
    total_dur: float,
    script_text: str | None = None,
) -> list[dict]:
    """
    Inspect the EDL before video rendering and coalesce 15-to-90s micro-windows
    that require targeted Gemini multimodal re-arbitration.

    ASD-STE100:
    Identify and merge overlapping pre-render anomalies for Gemini micro-window repair:
    1. POTENTIAL_RESIDUAL_RETAKE / TAIL_HEAD_RETAKE: Adjacent clips share retake phrasing,
       tail-to-head stumbled clauses, or overlapping physical sub-spans.
    2. INTRA_CLIP_REPEAT: A single clip contains an immediate internal repetition or punctuated stutter.
    3. SCRIPT_TAKE_COLLISION / SCRIPT_PHRASE_MULTIPLICITY / UNSCRIPTED_EXTRANEOUS_SPAN:
       Mode A script take collisions, duplicate script phrases, or unscripted extraneous spans.
    4. UNANCHORED_CLIP: Clip lacks valid sentence IDs.
    5. MISSING_SCRIPT_BLOCK: A script block exists in ASR sentences but is absent from the EDL.
    6. PARTIAL_SCRIPT_BLOCK_BACKFILL: Only a partial tail pick-up of a script block was selected
       while the missing opening prefix exists in adjacent unselected sentences.
    """
    clips = model_edl.get("final_edl", []) if isinstance(model_edl, dict) else []
    if not whisper_units or not clips:
        return []

    unit_by_id = {int(u["id"]): u for u in whisper_units if "id" in u}
    script_blocks = extract_script_blocks(script_text) if script_text else []
    anomalies: list[dict] = []

    def _add_or_merge_anomaly(cand: dict) -> None:
        r_range = cand.get("replace_range")
        if r_range is not None:
            r_start, r_end = r_range
            for ex in anomalies:
                ex_range = ex.get("replace_range")
                if ex_range is None:
                    continue
                ex_start, ex_end = ex_range
                if max(r_start, ex_start) < min(r_end, ex_end):
                    merged_win_start = round(min(ex["win_start"], cand["win_start"]), 2)
                    merged_win_end = round(max(ex["win_end"], cand["win_end"]), 2)
                    if (merged_win_end - merged_win_start) <= MAX_MICRO_WINDOW_SEC:
                        ex["replace_range"] = (min(r_start, ex_start), max(r_end, ex_end))
                        ex["win_start"] = merged_win_start
                        ex["win_end"] = merged_win_end
                        sub_types = list(ex.get("sub_types") or [ex["type"]])
                        if cand["type"] not in sub_types:
                            sub_types.append(cand["type"])
                        ex["sub_types"] = sub_types
                        if cand.get("reason") and cand["reason"] not in ex["reason"]:
                            ex["reason"] = f"{ex['reason']} | {cand['reason']}"
                        if ex.get("script_block") is None and cand.get("script_block") is not None:
                            ex["script_block"] = cand["script_block"]
                    return
        if len(anomalies) < MAX_MICRO_WINDOW_REPAIRS:
            cand_copy = dict(cand)
            cand_copy.setdefault("sub_types", [cand["type"]])
            anomalies.append(cand_copy)

    # 1. Check adjacent clips for overlapping sub-spans, opening retakes, or tail-to-head stumbles
    for idx in range(len(clips) - 1):
        c_curr = clips[idx]
        c_next = clips[idx + 1]
        if c_curr.get("llm_verified_window") and c_next.get("llm_verified_window"):
            continue
        if _is_complementary_composite_pair(c_curr, c_next, unit_by_id, whisper_units, script_blocks):
            continue

        sids_curr = _resolve_clip_sentence_ids(c_curr, whisper_units)
        sids_next = _resolve_clip_sentence_ids(c_next, whisper_units)

        t_curr_in, t_curr_out = _clip_time_bounds(c_curr, unit_by_id, whisper_units)
        t_next_in, t_next_out = _clip_time_bounds(c_next, unit_by_id, whisper_units)

        span_sec = t_next_out - t_curr_in
        if span_sec > MAX_MICRO_WINDOW_SEC or (t_next_in - t_curr_out) > 30.0:
            continue

        overlap_ids = set(sids_curr).intersection(set(sids_next))
        time_overlaps = (t_next_in < t_curr_out - 0.05) and (t_next_out > t_curr_in + 0.05)
        llm_curr, w_curr = _clip_dual_texts(c_curr, unit_by_id, whisper_units)
        llm_next, w_next = _clip_dual_texts(c_next, unit_by_id, whisper_units)
        curr_untrimmed = len(normalize_text(llm_curr)) >= len(normalize_text(w_curr)) * 0.92
        next_untrimmed = len(normalize_text(llm_next)) >= len(normalize_text(w_next)) * 0.92
        opening_whitelisted = _is_phrase_legitimately_repeated_in_script(
            normalize_text(llm_curr or w_curr)[:6],
            script_text,
            script_blocks,
        )

        anomaly_type = None
        reason = ""
        if overlap_ids and time_overlaps:
            anomaly_type = "POTENTIAL_RESIDUAL_RETAKE"
            reason = f"Adjacent clips {idx + 1} and {idx + 2} share overlapping Sentence IDs {sorted(overlap_ids)}."
        elif not opening_whitelisted and (
            (llm_curr and llm_next and _is_opening_retake_pair(llm_curr, llm_next))
            or (curr_untrimmed and next_untrimmed and w_curr and w_next and _is_opening_retake_pair(w_curr, w_next))
        ):
            anomaly_type = "POTENTIAL_RESIDUAL_RETAKE"
            reason = (
                f"Adjacent clips {idx + 1} ('{llm_curr[:24]}') and {idx + 2} ('{llm_next[:24]}') "
                "contain overlapping opening clauses that may indicate an unpruned retake or parallelism."
            )
        else:
            tail_overlap = _detect_tail_head_overlap(
                llm_curr,
                llm_next,
                script_text=script_text,
                script_blocks=script_blocks,
            )
            if tail_overlap is None and curr_untrimmed and next_untrimmed:
                tail_overlap = _detect_tail_head_overlap(
                    w_curr,
                    w_next,
                    script_text=script_text,
                    script_blocks=script_blocks,
                )
            if tail_overlap:
                anomaly_type = "TAIL_HEAD_RETAKE"
                reason = (
                    f"Clip {idx + 1} tail and Clip {idx + 2} head repeat the phrase '{tail_overlap}', "
                    "indicating a possible trailing false start before a sentence restart."
                )

        if anomaly_type is not None:
            win_start = max(0.0, t_curr_in - 3.0)
            win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, t_next_out + 3.0))
            if (win_end - win_start) <= MAX_MICRO_WINDOW_SEC:
                _add_or_merge_anomaly({
                    "type": anomaly_type,
                    "replace_range": (idx, idx + 2),
                    "win_start": round(win_start, 2),
                    "win_end": round(win_end, 2),
                    "reason": reason,
                })

    # 2. Check for intra-clip immediate repetitions, punctuated short stutters, or multi-sentence internal retakes
    for idx, c in enumerate(clips):
        if c.get("llm_verified_window"):
            continue
        llm_text, w_text = _clip_dual_texts(c, unit_by_id, whisper_units)
        llm_norm = normalize_text(llm_text)
        w_norm = normalize_text(w_text)
        clip_untrimmed = len(llm_norm) >= len(w_norm) * 0.92
        repeated_sub = _detect_intra_clip_repeat(
            llm_text,
            script_text=script_text,
            script_blocks=script_blocks,
            include_punctuated_short_stutter=True,
        )
        if repeated_sub is None and clip_untrimmed:
            repeated_sub = _detect_intra_clip_repeat(
                w_text,
                script_text=script_text,
                script_blocks=script_blocks,
                include_punctuated_short_stutter=True,
            )
        if repeated_sub:
            t_in, t_out = _clip_time_bounds(c, unit_by_id, whisper_units)
            win_start = max(0.0, t_in - 3.0)
            win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, t_out + 3.0))
            if (win_end - win_start) <= MAX_MICRO_WINDOW_SEC:
                _add_or_merge_anomaly({
                    "type": "INTRA_CLIP_REPEAT",
                    "replace_range": (idx, idx + 1),
                    "win_start": round(win_start, 2),
                    "win_end": round(win_end, 2),
                    "reason": f"Clip {idx + 1} contains an internal repeated phrase '{repeated_sub}' that requires stutter vs. emphasis review.",
                })
            continue

        sids = _resolve_clip_sentence_ids(c, whisper_units)
        if len(sids) >= 2:
            for k in range(len(sids) - 1):
                u_a = unit_by_id.get(sids[k])
                u_b = unit_by_id.get(sids[k + 1])
                if not u_a or not u_b:
                    continue
                t_a = str(u_a.get("text", "") or "")
                t_b = str(u_b.get("text", "") or "")
                n_a = normalize_text(t_a)
                if not n_a:
                    continue
                open_prefix = n_a[: min(6, len(n_a))]
                internal_anomaly_type = None
                internal_reason = ""
                if (
                    len(open_prefix) >= 3
                    and not _is_phrase_legitimately_repeated_in_script(open_prefix, script_text, script_blocks)
                    and _is_opening_retake_pair(t_a, t_b)
                    and (clip_untrimmed or llm_norm.count(open_prefix) >= 2)
                ):
                    internal_anomaly_type = "POTENTIAL_RESIDUAL_RETAKE"
                    internal_reason = (
                        f"Clip {idx + 1} spans Sentence IDs {sids[k]}->{sids[k + 1]} with repeated opening '{t_a[:20]}'."
                    )
                else:
                    internal_ov = _detect_tail_head_overlap(
                        t_a,
                        t_b,
                        script_text=script_text,
                        script_blocks=script_blocks,
                    )
                    if internal_ov and (clip_untrimmed or llm_norm.count(internal_ov) >= 2):
                        internal_anomaly_type = "TAIL_HEAD_RETAKE"
                        internal_reason = (
                            f"Clip {idx + 1} spans Sentence IDs {sids[k]}->{sids[k + 1]} with boundary stumble '{internal_ov}'."
                        )
                if internal_anomaly_type is not None:
                    t_in, t_out = _clip_time_bounds(c, unit_by_id, whisper_units)
                    win_start = max(0.0, t_in - 3.0)
                    win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, t_out + 3.0))
                    if (win_end - win_start) <= MAX_MICRO_WINDOW_SEC:
                        _add_or_merge_anomaly({
                            "type": internal_anomaly_type,
                            "replace_range": (idx, idx + 1),
                            "win_start": round(win_start, 2),
                            "win_end": round(win_end, 2),
                            "reason": internal_reason,
                        })
                    break

    # 3. Mode A: Check for script block take collisions, phrase multiplicities, and unscripted spans
    if script_text:
        collisions = _detect_script_take_collisions(clips, whisper_units, script_text, min_coverage=0.45)
        for col in collisions:
            if col.get("llm_verified"):
                continue
            idx_a, idx_b = col["idx_a"], col["idx_b"]
            t_a_in, _ = _clip_time_bounds(clips[idx_a], unit_by_id, whisper_units)
            _, t_b_out = _clip_time_bounds(clips[idx_b], unit_by_id, whisper_units)
            win_start = max(0.0, t_a_in - 3.0)
            win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, t_b_out + 3.0))
            if (win_end - win_start) <= MAX_MICRO_WINDOW_SEC:
                _add_or_merge_anomaly({
                    "type": "SCRIPT_TAKE_COLLISION",
                    "replace_range": (idx_a, idx_b + 1),
                    "win_start": round(win_start, 2),
                    "win_end": round(win_end, 2),
                    "script_block": col["script_block"],
                    "reason": (
                        f"Clips {col['clip_a']} and {col['clip_b']} both map to {col['label']} "
                        f"(coverage {col['score_a']} vs {col['score_b']}); arbitrate single winning take vs. valid multi-sentence sequence."
                    ),
                })

        multiplicities, extraneous_spans = _detect_script_precision_anomalies(
            clips=clips,
            whisper_units=whisper_units,
            total_dur=total_dur,
            script_text=script_text,
            script_blocks=script_blocks,
        )
        for mult in multiplicities:
            if mult.get("llm_verified"):
                continue
            if (mult["win_end"] - mult["win_start"]) <= MAX_MICRO_WINDOW_SEC:
                _add_or_merge_anomaly({
                    "type": "SCRIPT_PHRASE_MULTIPLICITY",
                    "replace_range": mult["replace_range"],
                    "win_start": mult["win_start"],
                    "win_end": mult["win_end"],
                    "reason": mult["reason"],
                })
        for ext in extraneous_spans:
            if (ext["win_end"] - ext["win_start"]) <= MAX_MICRO_WINDOW_SEC:
                _add_or_merge_anomaly({
                    "type": "UNSCRIPTED_EXTRANEOUS_SPAN",
                    "replace_range": ext["replace_range"],
                    "win_start": ext["win_start"],
                    "win_end": ext["win_end"],
                    "reason": ext["reason"],
                })

    # 4. Check for unanchored clips (missing sentence_ids)
    for idx, c in enumerate(clips):
        sids = _resolve_clip_sentence_ids(c, whisper_units)
        if not sids:
            raw_in = float(c.get("source_in", 0.0))
            raw_out = float(c.get("source_out", min(total_dur, raw_in + 10.0)))
            win_start = max(0.0, raw_in - 8.0)
            win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, raw_out + 8.0))
            if (win_end - win_start) <= MAX_MICRO_WINDOW_SEC:
                _add_or_merge_anomaly({
                    "type": "UNANCHORED_CLIP",
                    "replace_range": (idx, idx + 1),
                    "win_start": round(win_start, 2),
                    "win_end": round(win_end, 2),
                    "reason": f"Clip {idx + 1} has no valid sentence_ids and requires exact sentence anchoring.",
                })

    # 5 & 6. Mode A: Check for missed script blocks or partial pick-up blocks missing their opening prefix
    if script_blocks and len(anomalies) < MAX_MICRO_WINDOW_REPAIRS:
        selected_sids = set()
        selected_texts: list[str] = []
        clip_norms: list[str] = []
        for c in clips:
            c_sids = _resolve_clip_sentence_ids(c, whisper_units)
            selected_sids.update(c_sids)
            c_text = c.get("transcript", "") or ""
            if not c_text and c_sids:
                c_text = join_words_for_display([str(unit_by_id[sid].get("text", "")) for sid in c_sids if sid in unit_by_id])
            c_n = normalize_text(c_text)
            clip_norms.append(c_n)
            selected_texts.append(c_n)
        selected_norm_text = "".join(t for t in selected_texts if t)

        for blk in script_blocks:
            b_norm = blk["norm_text"]
            blk_cov = _script_block_coverage_score(b_norm, selected_norm_text)

            # Case 6: Partial Tail Pick-Up Backfill (0.35 <= blk_cov < 0.78, opening prefix missing)
            if 0.35 <= blk_cov < 0.78 and len(b_norm) >= 8:
                head_len = min(12, max(4, len(b_norm) // 2))
                b_head = b_norm[:head_len]
                tail_clip_idx = None
                for c_i, c_n in enumerate(clip_norms):
                    if _script_block_coverage_score(b_norm, c_n) >= 0.35 and _script_block_coverage_score(b_head, c_n) < 0.40:
                        tail_clip_idx = c_i
                        break
                if tail_clip_idx is not None and not clips[tail_clip_idx].get("llm_verified_window"):
                    t_c_in, t_c_out = _clip_time_bounds(clips[tail_clip_idx], unit_by_id, whisper_units)
                    prefix_cands = [
                        u for u in whisper_units
                        if int(u.get("id", 0)) not in selected_sids
                        and float(u.get("end", 0.0)) <= t_c_in + 0.1
                        and (t_c_in - float(u.get("end", 0.0))) <= 35.0
                        and _script_block_coverage_score(b_head, normalize_text(u.get("text", ""))) >= 0.70
                        and is_viable_sub_span(u.get("text", ""))
                    ]
                    if prefix_cands:
                        chosen_pre = prefix_cands[-1]
                        win_start = max(0.0, float(chosen_pre.get("start", 0.0)) - 3.0)
                        win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, t_c_out + 3.0))
                        if (win_end - win_start) <= MAX_MICRO_WINDOW_SEC:
                            _add_or_merge_anomaly({
                                "type": "PARTIAL_SCRIPT_BLOCK_BACKFILL",
                                "replace_range": (tail_clip_idx, tail_clip_idx + 1),
                                "win_start": round(win_start, 2),
                                "win_end": round(win_end, 2),
                                "script_block": blk,
                                "reason": (
                                    f"Clip {tail_clip_idx + 1} covers only the tail pick-up of {blk['label']} "
                                    f"({blk_cov * 100:.0f}%); unselected Sentence ID {chosen_pre['id']} contains the missing opening prefix."
                                ),
                            })
                            continue

            if blk_cov >= 0.55 or len(anomalies) >= MAX_MICRO_WINDOW_REPAIRS:
                continue

            # Case 5: Completely missed script block in unselected ASR clusters
            cand_units = [
                u for u in whisper_units
                if int(u.get("id", 0)) not in selected_sids
                and _unit_overlaps_script_block(b_norm, normalize_text(u.get("text", "")))
            ]
            if not cand_units:
                continue

            clusters: list[list[dict]] = []
            for u in cand_units:
                if not clusters:
                    clusters.append([u])
                else:
                    prev_end = float(clusters[-1][-1].get("end", 0.0))
                    curr_start = float(u.get("start", 0.0))
                    if (curr_start - prev_end) <= 12.0:
                        clusters[-1].append(u)
                    else:
                        clusters.append([u])

            valid_clusters: list[tuple[float, list[dict]]] = []
            for cl in clusters:
                cl_norm = "".join(normalize_text(u.get("text", "")) for u in cl)
                cov = _script_block_coverage_score(b_norm, cl_norm)
                if cov >= 0.50:
                    valid_clusters.append((cov, cl))

            if not valid_clusters:
                continue

            _, chosen_cluster = valid_clusters[-1]
            win_start = max(0.0, float(chosen_cluster[0].get("start", 0.0)) - 4.0)
            win_end = min(total_dur, max(win_start + MIN_MICRO_WINDOW_SEC, float(chosen_cluster[-1].get("end", 0.0)) + 4.0))
            if (win_end - win_start) <= MAX_MICRO_WINDOW_SEC:
                _add_or_merge_anomaly({
                    "type": "MISSING_SCRIPT_BLOCK",
                    "replace_range": None,
                    "win_start": round(win_start, 2),
                    "win_end": round(win_end, 2),
                    "script_block": blk,
                    "reason": f"{blk['label']} was omitted from the initial EDL but matches sentences in {win_start:.1f}s..{win_end:.1f}s.",
                })

    return anomalies[:MAX_MICRO_WINDOW_REPAIRS]


def repair_edl_micro_windows(
    model_edl: dict,
    whisper_units: list[dict],
    total_dur: float,
    run_window_inference_fn,
    script_text: str | None = None,
) -> tuple[dict, list[dict]]:
    """
    Execute targeted micro-window re-scans for detected pre-render anomalies and splice
    repaired clips back into model_edl with Last-Take-Anchored arbitration.

    ASD-STE100:
    1. Normalize chronological clip order and composite seam occlusions first.
    2. Re-scan only coalesced 15-to-90s video windows and stamp verified clips with `llm_verified_window`.
    """
    raw_clips = model_edl.get("final_edl", []) if isinstance(model_edl, dict) else []
    if not whisper_units or not raw_clips:
        return model_edl, []

    deduped_initial = deduplicate_and_sort_clips(
        clips=raw_clips,
        whisper_units=whisper_units,
        script_text=script_text,
    )
    working_edl = dict(model_edl)
    working_edl["final_edl"] = deduped_initial

    anomalies = detect_micro_window_anomalies(
        model_edl=working_edl,
        whisper_units=whisper_units,
        total_dur=total_dur,
        script_text=script_text,
    )
    if not anomalies:
        return working_edl, []

    clips = [dict(c) for c in deduped_initial]
    applied_repairs = []

    indexed_anomalies = sorted(
        anomalies,
        key=lambda a: (a["replace_range"][0] if a.get("replace_range") is not None else 999999),
        reverse=True,
    )

    for anomaly in indexed_anomalies:
        win_start = anomaly["win_start"]
        win_end = anomaly["win_end"]
        win_units = [
            u for u in whisper_units
            if float(u.get("end", 0.0)) >= win_start and float(u.get("start", 0.0)) <= win_end
        ]
        if not win_units:
            continue

        start_offset = f"{int(math.floor(win_start))}s"
        end_offset = f"{int(math.ceil(win_end))}s"
        type_summary = "+".join(anomaly.get("sub_types") or [anomaly["type"]])
        logger.info(
            "    [Micro-Window Re-Scan] %s | Window: %s ~ %s (%d candidate sentences)",
            type_summary,
            start_offset,
            end_offset,
            len(win_units),
        )

        try:
            repaired_edl = run_window_inference_fn(
                win_units=win_units,
                win_start=win_start,
                win_end=win_end,
                start_offset=start_offset,
                end_offset=end_offset,
                anomaly=anomaly,
            )
        except Exception as exc:
            logger.warning("    [Micro-Window Skipped] Window %s~%s failed; keeping original clips: %s", start_offset, end_offset, exc)
            continue

        new_clips = repaired_edl.get("final_edl", []) if isinstance(repaired_edl, dict) else []
        valid_new_clips = []
        for c in new_clips:
            if _resolve_clip_sentence_ids(c, win_units):
                c_copy = dict(c)
                c_copy["llm_verified_window"] = True
                valid_new_clips.append(c_copy)
        if not valid_new_clips:
            continue

        replace_range = anomaly.get("replace_range")
        if replace_range is not None:
            r_start, r_end = replace_range
            clips[r_start:r_end] = valid_new_clips
        else:
            clips.extend(valid_new_clips)

        applied_repairs.append({
            "type": type_summary,
            "window": f"{start_offset}~{end_offset}",
            "reason": anomaly["reason"],
            "repaired_clip_count": len(valid_new_clips),
        })

    if applied_repairs:
        clips = deduplicate_and_sort_clips(
            clips=clips,
            whisper_units=whisper_units,
            script_text=script_text,
        )

    updated_edl = dict(model_edl)
    updated_edl["final_edl"] = clips
    return updated_edl, list(reversed(applied_repairs))

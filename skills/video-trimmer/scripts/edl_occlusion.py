"""
Right-to-Left composite span occlusion, retake overlap trimming, short-clause recovery, and clip deduplication.

ASD-STE100:
1. Resolve clip sentence IDs, physical time bounds, and dual transcripts (`transcript` vs `asr_transcript`).
2. Detect opening retakes, tail-to-head stumbles, and intra-clip repetitions.
3. Apply Right-to-Left Span Occlusion across composite pick-up pairs and expanded word-level sub-units.
4. Recover unclaimed adjacent opening or closing short clauses explicitly included in `transcript`.
"""

import difflib
import re

from .chirp3_asr import join_words_for_display
from .script_aligner import (
    _is_phrase_legitimately_repeated_in_script,
    _script_block_coverage_score,
    extract_script_blocks,
    normalize_text,
)
from .transcribe import (
    _build_word_char_timeline,
    _find_best_char_span,
    _slice_words_on_char_span,
    is_earlier_sentence_ng_retake,
    is_viable_sub_span,
)


def _resolve_clip_sentence_ids(clip: dict, whisper_units: list[dict]) -> list[int]:
    """Return sorted valid ASR sentence IDs referenced by a clip."""
    if not whisper_units:
        return []
    valid_unit_ids = {int(u["id"]) for u in whisper_units if "id" in u}
    raw_ids = clip.get("sentence_ids")
    resolved = []
    if isinstance(raw_ids, list) and raw_ids:
        for x in raw_ids:
            if isinstance(x, (int, str)) and str(x).isdigit() and int(x) in valid_unit_ids:
                resolved.append(int(x))
    if not resolved:
        start_id = clip.get("start_sentence_id") or clip.get("start_segment_id")
        end_id = clip.get("end_sentence_id") or clip.get("end_segment_id")
        if isinstance(start_id, int) and isinstance(end_id, int):
            resolved = [uid for uid in sorted(valid_unit_ids) if start_id <= uid <= end_id]
    return sorted(set(resolved))


def _clip_time_bounds(clip: dict, unit_by_id: dict[int, dict], whisper_units: list[dict]) -> tuple[float, float]:
    """
    Return physical `(start, end)` time bounds for a clip.

    ASD-STE100:
    1. Prefer explicit `t_first` and `t_last` sub-span bounds when already resolved.
    2. When a clip trims a sub-span inside `sentence_ids`, estimate sub-span word bounds
       so complementary sub-spans sharing a `Sentence ID` do not falsely collide in time.
    """
    if clip.get("t_first") is not None and clip.get("t_last") is not None:
        tf = float(clip["t_first"])
        tl = max(tf, float(clip["t_last"]))
        return tf, tl

    sids = _resolve_clip_sentence_ids(clip, whisper_units)
    t_in = float(clip.get("source_in", 0.0))
    t_out = float(clip.get("source_out", t_in))
    if sids:
        first_u = unit_by_id.get(sids[0])
        last_u = unit_by_id.get(sids[-1])
        if first_u is not None:
            t_in = float(first_u.get("start", t_in))
        if last_u is not None:
            t_out = float(last_u.get("end", t_out))

        transcript = (clip.get("transcript") or "").strip()
        tgt_norm = normalize_text(transcript)
        if tgt_norm:
            char_times: list[tuple[float, float, str]] = []
            for sid in sids:
                u = unit_by_id.get(sid)
                if not u:
                    continue
                for w in (u.get("words") or []):
                    w_norm = normalize_text(w.get("word", ""))
                    if not w_norm:
                        continue
                    ws = float(w.get("start", 0.0))
                    we = float(w.get("end", ws))
                    wd = max(0.0, we - ws) / len(w_norm)
                    for idx_c, ch in enumerate(w_norm):
                        cs = ws + idx_c * wd
                        char_times.append((cs, cs + wd, ch))
            if char_times and len(tgt_norm) < len(char_times) * 0.90:
                w_str = "".join(item[2] for item in char_times)
                role = str(clip.get("composite_role") or "").strip().lower()
                if role == "prefix":
                    pos = w_str.find(tgt_norm)
                    if pos != -1:
                        return round(char_times[pos][0], 3), round(char_times[pos + len(tgt_norm) - 1][1], 3)
                span = _find_best_char_span(tgt_norm, w_str)
                if span is not None and span[2] >= max(2, int(len(tgt_norm) * 0.65)):
                    return round(char_times[span[0]][0], 3), round(char_times[span[1]][1], 3)

    return t_in, max(t_in, t_out)


def _clip_dual_texts(clip: dict, unit_by_id: dict[int, dict], whisper_units: list[dict]) -> tuple[str, str]:
    """
    Return `(llm_transcript, asr_transcript)` for a clip so anomaly detectors and
    quality auditors can inspect both LLM-cleaned text and physical ASR words.
    """
    llm_text = (clip.get("transcript") or "").strip()
    w_text = (clip.get("asr_transcript") or "").strip()
    if not w_text:
        sids = _resolve_clip_sentence_ids(clip, whisper_units)
        t_first = clip.get("t_first")
        t_last = clip.get("t_last")
        if sids:
            collected_words: list[str] = []
            for sid in sids:
                u = unit_by_id.get(sid)
                if not u:
                    continue
                u_words = u.get("words") or []
                if u_words and t_first is not None and t_last is not None:
                    for w in u_words:
                        w_start = float(w.get("start", 0.0))
                        w_end = float(w.get("end", 0.0))
                        if w_end >= float(t_first) - 0.05 and w_start <= float(t_last) + 0.05:
                            collected_words.append(str(w.get("word", "")))
                else:
                    u_start = float(u.get("start", 0.0))
                    u_end = float(u.get("end", 0.0))
                    if t_first is not None and t_last is not None and (
                        float(t_first) > u_start + 0.20 or float(t_last) < u_end - 0.20
                    ):
                        continue
                    collected_words.append(str(u.get("text", "")).strip())
            if collected_words:
                w_text = join_words_for_display([x for x in collected_words if x])
    if not llm_text:
        llm_text = w_text
    if not w_text:
        w_text = llm_text
    return llm_text, w_text


def _extend_overlap_with_pre_clause_prefix(
    tail_a: str,
    head_b: str,
    match_a_start: int,
    match_a_end: int,
    match_b_start: int,
) -> str:
    """
    Extend a tail-to-head overlap leftward when the pre-overlap words in `tail_a` fuzzy-match
    the opening prefix `head_b[:match_b_start]` (differs only by a 1-to-2 char filler or adverb).
    """
    base_overlap = tail_a[match_a_start:match_a_end]
    if match_b_start <= 0 or match_a_start <= 0:
        return base_overlap

    pre_b = head_b[:match_b_start]
    max_lookback = min(match_a_start, match_b_start + 3)
    pre_a_win = tail_a[match_a_start - max_lookback : match_a_start]
    for idx in range(0, len(pre_a_win)):
        cand_pre = pre_a_win[idx:]
        if len(cand_pre) < 2:
            break
        ratio = difflib.SequenceMatcher(None, cand_pre, pre_b, autojunk=False).ratio()
        if ratio >= 0.60 or (len(pre_b) >= 2 and cand_pre[:2] == pre_b[:2]):
            return tail_a[match_a_start - max_lookback + idx : match_a_end]
    return base_overlap


def _detect_tail_head_overlap(
    text_a: str,
    text_b: str,
    min_exact_chars: int = 6,
    min_stumble_chars: int = 4,
    max_trailing_stumble: int = 2,
    script_text: str | None = None,
    script_blocks: list[dict] | None = None,
    extend_pre_overlap_prefix: bool = False,
) -> str | None:
    """
    Detect when the tail of `text_a` contains an aborted false start that repeats at the
    beginning of `text_b` (`TAIL_HEAD_RETAKE`).

    ASD-STE100:
    1. Check the last 48 normalized characters of `text_a` against the first 48 normalized
       characters of `text_b`.
    2. Require the match in `text_b` to start near the opening (`longest.b <= 6`) so repeated
       objects in the middle of the next sentence are never mistaken for stumbles.
    3. Exempt phrases that legitimately repeat in the reference script.
    """
    norm_a = normalize_text(text_a)
    norm_b = normalize_text(text_b)
    if len(norm_a) < min_stumble_chars or len(norm_b) < min_stumble_chars:
        return None

    tail_a = norm_a[-48:]
    head_b = norm_b[:48]

    # 1. Suffix-to-prefix check with tolerance for 1-2 trailing stumble chars at the end of tail_a
    for stumble_tail in range(0, max_trailing_stumble + 1):
        a_end = len(tail_a) - stumble_tail
        min_k = min_stumble_chars if stumble_tail > 0 else min(min_exact_chars, 5)
        if a_end < min_k:
            continue
        max_lead_b = 0 if stumble_tail > 0 else 2
        for lead_b in range(0, max_lead_b + 1):
            max_k = min(a_end, len(head_b) - lead_b)
            for k in range(max_k, min_k - 1, -1):
                sub_a = tail_a[a_end - k : a_end]
                sub_b = head_b[lead_b : lead_b + k]
                if sub_a == sub_b:
                    if stumble_tail > 0 and (lead_b + k) < len(head_b):
                        if tail_a[a_end : a_end + 1] == head_b[lead_b + k : lead_b + k + 1]:
                            continue
                    if _is_phrase_legitimately_repeated_in_script(sub_a, script_text, script_blocks):
                        continue
                    if extend_pre_overlap_prefix and lead_b > 0:
                        return _extend_overlap_with_pre_clause_prefix(
                            tail_a, head_b, a_end - k, a_end, lead_b
                        )
                    return sub_a

    # 2. Shared clause near the tail-to-head boundary:
    #    - Short matches (< 9 chars) must start near the opening of `text_b` (`longest.b <= 6`).
    #    - Long clause restarts (>= 9 chars) are recognized even after a rephrased lead-in clause.
    matcher = difflib.SequenceMatcher(None, tail_a, head_b, autojunk=False)
    longest = matcher.find_longest_match(0, len(tail_a), 0, len(head_b))
    is_near_tail_a = (longest.a + longest.size) >= (len(tail_a) - 7)
    is_valid_head_pos = (longest.b <= 6) or (longest.size >= 9)
    if is_near_tail_a and is_valid_head_pos and longest.size >= min_exact_chars:
        cand = tail_a[longest.a : longest.a + longest.size]
        if _is_phrase_legitimately_repeated_in_script(cand, script_text, script_blocks):
            return None
        if extend_pre_overlap_prefix and longest.b > 0:
            return _extend_overlap_with_pre_clause_prefix(
                tail_a, head_b, longest.a, longest.a + longest.size, longest.b
            )
        return cand

    return None


def _detect_intra_clip_repeat(
    text: str,
    min_repeat_chars: int = 6,
    max_gap_chars: int = 12,
    script_text: str | None = None,
    script_blocks: list[dict] | None = None,
    include_punctuated_short_stutter: bool = False,
) -> str | None:
    """
    Detect when a single clip contains an immediate internal repetition (`A + A + B`).

    ASD-STE100:
    1. Scan normalized text for substrings >= `min_repeat_chars` repeating within `max_gap_chars`,
       exempting phrases that legitimately repeat in the reference script.
    2. When `include_punctuated_short_stutter=True` (pre-render micro-window radar only), also flag
       short 2-to-5 character clause-opening stutters so Gemini can evaluate them by audio.
    """
    norm = normalize_text(text)
    n = len(norm)
    if n >= min_repeat_chars * 2:
        max_k = min(24, n // 2)
        for k in range(max_k, min_repeat_chars - 1, -1):
            for i in range(0, n - 2 * k + 1):
                sub = norm[i : i + k]
                search_end = min(n, i + 2 * k + max_gap_chars)
                next_pos = norm.find(sub, i + k, search_end)
                if next_pos != -1:
                    if _is_phrase_legitimately_repeated_in_script(sub, script_text, script_blocks):
                        continue
                    return sub

    # 2. Short clause-opening stutter candidate (for LLM micro-window multimodal review only)
    if include_punctuated_short_stutter and text:
        if n >= 6:
            for k in range(min(5, n // 2), 2, -1):
                cand_open = norm[:k]
                if norm[k : 2 * k] == cand_open or (k >= 4 and norm[k : 2 * k - 1] == cand_open[:-1]):
                    if not _is_phrase_legitimately_repeated_in_script(cand_open, script_text, script_blocks):
                        return cand_open

        raw_clauses = [
            c.strip()
            for c in re.split(r"[，,、；;：:…—]+", text)
            if c.strip()
        ]
        for idx in range(len(raw_clauses) - 1):
            c1_raw, c2_raw = raw_clauses[idx], raw_clauses[idx + 1]
            c1 = normalize_text(c1_raw)
            c2 = normalize_text(c2_raw)
            if len(c1) < 2 or len(c2) < 2:
                continue
            if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", c1_raw):
                max_k = min(5, len(c1), len(c2))
                for k in range(max_k, 1, -1):
                    if len(c1) <= k + 2 and c1[:k] == c2[:k]:
                        cand_short = c1[:k]
                        if not _is_phrase_legitimately_repeated_in_script(cand_short, script_text, script_blocks):
                            return cand_short
            else:
                w1 = re.findall(r"[A-Za-z0-9]+", c1_raw.lower())
                w2 = re.findall(r"[A-Za-z0-9]+", c2_raw.lower())
                max_w = min(3, len(w1), len(w2))
                for kw in range(max_w, 0, -1):
                    if len(w1) <= kw + 1 and w1[:kw] == w2[:kw] and len("".join(w1[:kw])) >= 2:
                        cand_short = "".join(w1[:kw])
                        if not _is_phrase_legitimately_repeated_in_script(cand_short, script_text, script_blocks):
                            return cand_short

    return None


def _is_opening_retake_pair(prev_text: str, next_text: str) -> bool:
    """
    Return True only when `prev_text` is an earlier opening false-start or full retake of `next_text`.

    ASD-STE100:
    Evaluate directionally (`prev_text -> next_text`) so a long sentence `"A + B"` followed by
    a partial tail pick-up `"B"` is handled as a tail-to-head overlap instead of dropping `"A + B"`.
    """
    if is_earlier_sentence_ng_retake(prev_text, next_text):
        return True
    na = normalize_text(prev_text)
    nb = normalize_text(next_text)
    if len(na) >= 4 and len(nb) >= 4 and na[:4] == nb[:4] and len(nb) >= int(len(na) * 0.75):
        return True
    return False


def _trim_raw_transcript_to_norm_len(transcript: str, keep_norm_chars: int) -> str:
    """Keep the first `keep_norm_chars` normalized characters of `transcript` and strip trailing seam punctuation."""
    if keep_norm_chars <= 0:
        return ""
    norm_raw_indices = [
        raw_i for raw_i, ch in enumerate(transcript) if re.match(r"[\w一-鿿]", ch)
    ]
    if keep_norm_chars >= len(norm_raw_indices):
        return transcript.strip()
    end_raw = norm_raw_indices[keep_norm_chars]
    return transcript[:end_raw].rstrip(" ，,、；;：:——-　").strip()


def _trim_raw_transcript_head_by_norm_chars(transcript: str, drop_norm_chars: int) -> str:
    """Drop the first `drop_norm_chars` normalized characters of `transcript` and strip leading seam punctuation."""
    if drop_norm_chars <= 0 or not transcript:
        return (transcript or "").strip()
    norm_raw_indices = [
        raw_i for raw_i, ch in enumerate(transcript) if re.match(r"[\w一-鿿]", ch)
    ]
    if drop_norm_chars >= len(norm_raw_indices):
        return ""
    start_raw = norm_raw_indices[drop_norm_chars]
    return transcript[start_raw:].lstrip(" ，,、。！？!?；;：:——-　").strip()


def _is_complementary_composite_pair(
    clip_a: dict,
    clip_b: dict,
    unit_by_id: dict[int, dict],
    whisper_units: list[dict],
    script_blocks: list[dict] | None = None,
) -> bool:
    """
    Return True when `clip_a` (prefix) and `clip_b` (continuation) form a valid non-overlapping
    composite sentence pair that must not be collapsed into a single take.
    """
    role_a = str(clip_a.get("composite_role") or "").strip().lower()
    role_b = str(clip_b.get("composite_role") or "").strip().lower()
    explicit_composite = (
        role_a == "prefix"
        or role_b == "continuation"
        or bool(clip_b.get("is_composite_splice"))
    )

    llm_a, _ = _clip_dual_texts(clip_a, unit_by_id, whisper_units)
    llm_b, _ = _clip_dual_texts(clip_b, unit_by_id, whisper_units)
    norm_a = normalize_text(llm_a)
    norm_b = normalize_text(llm_b)
    if not is_viable_sub_span(llm_a) or not is_viable_sub_span(llm_b):
        return False

    # Reject pairs that still contain opening retake or unexempted tail-head overlap
    if (
        not _is_phrase_legitimately_repeated_in_script(norm_a[:6], script_blocks=script_blocks)
        and _is_opening_retake_pair(llm_a, llm_b)
    ) or _detect_tail_head_overlap(llm_a, llm_b, script_blocks=script_blocks):
        return False

    _, t_a_out = _clip_time_bounds(clip_a, unit_by_id, whisper_units)
    t_b_in, t_b_out = _clip_time_bounds(clip_b, unit_by_id, whisper_units)
    if t_b_in < t_a_out - 0.05 or (t_b_in - t_a_out) > 45.0 or t_b_out <= t_a_out:
        return False

    matcher = difflib.SequenceMatcher(None, norm_a, norm_b, autojunk=False)
    shared = matcher.find_longest_match(0, len(norm_a), 0, len(norm_b))
    if shared.size >= 4:
        shared_str = norm_a[shared.a : shared.a + shared.size]
        if not _is_phrase_legitimately_repeated_in_script(shared_str, script_blocks=script_blocks):
            return False

    if explicit_composite:
        return True

    if script_blocks:
        combined = norm_a + norm_b
        for blk in script_blocks:
            b_norm = blk["norm_text"]
            cov_a = _script_block_coverage_score(b_norm, norm_a)
            cov_b = _script_block_coverage_score(b_norm, norm_b)
            cov_comb = _script_block_coverage_score(b_norm, combined)
            if cov_comb >= max(cov_a, cov_b) + 0.15 and cov_comb >= 0.65 and cov_b < 0.85:
                pos_a = difflib.SequenceMatcher(None, b_norm, norm_a, autojunk=False).find_longest_match(0, len(b_norm), 0, len(norm_a))
                pos_b = difflib.SequenceMatcher(None, b_norm, norm_b, autojunk=False).find_longest_match(0, len(b_norm), 0, len(norm_b))
                if pos_a.size >= 3 and pos_b.size >= 3 and pos_a.a < pos_b.a:
                    return True

    return False


def resolve_composite_prefix_occlusions(
    clips: list[dict],
    whisper_units: list[dict],
    script_text: str | None = None,
) -> list[dict]:
    """
    Apply Right-to-Left Span Occlusion to composite pick-up pairs and Mode A unscripted overlaps.

    ASD-STE100:
    1. Anchor on the latest take (`continuation`) so `Take 2` owns 100% of any overlapping words.
    2. Trim any residual overlapping tail phrase from the preceding `prefix` clip's transcript.
    3. Drop the preceding clip (`Last-Take-Wins`) if the remaining unique prefix fails `is_viable_sub_span`.
    4. Preserve `llm_verified_window` pairs and route unscripted Mode B non-composite overlaps to LLM micro-window review.
    """
    if len(clips) <= 1:
        return [dict(c) for c in clips]

    unit_by_id = {int(u["id"]): u for u in whisper_units if "id" in u}
    script_blocks = extract_script_blocks(script_text) if script_text else []
    working = [dict(c) for c in clips]
    keep_mask = [True] * len(working)

    for i in range(len(working) - 1, 0, -1):
        if not keep_mask[i]:
            continue
        curr = working[i]
        prev_idx = i - 1
        while prev_idx >= 0 and not keep_mask[prev_idx]:
            prev_idx -= 1
        if prev_idx < 0:
            continue
        prev = working[prev_idx]

        t_p_in, t_p_out = _clip_time_bounds(prev, unit_by_id, whisper_units)
        t_c_in, _ = _clip_time_bounds(curr, unit_by_id, whisper_units)
        if (t_c_in - t_p_out) > 45.0 or t_c_in < t_p_in:
            continue

        role_p = str(prev.get("composite_role") or "").strip().lower()
        role_c = str(curr.get("composite_role") or "").strip().lower()
        explicit_composite = (
            role_p == "prefix"
            or role_c == "continuation"
            or bool(curr.get("is_composite_splice"))
        )

        if prev.get("llm_verified_window") and curr.get("llm_verified_window") and not explicit_composite:
            continue

        llm_p, _ = _clip_dual_texts(prev, unit_by_id, whisper_units)
        llm_c, _ = _clip_dual_texts(curr, unit_by_id, whisper_units)
        norm_p = normalize_text(llm_p)
        norm_c = normalize_text(llm_c)
        if not norm_p or not norm_c:
            continue

        # Check if Mode A script block indicates `curr` is a partial tail pick-up of the same primary block as `prev`
        mode_a_tail_pickup = False
        if script_blocks and not explicit_composite:
            best_p_blk = max(
                script_blocks,
                key=lambda b: _script_block_coverage_score(b["norm_text"], norm_p),
            )
            best_c_blk = max(
                script_blocks,
                key=lambda b: _script_block_coverage_score(b["norm_text"], norm_c),
            )
            if best_p_blk["block_id"] == best_c_blk["block_id"]:
                b_norm = best_p_blk["norm_text"]
                cov_p = _script_block_coverage_score(b_norm, norm_p)
                cov_c = _script_block_coverage_score(b_norm, norm_c)
                cov_comb = _script_block_coverage_score(b_norm, norm_p + norm_c)
                if 0.25 <= cov_p < 0.85 and 0.30 <= cov_c < 0.85 and cov_comb >= cov_c + 0.15:
                    mode_a_tail_pickup = True

        is_opening_retake = (
            not _is_phrase_legitimately_repeated_in_script(norm_p[:6], script_text, script_blocks)
            and _is_opening_retake_pair(llm_p, llm_c)
        )
        overlap_str = None
        if not is_opening_retake:
            overlap_str = _detect_tail_head_overlap(
                llm_p,
                llm_c,
                min_exact_chars=4,
                min_stumble_chars=3,
                script_text=script_text,
                script_blocks=script_blocks,
                extend_pre_overlap_prefix=True,
            )

        mode_a_unscripted_retake = bool(script_blocks) and (is_opening_retake or bool(overlap_str))
        if not explicit_composite and not mode_a_tail_pickup and not mode_a_unscripted_retake:
            continue

        # Full retake check: if `curr` restarts from the beginning of `prev`, `Last-Take-Wins` drops `prev`
        if is_opening_retake:
            keep_mask[prev_idx] = False
            curr["composite_role"] = "none"
            curr["is_composite_splice"] = False
            continue

        # Right-to-Left occlusion of any overlapping tail on `prev`
        if overlap_str:
            pos_in_tail = norm_p.rfind(overlap_str, max(0, len(norm_p) - 52))
            if pos_in_tail != -1:
                trimmed_tr = _trim_raw_transcript_to_norm_len(llm_p, pos_in_tail)
                if not is_viable_sub_span(trimmed_tr):
                    keep_mask[prev_idx] = False
                    curr["composite_role"] = "none"
                    curr["is_composite_splice"] = False
                    continue
                prev["transcript"] = trimmed_tr
                prev["trimmed_tail"] = True
                prev.pop("asr_transcript", None)
                prev.pop("t_last", None)

        # Verify remaining prefix viability
        final_p_tr = (prev.get("transcript") or "").strip()
        est_p_in, est_p_out = _clip_time_bounds(prev, unit_by_id, whisper_units)
        est_dur = est_p_out - est_p_in if est_p_out > est_p_in else None
        if not is_viable_sub_span(final_p_tr, est_dur):
            keep_mask[prev_idx] = False
            curr["composite_role"] = "none"
            curr["is_composite_splice"] = False
            continue

        if explicit_composite or mode_a_tail_pickup or overlap_str:
            prev["composite_role"] = "prefix"
            prev["trimmed_tail"] = True
            curr["composite_role"] = "continuation"
            curr["is_composite_splice"] = True
            curr["trimmed_head"] = bool(curr.get("trimmed_head", False))

    result = [c for idx, c in enumerate(working) if keep_mask[idx]]
    for idx, c in enumerate(result, start=1):
        c["clip_id"] = idx
    return result


def occlude_expanded_sub_units(
    filtered_units: list[dict],
    script_text: str | None = None,
) -> list[dict]:
    """
    Apply word-level Right-to-Left occlusion and immediate stutter trimming across expanded sub-units.

    ASD-STE100:
    1. Trim unscripted immediate opening stutters (`A + A + B`) using character-timeline word slicing.
    2. Trim unverified tail-to-head overlaps across expanded sub-units and lock `next_word_hard_wall`.
    3. Preserve `llm_verified_window` pairs in Mode B so intentional rhetorical parallelism stays intact.
    """
    if not filtered_units:
        return []

    script_blocks = extract_script_blocks(script_text) if script_text else []
    working = [dict(su) for su in filtered_units]

    # Pass 1: Trim immediate opening stutters (`A_stutter + A_clean + B`) on unverified or Mode A sub-units
    for su in working:
        if su.get("llm_verified_window") and not script_blocks:
            continue
        su_words = list(su.get("words") or [])
        if not su_words:
            continue
        dummy_sent = {"id": 0, "start": su["t_first"], "end": su["t_last"], "words": su_words}
        tl = _build_word_char_timeline([dummy_sent])
        s_norm = "".join(item["ch"] for item in tl)
        if len(s_norm) < 6:
            continue

        trim_head_k = 0
        for k in range(min(15, len(s_norm) // 2), 2, -1):
            pref = s_norm[:k]
            is_rep = (s_norm[k : 2 * k] == pref) or (k >= 4 and s_norm[k : 2 * k - 1] == pref[:-1])
            if is_rep and not _is_phrase_legitimately_repeated_in_script(pref, script_text, script_blocks):
                trim_head_k = k
                break

        if trim_head_k > 0 and trim_head_k < len(tl):
            sliced_list, _, extra_b = _slice_words_on_char_span([dummy_sent], tl, trim_head_k, len(tl) - 1)
            if sliced_list and sliced_list[0].get("words"):
                kept_w = sliced_list[0]["words"]
                new_w_text = sliced_list[0]["text"]
                excised_end = (
                    max(b_end for _, b_end in extra_b)
                    if extra_b
                    else float(tl[trim_head_k - 1]["w_end"])
                )
                su["words"] = kept_w
                su["t_first"] = round(float(kept_w[0]["start"]), 3)
                prev_end = su.get("prev_sentence_end")
                su["prev_sentence_end"] = (
                    round(max(float(prev_end), excised_end), 3)
                    if prev_end is not None
                    else round(excised_end, 3)
                )
                su["trimmed_head"] = True
                su["asr_transcript"] = new_w_text
                tr_raw = su.get("transcript") or ""
                tr_norm = normalize_text(tr_raw)
                if len(tr_norm) >= 2 * trim_head_k and (
                    tr_norm[:trim_head_k] == s_norm[:trim_head_k]
                ) and (
                    tr_norm[trim_head_k : 2 * trim_head_k] == tr_norm[:trim_head_k]
                    or (trim_head_k >= 4 and tr_norm[trim_head_k : 2 * trim_head_k - 1] == tr_norm[: trim_head_k - 1])
                ):
                    su["transcript"] = _trim_raw_transcript_head_by_norm_chars(tr_raw, trim_head_k) or new_w_text
                elif not tr_raw:
                    su["transcript"] = new_w_text

    if len(working) <= 1:
        return working

    # Pass 2: Right-to-Left retake and tail-to-head overlap occlusion across adjacent sub-units
    keep_mask = [True] * len(working)
    closure_chars = ("。", "！", "？", "!", "?")

    for i in range(len(working) - 1, 0, -1):
        if not keep_mask[i]:
            continue
        curr_su = working[i]

        while True:
            prev_idx = i - 1
            while prev_idx >= 0 and not keep_mask[prev_idx]:
                prev_idx -= 1
            if prev_idx < 0:
                break
            prev_su = working[prev_idx]

            gap = float(curr_su.get("t_first", 0.0)) - float(prev_su.get("t_last", 0.0))
            if gap > 25.0 or gap < -0.5:
                break

            explicit_composite = (
                prev_su.get("composite_role") == "prefix"
                or curr_su.get("composite_role") == "continuation"
                or bool(curr_su.get("is_composite_splice"))
            )
            p_parent = prev_su.get("parent_clip")
            c_parent = curr_su.get("parent_clip")
            same_parent = p_parent is not None and c_parent is not None and (
                p_parent is c_parent
                or (p_parent.get("clip_id") is not None and p_parent.get("clip_id") == c_parent.get("clip_id"))
            )
            pair_verified = bool(prev_su.get("llm_verified_window") and curr_su.get("llm_verified_window"))

            if pair_verified and not explicit_composite and not same_parent and not script_blocks:
                break

            p_text = prev_su.get("asr_transcript") or prev_su.get("transcript") or ""
            c_text = curr_su.get("asr_transcript") or curr_su.get("transcript") or ""
            p_norm = normalize_text(p_text)
            c_norm = normalize_text(c_text)
            if not p_norm or not c_norm:
                break

            can_drop_opening = bool(script_blocks) or explicit_composite or same_parent
            if (
                can_drop_opening
                and not _is_phrase_legitimately_repeated_in_script(p_norm[:6], script_text, script_blocks)
                and _is_opening_retake_pair(p_text, c_text)
            ):
                keep_mask[prev_idx] = False
                curr_su["trimmed_head"] = True
                continue

            ov = _detect_tail_head_overlap(
                p_text,
                c_text,
                min_exact_chars=4,
                min_stumble_chars=3,
                script_text=script_text,
                script_blocks=script_blocks,
                extend_pre_overlap_prefix=True,
            )
            if not ov:
                break

            if same_parent and not script_blocks:
                parent_tr = normalize_text((p_parent or {}).get("transcript") or "")
                if parent_tr and parent_tr.count(ov) >= 2:
                    break

            pos_in_tail = p_norm.rfind(ov, max(0, len(p_norm) - 52))
            if pos_in_tail == -1:
                break
            if pos_in_tail <= 0:
                keep_mask[prev_idx] = False
                curr_su["trimmed_head"] = True
                continue

            p_words = list(prev_su.get("words") or [])
            if not p_words:
                break

            dummy_p = {"id": 0, "start": prev_su["t_first"], "end": prev_su["t_last"], "words": p_words}
            p_tl = _build_word_char_timeline([dummy_p])
            if pos_in_tail >= len(p_tl):
                break

            sliced_p, _, extra_p = _slice_words_on_char_span([dummy_p], p_tl, 0, pos_in_tail - 1)
            if not sliced_p or not sliced_p[0].get("words"):
                keep_mask[prev_idx] = False
                curr_su["trimmed_head"] = True
                continue

            keep_w = sliced_p[0]["words"]
            new_text = sliced_p[0]["text"]
            new_t_last = round(float(keep_w[-1]["end"]), 3)
            new_dur = new_t_last - float(keep_w[0]["start"])
            if not is_viable_sub_span(new_text, new_dur):
                keep_mask[prev_idx] = False
                curr_su["trimmed_head"] = True
                continue

            cut_candidates = [b_s for b_s, _ in extra_p if b_s >= new_t_last - 0.005]
            cut_word_start = (
                round(min(cut_candidates), 3)
                if cut_candidates
                else round(float(p_tl[pos_in_tail]["w_start"]), 3)
            )
            hard_wall = max(new_t_last, round(cut_word_start - 0.001, 3))

            prev_su["words"] = keep_w
            prev_su["t_last"] = new_t_last
            prev_next = prev_su.get("next_sentence_start")
            prev_su["next_sentence_start"] = (
                round(min(float(prev_next), cut_word_start), 3)
                if prev_next is not None
                else cut_word_start
            )
            prev_wall = prev_su.get("next_word_hard_wall")
            prev_su["next_word_hard_wall"] = (
                round(min(float(prev_wall), hard_wall), 3)
                if prev_wall is not None
                else hard_wall
            )
            prev_su["trimmed_tail"] = True
            prev_su["asr_transcript"] = new_text

            raw_p_tr = prev_su.get("transcript") or ""
            raw_c_tr = curr_su.get("transcript") or c_text
            ov_tr = _detect_tail_head_overlap(
                raw_p_tr,
                raw_c_tr,
                min_exact_chars=4,
                min_stumble_chars=3,
                script_text=script_text,
                script_blocks=script_blocks,
                extend_pre_overlap_prefix=True,
            )
            raw_p_norm = normalize_text(raw_p_tr)
            if ov_tr and raw_p_norm:
                tr_pos = raw_p_norm.rfind(ov_tr, max(0, len(raw_p_norm) - 52))
                if tr_pos > 0:
                    prev_su["transcript"] = _trim_raw_transcript_to_norm_len(raw_p_tr, tr_pos) or new_text
                else:
                    prev_su["transcript"] = new_text
            elif len(raw_p_norm) > pos_in_tail:
                prev_su["transcript"] = _trim_raw_transcript_to_norm_len(raw_p_tr, pos_in_tail) or new_text
            else:
                prev_su["transcript"] = raw_p_tr or new_text

            if explicit_composite or not any(str(keep_w[-1].get("word", "")).strip().endswith(p) for p in closure_chars):
                prev_su["composite_role"] = "prefix"
                curr_su["composite_role"] = "continuation"
                curr_su["is_composite_splice"] = True
            curr_su["trimmed_head"] = True
            break

    return [su for idx, su in enumerate(working) if keep_mask[idx]]


def _match_omitted_short_clause(cand_norm: str, missing_candidates: list[str]) -> bool:
    """
    Return True when normalized adjacent short-clause text matches an omitted transcript edge.

    ASD-STE100:
    1. Require 2-to-12 characters for CJK/Kana/Hangul clauses or 4-to-30 characters for Latin text.
    2. Require tight length agreement with the adjacent physical sentence to reject longer NG takes.
    3. Require >= 0.70 sequence similarity and at least 2 CJK (or 4 Latin) matched characters.
    """
    if not cand_norm:
        return False
    for missing in missing_candidates:
        if not missing:
            continue
        is_cjk = any("\u3040" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af" for ch in missing)
        if is_cjk:
            if not (2 <= len(missing) <= 12):
                continue
            if abs(len(cand_norm) - len(missing)) > 2:
                continue
            min_matched = min(len(missing), 2)
        else:
            if not (4 <= len(missing) <= 30):
                continue
            if abs(len(cand_norm) - len(missing)) > 5:
                continue
            min_matched = min(len(missing), 4)

        sm = difflib.SequenceMatcher(None, missing, cand_norm, autojunk=False)
        matched_chars = sum(b.size for b in sm.get_matching_blocks())
        if sm.ratio() >= 0.70 and matched_chars >= min_matched:
            return True
    return False


def _extend_omitted_adjacent_short_clauses(
    clips: list[dict],
    whisper_units: list[dict],
) -> list[dict]:
    """
    Extend `sentence_ids` by one unclaimed adjacent short clause when `transcript` explicitly includes it.

    ASD-STE100:
    1. Compare LLM `transcript` against the physical ASR text of `sentence_ids`.
    2. Prepend `sids[0] - 1` or append `sids[-1] + 1` only when unclaimed, target-speaker,
       within a 0.50s pause, <= 2.50s duration, and matching the omitted edge text (>= 0.70).
    3. Never cross composite splice seams, trimmed edges, or neighboring clip time bounds.
    """
    if not clips or not whisper_units:
        return clips

    unit_by_id = {int(u["id"]): u for u in whisper_units if "id" in u}
    if not unit_by_id:
        return clips

    claimed_ids: set[int] = set()
    for c in clips:
        claimed_ids.update(_resolve_clip_sentence_ids(c, whisper_units))

    for idx, c in enumerate(clips):
        sids = _resolve_clip_sentence_ids(c, whisper_units)
        if not sids or sids[0] not in unit_by_id or sids[-1] not in unit_by_id:
            continue

        llm_text = (c.get("transcript") or "").strip()
        t_norm = normalize_text(llm_text)
        if len(t_norm) < 4:
            continue

        _, w_text = _clip_dual_texts(c, unit_by_id, whisper_units)
        u_norm = normalize_text(w_text)
        if not u_norm:
            continue

        blocks = [
            b
            for b in difflib.SequenceMatcher(None, t_norm, u_norm, autojunk=False).get_matching_blocks()
            if b.size >= 2
        ]
        if not blocks or sum(b.size for b in blocks) < max(3, int(min(len(t_norm), len(u_norm)) * 0.50)):
            continue

        role = str(c.get("composite_role") or "none").strip().lower()
        is_splice = bool(c.get("is_composite_splice", False))

        # 1. Check omitted leading short clause (sids[0] - 1)
        first_blk = blocks[0]
        if not is_splice and role != "continuation" and not c.get("trimmed_head") and first_blk.b <= 2 and first_blk.a >= 2:
            prev_sid = sids[0] - 1
            prev_u = unit_by_id.get(prev_sid)
            first_u = unit_by_id.get(sids[0])
            if prev_sid not in claimed_ids and prev_u is not None and first_u is not None and prev_u.get("is_target_speaker", True):
                prev_start = float(prev_u.get("start", 0.0))
                prev_end = float(prev_u.get("end", prev_start))
                first_start = float(first_u.get("start", prev_end))
                gap_head = first_start - prev_end
                dur_head = prev_end - prev_start
                prev_clip_out = (
                    _clip_time_bounds(clips[idx - 1], unit_by_id, whisper_units)[1]
                    if idx > 0
                    else -1.0
                )
                if -0.02 <= gap_head <= 0.50 and 0.05 <= dur_head <= 2.50 and prev_start >= prev_clip_out - 0.02:
                    head_cands = [t_norm[:first_blk.a]]
                    if 1 <= first_blk.b <= 2 and first_blk.a > first_blk.b:
                        head_cands.append(t_norm[: first_blk.a - first_blk.b])
                    cand_norm = normalize_text(str(prev_u.get("text", "")))
                    if _match_omitted_short_clause(cand_norm, head_cands):
                        sids = [prev_sid] + sids
                        claimed_ids.add(prev_sid)
                        c["sentence_ids"] = sids
                        if c.get("start_sentence_id") is not None:
                            c["start_sentence_id"] = sids[0]
                        if c.get("t_first") is not None:
                            c["t_first"] = prev_start
                        if c.get("source_in") is not None and prev_start < float(c["source_in"]):
                            c["source_in"] = prev_start
                        c.pop("asr_transcript", None)

                        # Recompute alignment blocks before evaluating tail extension
                        _, w_text = _clip_dual_texts(c, unit_by_id, whisper_units)
                        u_norm = normalize_text(w_text)
                        blocks = [
                            b
                            for b in difflib.SequenceMatcher(None, t_norm, u_norm, autojunk=False).get_matching_blocks()
                            if b.size >= 2
                        ]

        # 2. Check omitted trailing short clause (sids[-1] + 1)
        if blocks and role != "prefix" and not c.get("trimmed_tail"):
            last_blk = blocks[-1]
            rem_u = len(u_norm) - (last_blk.b + last_blk.size)
            rem_t = len(t_norm) - (last_blk.a + last_blk.size)
            if rem_u <= 2 and rem_t >= 2:
                next_sid = sids[-1] + 1
                next_u = unit_by_id.get(next_sid)
                last_u = unit_by_id.get(sids[-1])
                if next_sid not in claimed_ids and next_u is not None and last_u is not None and next_u.get("is_target_speaker", True):
                    last_end = float(last_u.get("end", 0.0))
                    next_start = float(next_u.get("start", last_end))
                    next_end = float(next_u.get("end", next_start))
                    gap_tail = next_start - last_end
                    dur_tail = next_end - next_start
                    next_clip_in = (
                        _clip_time_bounds(clips[idx + 1], unit_by_id, whisper_units)[0]
                        if idx + 1 < len(clips)
                        else float("inf")
                    )
                    if -0.02 <= gap_tail <= 0.50 and 0.05 <= dur_tail <= 2.50 and next_end <= next_clip_in + 0.02:
                        tail_cands = [t_norm[last_blk.a + last_blk.size :]]
                        if 1 <= rem_u <= 2 and rem_t > rem_u:
                            tail_cands.append(t_norm[last_blk.a + last_blk.size + rem_u :])
                        cand_norm = normalize_text(str(next_u.get("text", "")))
                        if _match_omitted_short_clause(cand_norm, tail_cands):
                            sids = sids + [next_sid]
                            claimed_ids.add(next_sid)
                            c["sentence_ids"] = sids
                            if c.get("end_sentence_id") is not None:
                                c["end_sentence_id"] = sids[-1]
                            if c.get("t_last") is not None:
                                c["t_last"] = next_end
                            if c.get("source_out") is not None and next_end > float(c["source_out"]):
                                c["source_out"] = next_end
                            c.pop("asr_transcript", None)

    return clips


def deduplicate_and_sort_clips(
    clips: list[dict],
    whisper_units: list[dict],
    script_text: str | None = None,
) -> list[dict]:
    """
    Sort clips chronologically, apply Right-to-Left composite prefix occlusion, and prune
    deterministic physical timeline overlaps or verbatim full-block duplicates.

    ASD-STE100:
    1. Sort all clips by physical start time and occlude overlapping composite pick-up prefixes.
    2. Retain valid non-overlapping `prefix -> continuation` sub-span pairs.
    3. Deterministically drop an earlier clip only on physical time overlap or a near-verbatim
       (>= 85% block coverage and >= 85% similarity) full-block duplicate; route ambiguous
       semantic overlaps to `repair_edl_micro_windows` for LLM arbitration.
    4. Extend `sentence_ids` to include an unclaimed adjacent short clause when `transcript`
       explicitly contains that opening or closing clause.
    """
    if not clips:
        return []

    unit_by_id = {int(u["id"]): u for u in whisper_units if "id" in u}

    def _sort_key(c: dict) -> tuple[float, float]:
        t_in, t_out = _clip_time_bounds(c, unit_by_id, whisper_units)
        return (t_in, t_out)

    sorted_clips = sorted((dict(c) for c in clips), key=_sort_key)
    sorted_clips = resolve_composite_prefix_occlusions(
        clips=sorted_clips,
        whisper_units=whisper_units,
        script_text=script_text,
    )
    script_blocks = extract_script_blocks(script_text) if script_text else []

    def _matched_script_block_info(c: dict) -> tuple[int | None, float, str]:
        if not script_blocks:
            return None, 0.0, ""
        llm_text, w_text = _clip_dual_texts(c, unit_by_id, whisper_units)
        llm_norm = normalize_text(llm_text)
        w_norm = normalize_text(w_text)
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
                if score >= 0.45 and score > best_score and len(norm) <= len(blk["norm_text"]) * 1.85:
                    best_score = score
                    best_blk_id = blk["block_id"]
                    best_norm = norm
        return best_blk_id, best_score, best_norm

    deduped: list[dict] = []
    for curr in sorted_clips:
        curr_sids = set(_resolve_clip_sentence_ids(curr, whisper_units))
        curr_in, curr_out = _clip_time_bounds(curr, unit_by_id, whisper_units)
        curr_blk_id, curr_score, curr_norm = _matched_script_block_info(curr)

        kept_earlier: list[dict] = []
        for idx_p, prev in enumerate(deduped):
            if idx_p == len(deduped) - 1 and _is_complementary_composite_pair(
                prev, curr, unit_by_id, whisper_units, script_blocks
            ):
                kept_earlier.append(prev)
                continue

            prev_sids = set(_resolve_clip_sentence_ids(prev, whisper_units))
            prev_in, prev_out = _clip_time_bounds(prev, unit_by_id, whisper_units)
            prev_blk_id, prev_score, prev_norm = _matched_script_block_info(prev)

            time_overlaps = (curr_in < prev_out - 0.05) and (curr_out > prev_in + 0.05)
            shares_sids = (
                bool(curr_sids and prev_sids and curr_sids.intersection(prev_sids))
                and time_overlaps
            )
            near_verbatim_same_block = (
                curr_blk_id is not None
                and curr_blk_id == prev_blk_id
                and curr_score >= 0.85
                and prev_score >= 0.85
                and bool(prev_norm and curr_norm)
                and difflib.SequenceMatcher(None, prev_norm, curr_norm, autojunk=False).ratio() >= 0.85
                and not prev.get("llm_verified_window")
            )

            if shares_sids or time_overlaps or near_verbatim_same_block:
                continue
            kept_earlier.append(prev)

        kept_earlier.append(curr)
        deduped = kept_earlier

    deduped = _extend_omitted_adjacent_short_clauses(deduped, whisper_units)
    for idx, c in enumerate(deduped, start=1):
        c["clip_id"] = idx
    return deduped

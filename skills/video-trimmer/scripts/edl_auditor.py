"""
EDL quality auditor, deterministic timing sanitizer, and surgical micro-window repair.

ASD-STE100:
1. Detect missing script blocks, unanchored clips, and adjacent retake candidates before rendering.
2. Re-scan only the affected 15-to-90-second video micro-window via Vertex AI VideoMetadata.
3. Sanitize physical clip boundaries to eliminate overlaps and micro-clips (< 0.45s).
4. Generate an 8-dimension rough-cut audit report (.json and .md) with a top-level agent_verdict.
"""

import difflib
import logging
import math
import re
from pathlib import Path

from .acoustic import calculate_clip_cps
from .chirp3_asr import join_words_for_display
from .transcribe import (
    _build_word_char_timeline,
    _find_best_char_span,
    _slice_words_on_char_span,
    is_earlier_sentence_ng_retake,
    is_viable_sub_span,
    normalize_text,
)

logger = logging.getLogger(__name__)

MAX_MICRO_WINDOW_REPAIRS = 5
MIN_MICRO_WINDOW_SEC = 15.0
MAX_MICRO_WINDOW_SEC = 90.0


# Document metadata prefixes that represent non-spoken headers (entire line is skipped)
SCRIPT_METADATA_LINE_PREFIXES = (
    "標題：", "標題:", "主題：", "主題:", "大綱：", "大綱:",
    "備註：", "備註:", "說明：", "說明:", "專案：", "專案:",
    "title:", "subject:", "topic:", "outline:", "author:",
    "date:", "duration:", "scene:", "note:", "notes:",
)

# Section label prefixes where the prefix itself is non-spoken, but spoken text may follow on the same line
SCRIPT_SECTION_LABEL_PREFIXES = (
    "內文：", "內文:", "講稿：", "講稿:", "腳本：", "腳本:",
    "台詞：", "台詞:", "旁白：", "旁白:", "口白：", "口白:",
    "body:", "script:", "narration:", "voiceover:", "vo:",
)


def extract_script_blocks(script_text: str | None) -> list[dict]:
    """
    Parse raw script text into sequential numbered script block dicts.

    ASD-STE100:
    1. Strip YAML frontmatter (--- ... ---), markdown headings (#), and horizontal rules.
    2. Exclude non-spoken metadata lines (such as Title, Subject, Outline, Notes).
    3. Strip section label prefixes (such as Body: or Script:) and stage directions.
    4. Extract numbered spoken script blocks for Prompt and Auditor alignment.
    """
    if not script_text or not script_text.strip():
        return []

    raw_lines = script_text.splitlines()
    blocks: list[dict] = []
    block_idx = 1
    in_frontmatter = False

    for line_num, raw_line in enumerate(raw_lines):
        stripped = raw_line.strip()
        if line_num == 0 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped in ("---", "..."):
                in_frontmatter = False
            continue

        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^[-=*_]{3,}$", stripped):
            continue

        # Remove leading markdown list/quote markers before checking metadata prefixes
        unmarked = re.sub(r"^\s*[>*\-\d.]+\s*", "", stripped).strip()
        unmarked_lower = unmarked.lower()

        if any(unmarked_lower.startswith(p) for p in SCRIPT_METADATA_LINE_PREFIXES):
            continue

        for label_prefix in SCRIPT_SECTION_LABEL_PREFIXES:
            if unmarked_lower.startswith(label_prefix):
                unmarked = unmarked[len(label_prefix):].strip()
                break

        if not unmarked:
            continue

        # Strip inline stage directions / camera cues in brackets
        cleaned = re.sub(r"（.*?）|\(.*?\)|【.*?】|\[.*?\]", "", unmarked).strip()
        norm = normalize_text(cleaned)
        if len(norm) >= 4:
            blocks.append({
                "block_id": block_idx,
                "label": f"[Script Block {block_idx:02d}]",
                "raw_text": unmarked,
                "norm_text": norm,
            })
            block_idx += 1

    return blocks


def _script_block_coverage_score(block_norm: str, target_norm: str) -> float:
    """
    Compute the fraction of `block_norm` characters covered in order within `target_norm`.

    ASD-STE100:
    Always use `len(block_norm)` as the denominator so short aborted NG fragments
    cannot produce high similarity scores, while complete takes with minor ASR homophone
    differences achieve high coverage.
    """
    if not block_norm or not target_norm:
        return 0.0
    if block_norm in target_norm:
        return 1.0

    block_len = len(block_norm)
    target_len = len(target_norm)
    min_match_size = 2 if any("\u4e00" <= ch <= "\u9fff" for ch in block_norm) else 3

    def _window_coverage(win_text: str) -> float:
        if not win_text:
            return 0.0
        if block_norm in win_text:
            return 1.0
        matcher = difflib.SequenceMatcher(None, block_norm, win_text, autojunk=False)
        matched_chars = sum(
            m.size for m in matcher.get_matching_blocks() if m.size >= min_match_size
        )
        return min(1.0, matched_chars / block_len)

    if target_len <= block_len * 2:
        return _window_coverage(target_norm)

    win_size = min(target_len, max(block_len + 24, int(block_len * 1.8)))
    stride = max(4, block_len // 3)
    best_score = 0.0

    # Evaluate sliding windows across target_norm
    for start_idx in range(0, max(1, target_len - win_size + 1), stride):
        score = _window_coverage(target_norm[start_idx : start_idx + win_size])
        if score > best_score:
            best_score = score
            if best_score >= 0.95:
                return best_score

    # Evaluate tail window and anchor window around longest contiguous match
    tail_score = _window_coverage(target_norm[max(0, target_len - win_size) :])
    best_score = max(best_score, tail_score)

    global_matcher = difflib.SequenceMatcher(None, block_norm, target_norm, autojunk=False)
    longest = global_matcher.find_longest_match(0, block_len, 0, target_len)
    if longest.size >= min_match_size:
        anchor_start = max(0, longest.b - longest.a - 12)
        anchor_end = min(target_len, anchor_start + win_size)
        best_score = max(best_score, _window_coverage(target_norm[anchor_start:anchor_end]))

    return best_score


def _unit_overlaps_script_block(block_norm: str, unit_norm: str) -> bool:
    """
    Return True if a single Whisper sentence unit belongs to a script block.

    ASD-STE100:
    Accept a sentence unit when it covers >= 35% of the script block or when >= 65%
    of the sentence unit (at least 5 chars) matches a contiguous clause of the script block.
    """
    if not block_norm or not unit_norm:
        return False
    if _script_block_coverage_score(block_norm, unit_norm) >= 0.35:
        return True
    if len(unit_norm) >= 5 and _script_block_coverage_score(unit_norm, block_norm) >= 0.65:
        return True
    return False


def _resolve_clip_sentence_ids(clip: dict, whisper_units: list[dict]) -> list[int]:
    """Return sorted valid Whisper sentence IDs referenced by a clip."""
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


def _count_fuzzy_script_occurrences(
    phrase_norm: str,
    script_norm: str,
    max_substitutions: int | None = None,
) -> int:
    """
    Count non-overlapping occurrences of `phrase_norm` in `script_norm` while tolerating
    minor ASR homophone substitutions (1 char for phrases >= 5 chars, 2 chars for >= 10 chars).
    """
    p_len = len(phrase_norm)
    if p_len == 0 or len(script_norm) < p_len:
        return 0
    if max_substitutions is None:
        if p_len >= 10:
            max_sub = 2
        elif p_len >= 5:
            max_sub = 1
        else:
            max_sub = 0
    else:
        max_sub = max(0, max_substitutions)

    if max_sub == 0:
        return script_norm.count(phrase_norm)

    count = 0
    pos = 0
    limit = len(script_norm) - p_len
    while pos <= limit:
        window = script_norm[pos : pos + p_len]
        if window == phrase_norm:
            count += 1
            pos += p_len
            continue
        mismatches = 0
        for ch_a, ch_b in zip(phrase_norm, window):
            if ch_a != ch_b:
                mismatches += 1
                if mismatches > max_sub:
                    break
        if mismatches <= max_sub:
            count += 1
            pos += p_len
        else:
            pos += 1
    return count


def _is_phrase_legitimately_repeated_in_script(
    phrase_norm: str,
    script_text: str | None = None,
    script_blocks: list[dict] | None = None,
    max_gap_chars: int = 36,
) -> bool:
    """
    Return True when `phrase_norm` genuinely repeats within `max_gap_chars` in the reference script.

    ASD-STE100:
    Exempt intentional scripted parallelisms and anadiplosis (such as repeating a key phrase across
    adjacent clauses) from false-positive retake or overlap detection.
    """
    if not phrase_norm or len(phrase_norm) < 3:
        return False
    blocks = script_blocks if script_blocks is not None else (extract_script_blocks(script_text) if script_text else [])
    if not blocks:
        return False

    full_norm = "".join(b.get("norm_text", "") for b in blocks)
    search_corpora = [b.get("norm_text", "") for b in blocks if b.get("norm_text")] + ([full_norm] if full_norm else [])

    def _has_close_repeat(sub: str) -> bool:
        sub_len = len(sub)
        for corpus in search_corpora:
            if len(corpus) < sub_len * 2:
                continue
            pos1 = corpus.find(sub)
            while pos1 != -1:
                pos2 = corpus.find(sub, pos1 + sub_len, pos1 + 2 * sub_len + max_gap_chars)
                if pos2 != -1:
                    return True
                pos1 = corpus.find(sub, pos1 + 1)
        return False

    if _has_close_repeat(phrase_norm):
        return True

    # Tolerate 1-char homophone drift (such as "信號" vs "訊號") on phrases >= 5 chars
    if len(phrase_norm) >= 5:
        if _has_close_repeat(phrase_norm[1:]) or _has_close_repeat(phrase_norm[:-1]):
            return True
        if full_norm and len(full_norm) >= len(phrase_norm) * 2:
            sub_len = len(phrase_norm)
            match_positions: list[int] = []
            for pos in range(len(full_norm) - sub_len + 1):
                win = full_norm[pos : pos + sub_len]
                mismatches = 0
                for ca, cb in zip(phrase_norm, win):
                    if ca != cb:
                        mismatches += 1
                        if mismatches > 1:
                            break
                if mismatches <= 1 and (not match_positions or pos >= match_positions[-1] + sub_len):
                    if match_positions and (pos - match_positions[-1]) <= sub_len + max_gap_chars:
                        return True
                    match_positions.append(pos)

    return False


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
                    # Remaining unique prefix is too short (< 4 CJK chars / < 3 Latin words): drop `prev`
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

            # Case 5: Completely missed script block in unselected Whisper clusters
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

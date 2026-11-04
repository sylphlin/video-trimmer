"""
Script block parser, fuzzy coverage scorer, and legitimate script repetition whitelister.

ASD-STE100:
1. Normalize text strings for punctuation-insensitive alignment across locales.
2. Parse raw reference scripts into sequential numbered `[Script Block NN]` dicts.
3. Compute directional script block coverage scores and whitelist intentional script repetitions.
"""

import difflib
import re

# Document metadata prefixes that represent non-spoken headers (skip the entire line).
SCRIPT_METADATA_LINE_PREFIXES = (
    "標題：", "標題:", "主題：", "主題:", "大綱：", "大綱:",
    "備註：", "備註:", "說明：", "說明:", "專案：", "專案:",
    "title:", "subject:", "topic:", "outline:", "author:",
    "date:", "duration:", "scene:", "note:", "notes:",
)

# Section label prefixes where the prefix is non-spoken, while spoken text can follow on the same line.
SCRIPT_SECTION_LABEL_PREFIXES = (
    "內文：", "內文:", "講稿：", "講稿:", "腳本：", "腳本:",
    "台詞：", "台詞:", "旁白：", "旁白:", "口白：", "口白:",
    "body:", "script:", "narration:", "voiceover:", "vo:",
)


def normalize_text(text: str) -> str:
    """Strip punctuation and whitespace, and convert text to lowercase for alignment."""
    return re.sub(r"[^\w一-鿿]", "", text or "").lower()


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

        # Remove leading markdown list or quote markers before checking metadata prefixes
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

        # Strip inline stage directions and camera cues in brackets
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
    Return True if a single ASR sentence unit belongs to a script block.

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

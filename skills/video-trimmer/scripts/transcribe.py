"""Dual-engine word-level transcription (Chirp 3 default, Whisper optional), acoustic clause segmentation, sub-span trimming, and cross-clip coalescing.

ASD-STE100:
1. Extract physical word timestamps via Google Cloud STT V2 Chirp 3 (default) or local Whisper (optional).
2. Segment words into Sentence IDs purely on physical breath pauses, stretched onsets, and punctuation closure.
3. Align and slice sub-spans to the LLM-selected transcript (including intra-sentence stumbles) and coalesce contiguous clips.
"""

import difflib
import json
import logging
import math
import re
import subprocess
from pathlib import Path

from .chirp3_asr import is_cjk_char, join_words_for_display, run_chirp3_transcription
from .constants import (
    COMPOSITE_MIN_CJK_CHARS,
    COMPOSITE_MIN_LATIN_CHARS,
    COMPOSITE_MIN_LATIN_WORDS,
    COMPOSITE_MIN_PREFIX_SEC,
    DEFAULT_ASR_ENGINE,
    DEFAULT_CHIRP3_MODEL,
    DEFAULT_STT_LOCATION,
    DEFAULT_WHISPER_MODEL,
)
from .exceptions import VideoTrimmerError

logger = logging.getLogger(__name__)

try:
    import mlx_whisper
    HAS_MLX_WHISPER = True
except ImportError:
    HAS_MLX_WHISPER = False

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

HAS_WHISPER = HAS_MLX_WHISPER or HAS_FASTER_WHISPER


def normalize_text(text):
    """Strip punctuation and whitespace, and convert text to lowercase for alignment."""
    return re.sub(r'[^\w一-鿿]', '', text).lower()


def _compute_sentence_speaker(sentence):
    """Compute the majority speaker ID and target-speaker flag from word-level attributes."""
    words = sentence.get('words', [])
    if words:
        target_cnt = sum(1 for w in words if w.get('is_target_speaker', True))
        sentence['is_target_speaker'] = (target_cnt >= len(words) / 2.0)
        spk_counts = {}
        for w in words:
            spk = w.get('speaker_id')
            if spk:
                spk_counts[spk] = spk_counts.get(spk, 0) + 1
        if spk_counts:
            sentence['speaker_id'] = max(spk_counts.items(), key=lambda x: x[1])[0]


FILLER_PREFIXES = (
    "那麼", "那", "其實", "然後", "而且",
    "好", "來", "對", "就是", "所以",
    "OK", "ok", "現在", "另外",
)


def strip_filler_words(text: str) -> str:
    """
    Remove common spoken filler prefixes and non-alphanumeric characters.
    ASD-STE100: Use plain normalized string output.
    """
    cleaned = re.sub(r"^[^\w一-鿿]+", "", text or "").strip()
    changed = True
    while changed:
        changed = False
        for prefix in FILLER_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].lstrip("，,、 　")
                changed = True
                break
    return re.sub(r"[^\w一-鿿]", "", cleaned).lower()


def is_viable_sub_span(text: str, duration_sec: float | None = None) -> bool:
    """
    Check whether a candidate composite sub-span meets minimum lexical and acoustic length.

    ASD-STE100:
    1. Reject empty strings and micro-spans shorter than COMPOSITE_MIN_PREFIX_SEC (0.55s).
    2. Require at least 4 CJK characters, 3 Latin words, 10 Latin characters, or mixed CJK+Latin (>=2 + >=3).
    """
    norm = normalize_text(text or "")
    if len(norm) < 2:
        return False

    if duration_sec is not None and float(duration_sec) < COMPOSITE_MIN_PREFIX_SEC:
        return False

    cjk_count = sum(1 for ch in norm if is_cjk_char(ch))
    latin_words = re.findall(r"[A-Za-z0-9]+", text or "")
    latin_chars = sum(len(w) for w in latin_words)

    return (
        cjk_count >= COMPOSITE_MIN_CJK_CHARS
        or len(latin_words) >= COMPOSITE_MIN_LATIN_WORDS
        or latin_chars >= COMPOSITE_MIN_LATIN_CHARS
        or (cjk_count >= 2 and latin_chars >= 3)
    )


def is_fuzzy_prefix_restart(
    curr_text: str,
    next_text: str,
    similarity_threshold: float = 0.70,
    min_common_chars: int = 4,
) -> bool:
    """
    Detect if next_text restarts the opening clause of curr_text.
    Tolerate ASR homophone drift, filler particles, and truncated false starts.
    ASD-STE100: Return True when restart detected, else False.
    """
    c_clean = strip_filler_words(curr_text)
    n_clean = strip_filler_words(next_text)

    if len(c_clean) < 3 or len(n_clean) < 3:
        return False

    # Extract opening anchors (window of 6 to 14 characters)
    window_len = min(14, max(6, len(n_clean)))
    c_head = c_clean[:window_len]
    n_head = n_clean[:window_len]

    # 1. Exact common prefix of length >= min_common_chars
    anchor_len = min(min_common_chars, len(c_head), len(n_head))
    if anchor_len >= 3 and c_head[:anchor_len] == n_head[:anchor_len]:
        return True

    # 2. Opening clause containment (strictly compare c_head and n_head)
    # A restart occurs only when the start of n_clean matches the start of c_clean.
    # DO NOT search n_head across the entire c_clean string.
    if len(c_head) >= min_common_chars and c_head[:min_common_chars] in n_head[: min_common_chars + 3]:
        return True
    if len(n_head) >= min_common_chars and n_head[:min_common_chars] in c_head[: min_common_chars + 3]:
        return True

    # 3. Fuzzy SequenceMatcher comparison (handles homophones)
    compare_len = min(len(c_head), len(n_head))
    if compare_len >= 4:
        ratio = difflib.SequenceMatcher(None, c_head[:compare_len], n_head[:compare_len]).ratio()
        if ratio >= similarity_threshold:
            return True

    return False


def is_earlier_sentence_ng_retake(prev_text: str, next_text: str) -> bool:
    """
    Return True if prev_text is an aborted false-start prefix of next_text.

    ASD-STE100:
    Used by chunk-boundary backtracking and post-LLM retake anomaly detection.
    Never used to filter or delete sentences during Layer 1 segmentation.
    """
    if is_fuzzy_prefix_restart(prev_text, next_text):
        return True

    p_norm = normalize_text(prev_text)
    n_norm = normalize_text(next_text)
    if len(p_norm) < 3 or len(n_norm) < 3:
        return False

    # 1. Exact 3-char opening match where next_text is at least 75% as long as prev_text
    if p_norm[:3] == n_norm[:3] and len(n_norm) >= int(len(p_norm) * 0.75):
        return True

    # 2. prev_text minus a trailing 1-char stumble typo appears inside next_text
    if len(p_norm) >= 4 and p_norm[:-1] in n_norm:
        return True

    # 3. High sequence similarity (>= 0.72) between prev_text and the opening of next_text
    if len(p_norm) >= 5:
        head_window = n_norm[: len(p_norm) + 4]
        if difflib.SequenceMatcher(None, p_norm, head_window).ratio() >= 0.72:
            return True

    return False


CLOSURE_PUNCT = ("。", "！", "？", "!", "?", "……", "...")
CLAUSE_PUNCT = ("，", ",", "、", "；", ";", "：", ":")


def _split_sentence_on_paused_restarts(
    sentence_data: dict,
    min_pause_sec: float = 0.18,
    min_clause_chars: int = 5,
) -> list[dict]:
    """
    Split a merged sentence at physical intra-sentence pauses, stretched syllables, or punctuation boundaries.

    ASD-STE100:
    1. Split unconditionally after sentence-ending closure punctuation when trailing words follow.
    2. Split after clause comma punctuation when accompanied by a micro-pause (>= 0.03s) or comma-absorbed pause duration (>= 0.22s/char).
    3. Split at physical silence gaps (>= min_pause_sec), stretched word onsets, or pre-boundary hesitation drags without semantic string guessing.
    """
    words = sentence_data.get("words", [])
    if len(words) < 2:
        return [sentence_data]

    split_word_indices = []
    seg_start_idx = 0

    for j in range(1, len(words)):
        prev_word_raw = str(words[j - 1].get("word", "")).strip()
        gap = float(words[j].get("start", 0.0)) - float(words[j - 1].get("end", 0.0))
        prev_w_dur = float(words[j - 1].get("end", 0.0)) - float(words[j - 1].get("start", 0.0))
        prev_w_chars = max(1, len(normalize_text(prev_word_raw)))
        prev_w_dur_per_char = prev_w_dur / prev_w_chars
        prev_w_comma_pause = (gap >= 0.12) or ((prev_w_dur_per_char + max(0.0, gap)) >= 0.32) or (prev_w_dur_per_char >= 0.25)
        prev_w_hesitation_drag = (
            (prev_w_dur_per_char >= 0.275 and gap >= 0.06)
            or (prev_w_dur_per_char >= 0.31 and gap >= 0.035)
        )

        word_dur = float(words[j].get("end", 0.0)) - float(words[j].get("start", 0.0))
        w_chars = max(1, len(normalize_text(words[j].get("word", ""))))
        is_stretched_onset = (word_dur >= 1.20) and ((word_dur / w_chars) >= 0.45)

        prev_text = join_words_for_display([w.get("word", "") for w in words[seg_start_idx:j]])
        rem_text = join_words_for_display([w.get("word", "") for w in words[j:]])
        prev_norm = normalize_text(prev_text)
        rem_norm = normalize_text(rem_text)

        # 1. Closure punctuation inside a segment isolates trailing confirmation words
        if any(prev_word_raw.endswith(p) for p in CLOSURE_PUNCT):
            if len(prev_norm) >= 2 and len(rem_norm) >= 1:
                split_word_indices.append(j)
                seg_start_idx = j
                continue

        # 2. Clause comma punctuation with breath micro-pause or CTC comma-absorbed pause duration
        if any(prev_word_raw.endswith(p) for p in CLAUSE_PUNCT) and prev_w_comma_pause:
            if len(prev_norm) >= 6 and len(rem_norm) >= 2:
                split_word_indices.append(j)
                seg_start_idx = j
                continue

        # 3a. Opening 2-to-5 char syllable lengthening with micro-gap indicates vocal hesitation cutoff
        if (
            prev_w_hesitation_drag
            and 2 <= len(prev_norm) <= 5
            and (len(prev_norm) <= 4 or gap >= 0.06)
            and len(rem_norm) >= 4
        ):
            split_word_indices.append(j)
            seg_start_idx = j
            continue

        # 3b. Physical silence pause or stretched onset separating two clauses
        has_physical_pause = (gap >= min_pause_sec) or is_stretched_onset
        if not has_physical_pause:
            continue
        req_prev = 3 if gap >= 0.25 else min_clause_chars
        if len(prev_norm) < req_prev or len(rem_norm) < min_clause_chars:
            continue

        split_word_indices.append(j)
        seg_start_idx = j

    if not split_word_indices:
        return [sentence_data]

    sub_sentences = []
    boundaries = [0] + split_word_indices + [len(words)]
    for idx in range(len(boundaries) - 1):
        w_slice = words[boundaries[idx] : boundaries[idx + 1]]
        if not w_slice:
            continue
        sub_s = {
            "id": 0,
            "start": w_slice[0]["start"],
            "end": w_slice[-1]["end"],
            "text": join_words_for_display([w.get("word", "") for w in w_slice]),
            "words": list(w_slice),
            "orig_segment_ids": list(sentence_data.get("orig_segment_ids", [])),
            "speaker_id": sentence_data.get("speaker_id", "SPEAKER_00"),
            "is_target_speaker": sentence_data.get("is_target_speaker", True),
        }
        _compute_sentence_speaker(sub_s)
        sub_sentences.append(sub_s)

    return sub_sentences


def merge_whisper_segments_to_sentences(segments, max_gap=0.20, max_sentence_dur=8.0):
    """
    Group ASR segments into clause-level sentences using pure acoustic pauses and universal punctuation.

    ASD-STE100:
    Segment speech by physical breath pauses, punctuation boundaries, and speaker turns.
    Do not use language-specific conjunction tables or text-similarity heuristics to merge or filter takes.
    """
    if not segments:
        return []

    raw_sentences = []
    curr = None

    for s in segments:
        s_text = s["text"].strip()
        s_start = s["start"]
        s_end = s["end"]
        s_words = s.get("words", [])
        s_is_target = s.get("is_target_speaker", True)
        s_spk = s.get("speaker_id", "SPEAKER_00")

        if curr is None:
            curr = {
                "id": 1,
                "start": s_start,
                "end": s_end,
                "text": s_text,
                "words": list(s_words),
                "orig_segment_ids": [s["id"]],
                "speaker_id": s_spk,
                "is_target_speaker": s_is_target,
            }
            _compute_sentence_speaker(curr)
            continue

        gap = s_start - curr["end"]
        curr_text = curr["text"].strip()
        curr_norm_len = len(normalize_text(curr_text))
        s_norm_len = len(normalize_text(s_text))
        dur = s_end - curr["start"]

        first_w_stretched = False
        if s_words:
            fw = s_words[0]
            fw_dur = float(fw.get("end", 0.0)) - float(fw.get("start", 0.0))
            fw_chars = max(1, len(normalize_text(fw.get("word", ""))))
            first_w_stretched = (fw_dur >= 1.20) and ((fw_dur / fw_chars) >= 0.45)

        last_w_comma_pause = gap >= 0.12
        if curr["words"]:
            lw = curr["words"][-1]
            lw_dur = float(lw.get("end", 0.0)) - float(lw.get("start", 0.0))
            lw_chars = max(1, len(normalize_text(lw.get("word", ""))))
            lw_per_char = lw_dur / lw_chars
            last_w_comma_pause = last_w_comma_pause or ((lw_per_char + max(0.0, gap)) >= 0.32) or (lw_per_char >= 0.25)

        should_split = False
        # Rule 1: Physical respiration pause or stretched onset
        if (
            gap >= 0.55
            or ((gap >= max_gap or first_w_stretched) and curr_norm_len >= 5)
            or (gap >= 0.25 and curr_norm_len >= 3 and s_norm_len >= 4)
        ):
            should_split = True
        # Rule 2: Clause duration ceiling reached with micro-pause
        elif dur >= max_sentence_dur and gap >= 0.12:
            should_split = True
        # Rule 3a: Sentence-ending closure punctuation splits unconditionally
        elif any(curr_text.endswith(p) for p in CLOSURE_PUNCT):
            should_split = True
        # Rule 3b: Clause comma punctuation with breath pause or comma-absorbed pause duration
        elif any(curr_text.endswith(p) for p in CLAUSE_PUNCT) and (
            (curr_norm_len >= 6 and s_norm_len >= 2 and last_w_comma_pause)
            or gap >= 0.18
        ):
            should_split = True

        # Rule 4: Slate cue (requires physical pause >= 0.15s to protect inline acronyms) or script switch
        is_curr_slate = bool(re.match(r"^[A-Za-z0-9\s\-_]+$", curr_text))
        is_s_japanese = bool(re.search(r"[぀-ヿ]", s_text))
        is_curr_japanese = bool(re.search(r"[぀-ヿ]", curr_text))
        if is_curr_slate and not bool(re.match(r"^[A-Za-z0-9\s\-_]+$", s_text)) and gap >= 0.15:
            should_split = True
        elif is_curr_japanese != is_s_japanese:
            should_split = True

        # Rule 5: Target speaker validity transition (crew audio vs target host)
        if curr.get("is_target_speaker", True) != s_is_target:
            should_split = True

        # Rule 6: Speaker turn-taking transition
        curr_spk = curr.get("speaker_id")
        if s_spk and curr_spk and curr_spk != s_spk:
            should_split = True

        # Rule 7: Unconditional split on pauses >= 1.0s
        if gap >= 1.0:
            should_split = True

        if should_split:
            _compute_sentence_speaker(curr)
            raw_sentences.append(curr)
            curr = {
                "id": len(raw_sentences) + 1,
                "start": s_start,
                "end": s_end,
                "text": s_text,
                "words": list(s_words),
                "orig_segment_ids": [s["id"]],
                "speaker_id": s_spk,
                "is_target_speaker": s_is_target,
            }
            _compute_sentence_speaker(curr)
        else:
            curr["end"] = s_end
            curr["text"] = join_words_for_display([curr["text"], s_text])
            curr["words"].extend(s_words)
            curr["orig_segment_ids"].append(s["id"])
            _compute_sentence_speaker(curr)

    if curr is not None:
        _compute_sentence_speaker(curr)
        raw_sentences.append(curr)

    final_sentences = []
    for sent in raw_sentences:
        for sub_sent in _split_sentence_on_paused_restarts(sent):
            sub_sent["id"] = len(final_sentences) + 1
            final_sentences.append(sub_sent)

    return final_sentences


def _resolve_whisper_language_code(language: str | None) -> str | None:
    """Convert CLI language flag to a Whisper ISO-639-1 language code or None for auto-detect."""
    if not language or language.strip().lower() == "auto":
        return None
    norm = language.strip().lower().replace("_", "-")
    if norm in ("zh", "zh-tw", "zh-hk", "zh-cn", "cmn-hant-tw", "cmn-hans-cn", "yue-hant-hk"):
        return "zh"
    return norm.split("-")[0] or None


def _extract_mono_wav(video_path: Path, wav_path: Path) -> None:
    """Extract a 16kHz mono PCM WAV file from video_path via FFmpeg."""
    res = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", str(wav_path),
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 or not wav_path.exists():
        raise VideoTrimmerError(f"FFmpeg audio extraction failed for {video_path.name}: {res.stderr.strip()}")


def transcribe_video_words(
    video_path,
    out_json_path,
    asr_engine: str = DEFAULT_ASR_ENGINE,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    language: str = "auto",
    text_hint: str | None = None,
    project: str | None = None,
    gcs_bucket: str | None = None,
    stt_location: str | None = None,
    transcribe_model: str | None = None,
    force: bool = False,
):
    """
    Extract word-level acoustic ground-truth timestamps via Chirp 3 (default) or Whisper (optional).

    ASD-STE100:
    1. Default to Google Cloud Speech-to-Text V2 Chirp 3 (`asr_engine="chirp3"`) with millisecond precision.
    2. Fail fast with VideoTrimmerError on Cloud STT/GCS/ADC errors without falling back to Whisper.
    3. Tag `<basename>_words.json` with `asr_engine` and migrate legacy `<basename>_whisper_raw.json` caches automatically.
    """
    video_path = Path(video_path)
    out_json = Path(out_json_path)
    sentences_json = out_json.parent / f"{video_path.stem}_sentences.json"
    legacy_json = out_json.parent / f"{video_path.stem}_whisper_raw.json"
    engine = (asr_engine or DEFAULT_ASR_ENGINE).strip().lower()
    if engine not in ("chirp3", "whisper"):
        raise VideoTrimmerError(f"Unsupported ASR engine '{asr_engine}'. Expected 'chirp3' or 'whisper'.")

    raw_results = None
    cache_source = None
    if not force:
        if out_json.exists():
            cache_source = out_json
        elif legacy_json.exists() and legacy_json != out_json:
            cache_source = legacy_json

    if cache_source is not None:
        try:
            with open(cache_source, "r", encoding="utf-8") as f:
                cached_payload = json.load(f)
            if isinstance(cached_payload, list):
                cached_engine = "whisper"
                cached_segments = cached_payload
            elif isinstance(cached_payload, dict):
                cached_engine = str(cached_payload.get("asr_engine") or "whisper").strip().lower()
                cached_segments = list(cached_payload.get("segments") or [])
            else:
                cached_engine = ""
                cached_segments = []

            if cached_engine == engine and cached_segments:
                logger.info("Loading cached %s word timestamps: %s", engine, cache_source.name)
                raw_results = cached_segments
                if cache_source == legacy_json and out_json != legacy_json and not out_json.exists():
                    migrated_obj = (
                        cached_payload
                        if isinstance(cached_payload, dict)
                        else {"asr_engine": cached_engine, "language": language, "segments": cached_segments}
                    )
                    with open(out_json, "w", encoding="utf-8") as mf:
                        json.dump(migrated_obj, mf, ensure_ascii=False, indent=2)
                    logger.info("Migrated legacy word timestamp cache %s -> %s", legacy_json.name, out_json.name)
            elif cached_segments:
                logger.info(
                    "Cached word timestamps engine mismatch (cached=%s, requested=%s); re-transcribing...",
                    cached_engine,
                    engine,
                )
        except Exception as exc:
            logger.warning("Failed to read cached word timestamps %s (%s); re-transcribing...", cache_source.name, exc)

    if raw_results is None:
        if engine == "chirp3":
            model_id = transcribe_model or DEFAULT_CHIRP3_MODEL
            loc = stt_location or DEFAULT_STT_LOCATION
            raw_results, detected_lang = run_chirp3_transcription(
                media_path=video_path,
                language=language,
                text_hint=text_hint,
                project=project,
                gcs_bucket=gcs_bucket,
                stt_location=loc,
                model_name=model_id,
            )
            cache_obj = {
                "asr_engine": "chirp3",
                "model": model_id,
                "language": detected_lang or language,
                "segments": raw_results,
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(cache_obj, f, ensure_ascii=False, indent=2)
            logger.info("Chirp 3 transcription complete (%d raw segments).", len(raw_results))
        else:
            if not HAS_WHISPER:
                logger.info("[Notice] Neither mlx-whisper nor faster-whisper is installed. Skipping local Whisper transcription.")
                return [], []

            raw_results = []
            temp_wav = out_json.parent / f"temp_{video_path.stem}_whisper.wav"
            whisper_lang = _resolve_whisper_language_code(language)
            try:
                _extract_mono_wav(video_path, temp_wav)
                if HAS_MLX_WHISPER:
                    logger.info("Running mlx-whisper (%s) [Apple Silicon Metal GPU] for word-level transcription...", whisper_model)
                    mlx_repo = f"mlx-community/whisper-{whisper_model}-mlx"
                    mlx_kwargs = {"path_or_hf_repo": mlx_repo, "word_timestamps": True}
                    if whisper_lang:
                        mlx_kwargs["language"] = whisper_lang
                    res = mlx_whisper.transcribe(str(temp_wav), **mlx_kwargs)
                    for i, s in enumerate(res.get("segments", [])):
                        words_data = []
                        for w in s.get("words", []):
                            words_data.append({
                                "word": w["word"].strip(),
                                "start": round(float(w["start"]), 3),
                                "end": round(float(w["end"]), 3),
                            })
                        raw_results.append({
                            "id": i + 1,
                            "start": round(float(s["start"]), 3),
                            "end": round(float(s["end"]), 3),
                            "text": s["text"].strip(),
                            "words": words_data,
                        })
                else:
                    logger.info("Running faster-whisper (%s) for word-level transcription...", whisper_model)
                    model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
                    fw_kwargs = {"word_timestamps": True}
                    if whisper_lang:
                        fw_kwargs["language"] = whisper_lang
                    segments, _ = model.transcribe(str(temp_wav), **fw_kwargs)

                    for i, s in enumerate(segments):
                        words_data = []
                        if s.words:
                            for w in s.words:
                                words_data.append({
                                    "word": w.word.strip(),
                                    "start": round(float(w.start), 3),
                                    "end": round(float(w.end), 3),
                                })
                        raw_results.append({
                            "id": i + 1,
                            "start": round(float(s.start), 3),
                            "end": round(float(s.end), 3),
                            "text": s.text.strip(),
                            "words": words_data,
                        })
            finally:
                temp_wav.unlink(missing_ok=True)

            cache_obj = {
                "asr_engine": "whisper",
                "model": whisper_model,
                "language": language,
                "segments": raw_results,
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(cache_obj, f, ensure_ascii=False, indent=2)
            logger.info("Whisper transcription complete (%d raw segments).", len(raw_results))

    # Recompute clause-level sentences from raw_results so latest segmentation rules apply
    sentences = merge_whisper_segments_to_sentences(raw_results)
    with open(sentences_json, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)
    logger.info("Clause segmentation complete: %d raw segments grouped into %d sentences.", len(raw_results), len(sentences))

    return raw_results, sentences


def transcribe_video_whisper(
    video_path,
    out_json_path,
    model_name="small",
    language: str = "auto",
    force: bool = False,
):
    """Backward-compatible wrapper for local Whisper word-level transcription."""
    return transcribe_video_words(
        video_path=video_path,
        out_json_path=out_json_path,
        asr_engine="whisper",
        whisper_model=model_name,
        language=language,
        force=force,
    )


def format_whisper_transcript_for_prompt(whisper_sentences):
    """
    Format merged ASR sentences into the ground-truth text-based editing list for the Gemini prompt.

    ASD-STE100:
    Provide concise English instructions and numbered Sentence ID entries.
    """
    lines = [
        "\n---",
        "## Word-Level Ground-Truth Semantic Sentences",
        "The list below contains candidate `Sentence ID` units segmented from physical word-level timestamps at breath pauses and punctuation boundaries.",
        "Each `Sentence ID` includes its physical start and end timestamps (`start -> end`).",
        "Apply these multimodal diarization, Last-Take-Wins, and Partial Pick-Up rules:",
        "1. Select only valid sentences spoken on-camera by the primary target presenter with synchronized lip movement and direct camera gaze.",
        "2. Exclude all off-screen crew cues (such as 'Action', 'Cut', or slate codes like 'HOOK', 'Scene 1', 'Take 2' spoken while the presenter's mouth is closed or waiting).",
        "3. Exclude between-take chatter, throat clearing, and self-correction remarks directed to off-screen staff.",
        "4. Full Retake Dominance: when a later `Sentence ID` restarts the same sentence or script block from the beginning and completes it, exclude all earlier attempts and keep ONLY the final complete take.",
        "5. Partial Tail Pick-Up Backfill: when the latest `Sentence ID` is a partial pick-up covering ONLY the unfinished second half, emit the clean opening prefix (`\"composite_role\": \"prefix\"`) from the preceding attempt followed by the latest continuation (`\"composite_role\": \"continuation\"`) with zero repeated words.",
        "6. In `final_edl`, list ONLY contiguous retained `Sentence ID` integers in `sentence_ids` (never skip a `Sentence ID` inside one clip; emit separate clips for separate physical takes), write ONLY the clean retained spoken words in `transcript`, and align `source_in` and `source_out` to the sentence timestamps:\n",
    ]
    for s in whisper_sentences:
        lines.append(f"[Sentence ID: {s['id']:3d}] {s['start']:6.2f}s -> {s['end']:6.2f}s | {s['text']}")
    return "\n".join(lines)


def _find_index(units, item):
    for i, u in enumerate(units):
        if u is item:
            return i
    return None


def _neighbor_bounds(whisper_units, first_item, last_item):
    """Return (prev_sentence_end, next_sentence_start) for adjacent Whisper sentences."""
    idx_first = _find_index(whisper_units, first_item)
    idx_last = _find_index(whisper_units, last_item)
    prev_end = None
    next_start = None
    if idx_first is not None and idx_first > 0:
        prev_end = whisper_units[idx_first - 1]["end"]
    if idx_last is not None and idx_last + 1 < len(whisper_units):
        next_start = whisper_units[idx_last + 1]["start"]
    return prev_end, next_start


def align_clip_with_whisper(whisper_units, clip_data, total_dur):
    """
    Align a clip with Whisper physical word timestamps.

    ASD-STE100:
    1. Match by start_sentence_id / end_sentence_id first, then by time-window overlap.
    2. Skip leading words marked with is_target_speaker == False so t_first locks to the target presenter.
    3. Return (t_first, t_last, prev_sentence_end, next_sentence_start).
    """
    if not whisper_units:
        return clip_data.get("source_in", 0), clip_data.get("source_out", total_dur), None, None

    start_id = clip_data.get("start_sentence_id")
    if start_id is None:
        start_id = clip_data.get("start_segment_id")

    end_id = clip_data.get("end_sentence_id")
    if end_id is None:
        end_id = clip_data.get("end_segment_id")

    # 1. Match by Sentence ID range first
    if start_id is not None and end_id is not None:
        matched = [s for s in whisper_units if start_id <= s["id"] <= end_id]
        if matched:
            prev_end, next_start = _neighbor_bounds(whisper_units, matched[0], matched[-1])
            transcript = clip_data.get("transcript", "") or clip_data.get("content", "")
            if transcript:
                matched_words = []
                for s in matched:
                    matched_words.extend(s.get("words", []))
                char_timeline = []
                for w in matched_words:
                    w_norm = normalize_text(w["word"])
                    if not w_norm:
                        continue
                    w_is_target = w.get("is_target_speaker", True)
                    w_dur = (w["end"] - w["start"]) / len(w_norm)
                    for idx, ch in enumerate(w_norm):
                        ch_start = w["start"] + idx * w_dur
                        char_timeline.append((ch, round(ch_start, 3), round(ch_start + w_dur, 3), w_is_target))

                tgt_norm = normalize_text(transcript)
                whisper_str = ''.join([item[0] for item in char_timeline])
                if tgt_norm and whisper_str and char_timeline:
                    matcher = difflib.SequenceMatcher(None, tgt_norm, whisper_str)
                    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
                    if blocks and sum(b.size for b in blocks) >= max(2, len(tgt_norm) * 0.4):
                        # Lock to the first matched character spoken by the target presenter
                        t_first = None
                        for b in blocks:
                            for c_idx in range(b.b, b.b + b.size):
                                if char_timeline[c_idx][3]:  # is_target_speaker == True
                                    t_first = char_timeline[c_idx][1]
                                    break
                            if t_first is not None:
                                break
                        if t_first is None:
                            t_first = char_timeline[blocks[0].b][1]

                        last_b = blocks[-1]
                        t_last = char_timeline[min(len(char_timeline) - 1, last_b.b + last_b.size - 1)][2]
                        return t_first, t_last, prev_end, next_start

            first_words = matched[0].get("words", [])
            last_words = matched[-1].get("words", [])
            target_words = [w for w in first_words if w.get("is_target_speaker", True)]
            if target_words:
                t_first = target_words[0]["start"]
            elif first_words:
                t_first = first_words[0]["start"]
            else:
                t_first = matched[0]["start"]

            t_last = last_words[-1]["end"] if last_words else matched[-1]["end"]
            return t_first, t_last, prev_end, next_start

    raw_in = clip_data.get("source_in", 0)
    raw_out = clip_data.get("source_out", total_dur)
    transcript = clip_data.get("transcript", "")

    # 2. Filter candidate segments within the time window
    cand_segs = [s for s in whisper_units if not (s["end"] < raw_in - 2.5 or s["start"] > raw_out + 2.5)]
    if not cand_segs:
        return raw_in, raw_out, None, None

    prev_end, next_start = _neighbor_bounds(whisper_units, cand_segs[0], cand_segs[-1])

    # 3. Build character-level timeline and align with SequenceMatcher
    char_timeline = []
    for s in cand_segs:
        words = s.get("words", [])
        if words:
            for w in words:
                w_norm = normalize_text(w["word"])
                if not w_norm:
                    continue
                w_is_target = w.get("is_target_speaker", True)
                w_dur = (w["end"] - w["start"]) / len(w_norm)
                for idx, ch in enumerate(w_norm):
                    ch_start = w["start"] + idx * w_dur
                    char_timeline.append((ch, round(ch_start, 3), round(ch_start + w_dur, 3), w_is_target))
        else:
            s_norm = normalize_text(s["text"])
            if s_norm:
                s_is_target = s.get("is_target_speaker", True)
                s_dur = (s["end"] - s["start"]) / len(s_norm)
                for idx, ch in enumerate(s_norm):
                    ch_start = s["start"] + idx * s_dur
                    char_timeline.append((ch, round(ch_start, 3), round(ch_start + s_dur, 3), s_is_target))

    tgt_norm = normalize_text(transcript)
    whisper_str = ''.join([item[0] for item in char_timeline])

    if tgt_norm and whisper_str and char_timeline:
        matcher = difflib.SequenceMatcher(None, tgt_norm, whisper_str)
        blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
        if blocks:
            t_first = None
            for b in blocks:
                for c_idx in range(b.b, b.b + b.size):
                    if char_timeline[c_idx][3]:  # is_target_speaker == True
                        t_first = char_timeline[c_idx][1]
                        break
                if t_first is not None:
                    break
            if t_first is None:
                t_first = char_timeline[blocks[0].b][1]

            last_b = blocks[-1]
            t_last = char_timeline[min(len(char_timeline) - 1, last_b.b + last_b.size - 1)][2]
            return t_first, t_last, prev_end, next_start

    return cand_segs[0]["start"], cand_segs[-1]["end"], prev_end, next_start


def _find_best_char_span(
    tgt_norm: str,
    whisper_str: str,
    char_timeline: list[dict] | None = None,
) -> tuple[int, int, int] | None:
    """
    Locate the tightest, rightmost character span (start_idx, end_idx_inclusive, matched_chars)
    of `tgt_norm` inside `whisper_str`.

    ASD-STE100:
    1. Prefer an exact rightmost substring match (`rfind`) to enforce Last-Take-Wins on
       intra-sentence repeats (`A + A + B` -> second `A + B`).
    2. Compensate up to 2 leading or trailing ASR homophone characters when acoustically contiguous (< 0.25s).
    3. Slide a rightmost bounded window from `w_len - t_len` down to `0` without dropping matched edge words.
    """
    if not tgt_norm or not whisper_str:
        return None

    t_len = len(tgt_norm)
    w_len = len(whisper_str)

    # 1. Exact rightmost substring match (handles short tokens like "4%" and exact A+A+B repeats)
    exact_pos = whisper_str.rfind(tgt_norm)
    if exact_pos != -1:
        return exact_pos, exact_pos + t_len - 1, t_len

    min_block = 1 if t_len <= 3 else 2

    def _span_in_slice(w_slice: str, offset: int) -> tuple[int, int, int, int, int] | None:
        matcher = difflib.SequenceMatcher(None, tgt_norm, w_slice, autojunk=False)
        blocks = [b for b in matcher.get_matching_blocks() if b.size >= min_block]
        if not blocks:
            return None
        m_chars = sum(b.size for b in blocks)
        raw_first = offset + blocks[0].b
        raw_last = offset + min(len(w_slice) - 1, blocks[-1].b + blocks[-1].size - 1)
        raw_span = raw_last - raw_first + 1

        first_p = raw_first
        lead_unmatched = blocks[0].a
        if 0 < lead_unmatched <= 2 and m_chars >= max(2, int(t_len * 0.70)):
            for step_back in range(1, lead_unmatched + 1):
                cand_idx = raw_first - step_back
                if cand_idx < offset:
                    break
                if char_timeline is not None and (cand_idx + 1) < len(char_timeline):
                    gap_prev = float(char_timeline[cand_idx + 1].get("w_start", 0.0)) - float(
                        char_timeline[cand_idx].get("w_end", 0.0)
                    )
                    if gap_prev >= 0.25:
                        break
                first_p = cand_idx

        last_p = raw_last
        tail_unmatched = t_len - (blocks[-1].a + blocks[-1].size)
        if 0 < tail_unmatched <= 2 and m_chars >= max(2, int(t_len * 0.70)):
            max_slice_idx = offset + len(w_slice) - 1
            for step_fwd in range(1, tail_unmatched + 1):
                cand_idx = raw_last + step_fwd
                if cand_idx > max_slice_idx:
                    break
                if char_timeline is not None and cand_idx < len(char_timeline):
                    gap_next = float(char_timeline[cand_idx].get("w_start", 0.0)) - float(
                        char_timeline[cand_idx - 1].get("w_end", 0.0)
                    )
                    if gap_next >= 0.25:
                        break
                last_p = cand_idx

        comp = (raw_first - first_p) + (last_p - raw_last)
        return first_p, last_p, m_chars, raw_span, comp

    full_res = _span_in_slice(whisper_str, 0)
    if full_res is None:
        return None

    best_first, best_last, best_matched, best_raw_span, best_comp = full_res

    # 2. When whisper_str is longer than tgt_norm, scan rightmost bounded windows down from w_len - t_len
    #    so short 1-to-4 char opening stutters are skipped even when homophones prevent rfind.
    if w_len > t_len:
        win_len = min(w_len, max(t_len + 2, int(math.ceil(t_len * 1.15))))
        max_start = max(0, w_len - t_len)
        min_match_floor = int(t_len * 0.75)
        for start_idx in range(max_start, -1, -1):
            end_idx = min(w_len, start_idx + win_len)
            cand = _span_in_slice(whisper_str[start_idx:end_idx], start_idx)
            if cand is None:
                continue
            c_first, c_last, c_matched, c_raw_span, c_comp = cand

            is_better = False
            if c_matched >= min_match_floor:
                if c_matched > best_matched:
                    is_better = True
                elif c_matched == best_matched:
                    if c_raw_span < best_raw_span:
                        is_better = True
                    elif c_raw_span == best_raw_span:
                        if c_comp > best_comp or (c_comp == best_comp and c_first > best_first):
                            is_better = True
                elif c_matched == best_matched - 1 and c_raw_span <= best_raw_span - 4:
                    is_better = True

            if is_better:
                best_first, best_last, best_matched, best_raw_span, best_comp = (
                    c_first,
                    c_last,
                    c_matched,
                    c_raw_span,
                    c_comp,
                )

    return best_first, best_last, best_matched


def _build_word_char_timeline(matched_sentences: list[dict]) -> list[dict]:
    """
    Build a character-level index mapping each normalized character to its source sentence,
    word index, intra-word normalized offset, raw character index, and word time bounds.
    """
    char_timeline: list[dict] = []
    for s_idx, s in enumerate(matched_sentences):
        for w_idx, w in enumerate(s.get("words", [])):
            raw_word = str(w.get("word", ""))
            w_start = float(w.get("start", 0.0))
            w_end = float(w.get("end", w_start))
            norm_positions = [
                (raw_pos, ch.lower())
                for raw_pos, ch in enumerate(raw_word)
                if re.match(r"[\w一-鿿]", ch)
            ]
            word_norm_len = len(norm_positions)
            if word_norm_len == 0:
                continue
            for char_in_word, (raw_pos, norm_ch) in enumerate(norm_positions):
                char_timeline.append({
                    "s_idx": s_idx,
                    "w_idx": w_idx,
                    "char_in_word": char_in_word,
                    "word_norm_len": word_norm_len,
                    "norm_positions": norm_positions,
                    "raw_pos": raw_pos,
                    "ch": norm_ch,
                    "w_start": w_start,
                    "w_end": w_end,
                })
    return char_timeline


def _slice_words_on_char_span(
    matched_sentences: list[dict],
    char_timeline: list[dict],
    first_char_pos: int,
    last_char_pos: int,
) -> tuple[list[dict], set[tuple[int, int]], list[tuple[float, float]]]:
    """
    Slice `matched_sentences` to `[first_char_pos, last_char_pos]` in `char_timeline`.

    ASD-STE100:
    1. Proportionally interpolate timestamps when a cut lands inside a multi-character CJK word.
    2. Snap to whole-word boundaries when a cut lands inside an ASCII alphanumeric word.
    3. Return `(sliced_sentences, retained_word_keys, extra_external_bounds)` for hard-wall clamping.
    """
    first_entry = char_timeline[first_char_pos]
    last_entry = char_timeline[last_char_pos]
    first_s_idx, first_w_idx = first_entry["s_idx"], first_entry["w_idx"]
    last_s_idx, last_w_idx = last_entry["s_idx"], last_entry["w_idx"]

    head_was_trimmed = first_char_pos > 0
    tail_was_trimmed = last_char_pos < (len(char_timeline) - 1)

    trimmed: list[dict] = []
    retained_keys: set[tuple[int, int]] = set()
    extra_external_bounds: list[tuple[float, float]] = []

    for s_idx in range(first_s_idx, last_s_idx + 1):
        orig_s = matched_sentences[s_idx]
        s_id = int(orig_s.get("id", 0))
        words = list(orig_s.get("words", []))
        if not words:
            trimmed.append(dict(orig_s))
            continue

        w_start_idx = first_w_idx if s_idx == first_s_idx else 0
        w_end_idx = (last_w_idx + 1) if s_idx == last_s_idx else len(words)
        sliced_words: list[dict] = []

        for w_idx in range(w_start_idx, w_end_idx):
            orig_w = words[w_idx]
            w_copy = dict(orig_w)
            raw_word = str(orig_w.get("word", ""))
            orig_start = float(orig_w.get("start", 0.0))
            orig_end = float(orig_w.get("end", orig_start))
            orig_dur = max(0.0, orig_end - orig_start)

            is_first_word = (s_idx == first_s_idx and w_idx == first_w_idx)
            is_last_word = (s_idx == last_s_idx and w_idx == last_w_idx)

            raw_slice_start = 0
            raw_slice_end = len(raw_word)
            new_start = orig_start
            new_end = orig_end

            if is_first_word and first_entry["char_in_word"] > 0 and first_entry["word_norm_len"] > 1:
                prev_raw_pos = first_entry["norm_positions"][first_entry["char_in_word"] - 1][0]
                cur_raw_pos = first_entry["raw_pos"]
                both_ascii = (
                    raw_word[prev_raw_pos].isascii()
                    and raw_word[prev_raw_pos].isalnum()
                    and raw_word[cur_raw_pos].isascii()
                    and raw_word[cur_raw_pos].isalnum()
                )
                if not both_ascii:
                    ratio_s = first_entry["char_in_word"] / first_entry["word_norm_len"]
                    new_start = round(orig_start + ratio_s * orig_dur, 3)
                    raw_slice_start = cur_raw_pos
                    extra_external_bounds.append((orig_start, new_start))
                    head_was_trimmed = True

            if is_last_word and last_entry["char_in_word"] < (last_entry["word_norm_len"] - 1) and last_entry["word_norm_len"] > 1:
                cur_raw_pos = last_entry["raw_pos"]
                next_raw_pos = last_entry["norm_positions"][last_entry["char_in_word"] + 1][0]
                both_ascii = (
                    raw_word[cur_raw_pos].isascii()
                    and raw_word[cur_raw_pos].isalnum()
                    and raw_word[next_raw_pos].isascii()
                    and raw_word[next_raw_pos].isalnum()
                )
                if not both_ascii:
                    ratio_e = (last_entry["char_in_word"] + 1) / last_entry["word_norm_len"]
                    new_end = round(orig_start + ratio_e * orig_dur, 3)
                    raw_slice_end = next_raw_pos
                    extra_external_bounds.append((new_end, orig_end))
                    tail_was_trimmed = True

            if raw_slice_start > 0 or raw_slice_end < len(raw_word):
                w_copy["word"] = raw_word[raw_slice_start:raw_slice_end].strip() or raw_word
                w_copy["start"] = new_start
                w_copy["end"] = max(new_start, new_end)

            retained_keys.add((s_id, w_idx))
            sliced_words.append(w_copy)

        if not sliced_words:
            continue

        s_copy = dict(orig_s)
        s_copy["words"] = sliced_words
        s_copy["start"] = sliced_words[0]["start"]
        s_copy["end"] = sliced_words[-1]["end"]
        s_copy["text"] = join_words_for_display([w.get("word", "") for w in sliced_words])
        if s_idx == first_s_idx and (head_was_trimmed or w_start_idx > 0):
            s_copy["trimmed_head"] = True
        if s_idx == last_s_idx and (tail_was_trimmed or w_end_idx < len(words)):
            s_copy["trimmed_tail"] = True
        trimmed.append(s_copy)

    return trimmed, retained_keys, extra_external_bounds


def _split_transcript_at_norm_index(transcript: str, split_norm_idx: int) -> tuple[str, str]:
    """Split a display transcript at a normalized character index and strip seam punctuation."""
    norm_raw_indices = [
        raw_i for raw_i, ch in enumerate(transcript) if re.match(r"[\w一-鿿]", ch)
    ]
    if split_norm_idx <= 0 or split_norm_idx >= len(norm_raw_indices):
        return transcript, ""
    cut_raw_idx = norm_raw_indices[split_norm_idx]
    seam_punct = " ，,、；;：:——-　"
    left_str = transcript[:cut_raw_idx].rstrip(seam_punct)
    right_str = transcript[cut_raw_idx:].lstrip(seam_punct)
    return left_str, right_str


def _slice_transcript_norm_range(transcript: str, start_norm_idx: int, end_norm_idx: int) -> str:
    """Slice a display transcript to `[start_norm_idx, end_norm_idx)` in normalized character coordinates."""
    norm_raw_indices = [
        raw_i for raw_i, ch in enumerate(transcript) if re.match(r"[\w一-鿿]", ch)
    ]
    if not norm_raw_indices or start_norm_idx >= end_norm_idx or start_norm_idx >= len(norm_raw_indices):
        return ""
    raw_start = 0 if start_norm_idx <= 0 else norm_raw_indices[max(0, start_norm_idx)]
    if end_norm_idx >= len(norm_raw_indices):
        raw_end = len(transcript)
    else:
        next_char_raw = norm_raw_indices[end_norm_idx]
        last_char_raw = norm_raw_indices[end_norm_idx - 1]
        raw_end = last_char_raw + 1
        while raw_end < next_char_raw and transcript[raw_end] in "。！？!?，,、；;：:%）)」』”’\"'":
            raw_end += 1
    return transcript[raw_start:raw_end].lstrip(" ，,、；;：:——-　").strip()


def _align_multi_sentence_spans_to_transcript(
    matched_sentences: list[dict],
    transcript: str,
    tgt_norm: str,
) -> list[dict] | None:
    """
    Partition `transcript` monotonically from right to left across `matched_sentences` and align each sentence.

    ASD-STE100:
    1. Anchor from the rightmost sentence backward so shared boundary phrases belong to the latest take.
    2. Align each sentence against its assigned slice of `transcript` without semantic retake guessing.
    3. Record all unselected words across `matched_sentences` in `extra_bounds` to enforce physical hard walls.
    """
    if len(matched_sentences) < 2 or not tgt_norm:
        return None

    cur_end = len(tgt_norm)
    reversed_groups: list[dict] = []
    shared_extra_bounds: list[tuple[float, float]] = []
    all_retained_keys: set[tuple[int, int]] = set()
    total_matched_chars = 0

    for k in range(len(matched_sentences) - 1, -1, -1):
        s = matched_sentences[k]
        s_id = int(s.get("id", 0))
        s_words = s.get("words") or []
        s_tl = _build_word_char_timeline([s])
        s_str = "".join(item["ch"] for item in s_tl)

        if not s_str or cur_end <= 0:
            for w in s_words:
                ws = float(w.get("start", 0.0))
                shared_extra_bounds.append((ws, float(w.get("end", ws))))
            continue

        if k == 0:
            p = 0
        else:
            win_cap = min(cur_end, max(len(s_str) + 8, int(len(s_str) * 1.25)))
            win_offset = cur_end - win_cap
            tgt_tail = tgt_norm[win_offset:cur_end]
            min_blk = 1 if len(tgt_tail) <= 3 else 2
            matcher = difflib.SequenceMatcher(None, tgt_tail, s_str, autojunk=False)
            blocks = [b for b in matcher.get_matching_blocks() if b.size >= min_blk]
            valid_end_blocks = [
                idx_b for idx_b, b in enumerate(blocks)
                if (len(tgt_tail) - (b.a + b.size)) <= 3
            ]
            if not valid_end_blocks:
                # Sentence `s` does not cover the current suffix of `tgt_norm`; treat as omitted sentence
                for w in s_words:
                    ws = float(w.get("start", 0.0))
                    shared_extra_bounds.append((ws, float(w.get("end", ws))))
                continue

            anchor_i = valid_end_blocks[-1]
            first_blk = blocks[anchor_i]
            matched_in_tail = first_blk.size
            for idx_b in range(anchor_i - 1, -1, -1):
                cand_b = blocks[idx_b]
                gap_a = first_blk.a - (cand_b.a + cand_b.size)
                if gap_a <= 3:
                    first_blk = cand_b
                    matched_in_tail += cand_b.size
                else:
                    break

            if matched_in_tail < min(3, cur_end):
                for w in s_words:
                    ws = float(w.get("start", 0.0))
                    shared_extra_bounds.append((ws, float(w.get("end", ws))))
                continue

            lead_comp = min(2, first_blk.a, first_blk.b) if first_blk.b <= 2 else 0
            p = win_offset + max(0, first_blk.a - lead_comp)

        sub_tgt = tgt_norm[p:cur_end]
        if not sub_tgt:
            for w in s_words:
                ws = float(w.get("start", 0.0))
                shared_extra_bounds.append((ws, float(w.get("end", ws))))
            continue

        sub_tr = _slice_transcript_norm_range(transcript, p, cur_end) or sub_tgt
        sub_groups = _extract_matched_word_spans_by_transcript([s], sub_tr)
        s_span = _find_best_char_span(sub_tgt, s_str, char_timeline=s_tl)
        if s_span is not None:
            total_matched_chars += min(len(sub_tgt), s_span[2])

        s_kept_keys: set[tuple[int, int]] = set()
        for sg in sub_groups:
            s_kept_keys.update(sg.get("retained_keys") or set())
            shared_extra_bounds.extend(sg.get("extra_bounds") or [])
        for w_i, w in enumerate(s_words):
            if (s_id, w_i) not in s_kept_keys:
                ws = float(w.get("start", 0.0))
                shared_extra_bounds.append((ws, float(w.get("end", ws))))

        all_retained_keys.update(s_kept_keys)
        for sg in reversed(sub_groups):
            reversed_groups.append(sg)
        cur_end = p

    if not reversed_groups:
        return None
    if cur_end > max(3, int(len(tgt_norm) * 0.15)):
        return None
    if total_matched_chars < max(3, int(len(tgt_norm) * 0.75)):
        return None

    ordered_groups = list(reversed(reversed_groups))
    for sg in ordered_groups:
        sg["retained_keys"] = set(all_retained_keys)
        sg["extra_bounds"] = list(shared_extra_bounds)
    return ordered_groups


def _extract_matched_word_spans_by_transcript(
    matched_sentences: list[dict],
    transcript: str,
) -> list[dict]:
    """
    Extract matched word-span groups from `matched_sentences` aligned to `transcript`.

    ASD-STE100:
    1. When `transcript` matches a compact contiguous span, return a single trimmed group.
    2. When `matched_sentences` contains a mid-sentence stumble (`A + B_ng + B_ok + C`) excised by
       `transcript` (`A + B_ok + C`), anchor `B_ok + C` at the rightmost take first and backfill `A`
       only if both sub-spans pass `is_viable_sub_span`.
    3. When `matched_sentences` spans multiple sentences with multiple trimmed boundaries, partition
       `transcript` monotonically from right to left across sentences.
    """
    def _full_passthrough() -> list[dict]:
        keys: set[tuple[int, int]] = set()
        for s in matched_sentences:
            sid = int(s.get("id", 0))
            for w_i in range(len(s.get("words", []))):
                keys.add((sid, w_i))
        return [{
            "sentences": matched_sentences,
            "transcript": transcript,
            "composite_role": None,
            "is_composite_splice": False,
            "retained_keys": keys,
            "extra_bounds": [],
        }]

    if not transcript or not matched_sentences:
        return _full_passthrough()

    char_timeline = _build_word_char_timeline(matched_sentences)
    if not char_timeline:
        return _full_passthrough()

    tgt_norm = normalize_text(transcript)
    whisper_str = "".join(item["ch"] for item in char_timeline)
    if not tgt_norm or not whisper_str or tgt_norm == whisper_str:
        return _full_passthrough()

    span = _find_best_char_span(tgt_norm, whisper_str, char_timeline=char_timeline)
    min_required = len(tgt_norm) if len(tgt_norm) <= 3 else max(3, int(len(tgt_norm) * 0.60))

    # Check if a single compact span cleanly covers tgt_norm without spanning an excised mid-stumble
    if span is not None:
        first_pos, last_pos, matched_chars = span
        span_len = last_pos - first_pos + 1
        is_compact_single = (
            matched_chars >= max(min_required, int(len(tgt_norm) * 0.85))
            and span_len <= len(tgt_norm) + 3
        )
        if is_compact_single:
            if first_pos == 0 and last_pos == len(whisper_str) - 1:
                return _full_passthrough()
            sliced, keys, extra_bounds = _slice_words_on_char_span(
                matched_sentences, char_timeline, first_pos, last_pos
            )
            if sliced:
                return [{
                    "sentences": sliced,
                    "transcript": transcript,
                    "composite_role": None,
                    "is_composite_splice": False,
                    "retained_keys": keys,
                    "extra_bounds": extra_bounds,
                }]

    # Attempt Right-to-Left Two-Span Internal Stumble Excision (`A + B_ng + B_ok + C` -> `[A] + [B_ok + C]`)
    if len(tgt_norm) >= 8 and len(whisper_str) >= len(tgt_norm) + 3:
        best_two_span = None
        best_total_matched = 0
        for split_t in range(4, len(tgt_norm) - 3):
            tgt_prefix = tgt_norm[:split_t]
            tgt_suffix = tgt_norm[split_t:]

            suf_res = _find_best_char_span(tgt_suffix, whisper_str, char_timeline=char_timeline)
            if suf_res is None:
                continue
            suf_start, suf_end, suf_matched = suf_res
            if suf_matched < max(4, int(len(tgt_suffix) * 0.80)):
                continue
            if (suf_end - suf_start + 1) > len(tgt_suffix) + 3:
                continue
            if suf_start < 4:
                continue

            pre_res = _find_best_char_span(tgt_prefix, whisper_str[:suf_start], char_timeline=char_timeline[:suf_start])
            if pre_res is None:
                continue
            pre_start, pre_end, pre_matched = pre_res
            if pre_matched < max(4, int(len(tgt_prefix) * 0.80)):
                continue
            if (pre_end - pre_start + 1) > len(tgt_prefix) + 3:
                continue

            excised_chars = suf_start - pre_end - 1
            if excised_chars < 3:
                continue

            total_m = pre_matched + suf_matched
            # Prefer the smallest split_t (longest rightmost continuation) among top matches
            if total_m >= int(len(tgt_norm) * 0.85) and total_m > best_total_matched:
                best_total_matched = total_m
                best_two_span = (split_t, pre_start, pre_end, suf_start, suf_end)

        if best_two_span is not None:
            split_t, pre_start, pre_end, suf_start, suf_end = best_two_span
            pre_sents, pre_keys, pre_extra = _slice_words_on_char_span(
                matched_sentences, char_timeline, pre_start, pre_end
            )
            suf_sents, suf_keys, suf_extra = _slice_words_on_char_span(
                matched_sentences, char_timeline, suf_start, suf_end
            )
            if pre_sents and suf_sents:
                pre_dur = float(pre_sents[-1]["end"]) - float(pre_sents[0]["start"])
                suf_dur = float(suf_sents[-1]["end"]) - float(suf_sents[0]["start"])
                pre_tr, suf_tr = _split_transcript_at_norm_index(transcript, split_t)
                if not pre_tr:
                    pre_tr = join_words_for_display([s["text"] for s in pre_sents])
                if not suf_tr:
                    suf_tr = join_words_for_display([s["text"] for s in suf_sents])

                if is_viable_sub_span(pre_tr, pre_dur) and is_viable_sub_span(suf_tr, suf_dur):
                    pre_sents[-1] = dict(pre_sents[-1], trimmed_tail=True)
                    suf_sents[0] = dict(suf_sents[0], trimmed_head=True)
                    shared_extra = pre_extra + suf_extra
                    return [
                        {
                            "sentences": pre_sents,
                            "transcript": pre_tr,
                            "composite_role": "prefix",
                            "is_composite_splice": False,
                            "retained_keys": pre_keys,
                            "extra_bounds": shared_extra,
                        },
                        {
                            "sentences": suf_sents,
                            "transcript": suf_tr,
                            "composite_role": "continuation",
                            "is_composite_splice": True,
                            "retained_keys": suf_keys,
                            "extra_bounds": shared_extra,
                        },
                    ]
                # Prefix failed viability gate (e.g. < 0.55s or < 4 chars): keep rightmost continuation only
                if is_viable_sub_span(suf_tr, suf_dur):
                    suf_sents[0] = dict(suf_sents[0], trimmed_head=True)
                    return [{
                        "sentences": suf_sents,
                        "transcript": suf_tr,
                        "composite_role": None,
                        "is_composite_splice": False,
                        "retained_keys": suf_keys,
                        "extra_bounds": suf_extra,
                    }]

    # Multi-sentence monotonic Right-to-Left alignment when multiple boundaries were trimmed
    if len(matched_sentences) > 1:
        multi_groups = _align_multi_sentence_spans_to_transcript(matched_sentences, transcript, tgt_norm)
        if multi_groups:
            return multi_groups

    # Fallback to single best span when matched_chars meets minimum requirement
    if span is not None and span[2] >= min_required:
        sliced, keys, extra_bounds = _slice_words_on_char_span(
            matched_sentences, char_timeline, span[0], span[1]
        )
        if sliced:
            return [{
                "sentences": sliced,
                "transcript": transcript,
                "composite_role": None,
                "is_composite_splice": False,
                "retained_keys": keys,
                "extra_bounds": extra_bounds,
            }]

    return _full_passthrough()


def _trim_matched_words_by_transcript(matched_sentences: list[dict], transcript: str) -> list[dict]:
    """
    Trim leading and trailing words of matched sentences to match the LLM-selected transcript.
    """
    groups = _extract_matched_word_spans_by_transcript(matched_sentences, transcript)
    if len(groups) == 1:
        return groups[0]["sentences"]
    combined: list[dict] = []
    for g in groups:
        combined.extend(g["sentences"])
    return combined if combined else matched_sentences


def _split_sentence_words_by_internal_gap(sentence: dict, max_word_gap: float = 0.45) -> list[dict]:
    """
    Split a sentence into compact speech chunks when internal word gaps exceed max_word_gap.

    ASD-STE100:
    Divide a sentence at physical silence gaps (>= max_word_gap) so the acoustic engine
    can tighten dead air inside a single sentence, and preserve `trimmed_head` / `trimmed_tail` flags.
    """
    words = [w for w in sentence.get("words", []) if w.get("is_target_speaker", True)]
    if not words:
        words = list(sentence.get("words", []))
    if not words:
        return [{
            "sentence_id": sentence.get("id", 0),
            "t_first": float(sentence.get("start", 0.0)),
            "t_last": float(sentence.get("end", 0.0)),
            "text": sentence.get("text", "").strip(),
            "words": [],
            "trimmed_head": bool(sentence.get("trimmed_head", False)),
            "trimmed_tail": bool(sentence.get("trimmed_tail", False)),
        }]

    chunks = []
    cur_words = [words[0]]
    for w in words[1:]:
        gap = float(w.get("start", 0.0)) - float(cur_words[-1].get("end", 0.0))
        if gap >= max_word_gap:
            chunks.append({
                "sentence_id": sentence.get("id", 0),
                "t_first": float(cur_words[0]["start"]),
                "t_last": float(cur_words[-1]["end"]),
                "text": join_words_for_display([x.get("word", "") for x in cur_words]),
                "words": list(cur_words),
                "trimmed_head": False,
                "trimmed_tail": False,
            })
            cur_words = [w]
        else:
            cur_words.append(w)

    if cur_words:
        chunks.append({
            "sentence_id": sentence.get("id", 0),
            "t_first": float(cur_words[0]["start"]),
            "t_last": float(cur_words[-1]["end"]),
            "text": join_words_for_display([x.get("word", "") for x in cur_words]),
            "words": list(cur_words),
            "trimmed_head": False,
            "trimmed_tail": False,
        })

    if chunks:
        chunks[0]["trimmed_head"] = bool(sentence.get("trimmed_head", False))
        chunks[-1]["trimmed_tail"] = bool(sentence.get("trimmed_tail", False))

    return chunks


def resolve_clip_sub_units(
    whisper_units: list[dict],
    clip_data: dict,
    total_dur: float,
    max_internal_gap: float = 0.40,
    max_word_gap: float = 0.45,
) -> list[dict]:
    """
    Resolve a single EDL clip into one or more compact speech sub-units.

    ASD-STE100:
    1. Read `sentence_ids` first, or fall back to `start_sentence_id`..`end_sentence_id`.
    2. Slice word boundaries to `clip_data["transcript"]` (supporting intra-sentence stumble excision
       and CJK multi-character intra-token proportional interpolation).
    3. Compute word-level millisecond hard-wall neighbor bounds (`prev_sentence_end`, `next_sentence_start`).
    4. Coalesce consecutive sentence IDs when the physical gap is below max_internal_gap (0.40s)
       and neither boundary was trimmed or marked as a composite splice.
    """
    parent_role = str(clip_data.get("composite_role") or "none").strip().lower()
    if parent_role not in ("prefix", "continuation"):
        parent_role = "none"
    parent_is_splice = bool(clip_data.get("is_composite_splice", False)) or (parent_role == "continuation")

    if not whisper_units:
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(whisper_units, clip_data, total_dur)
        raw_t = clip_data.get("transcript", "") or clip_data.get("content", "")
        return [{
            "sentence_ids": [],
            "t_first": t_first,
            "t_last": t_last,
            "prev_sentence_end": prev_end,
            "next_sentence_start": next_start,
            "next_word_hard_wall": None,
            "words": [],
            "transcript": raw_t,
            "asr_transcript": raw_t,
            "trimmed_head": bool(clip_data.get("trimmed_head", False)),
            "trimmed_tail": bool(clip_data.get("trimmed_tail", False)),
            "composite_role": parent_role,
            "is_composite_splice": parent_is_splice,
        }]

    matched = []
    raw_sent_ids = clip_data.get("sentence_ids")
    if isinstance(raw_sent_ids, list) and raw_sent_ids:
        valid_ids = {int(x) for x in raw_sent_ids if isinstance(x, (int, str)) and str(x).isdigit()}
        matched = [s for s in whisper_units if s.get("id") in valid_ids]

    if not matched:
        start_id = clip_data.get("start_sentence_id")
        if start_id is None:
            start_id = clip_data.get("start_segment_id")
        end_id = clip_data.get("end_sentence_id")
        if end_id is None:
            end_id = clip_data.get("end_segment_id")
        if start_id is not None and end_id is not None:
            matched = [s for s in whisper_units if start_id <= s.get("id", 0) <= end_id]

    if not matched and ("source_in" in clip_data or "source_out" in clip_data):
        s_in = float(clip_data.get("source_in", 0.0))
        s_out = float(clip_data.get("source_out", total_dur))
        matched = [
            u for u in whisper_units
            if u.get("is_target_speaker", True)
            and float(u.get("end", 0.0)) >= s_in - 0.20
            and float(u.get("start", 0.0)) <= s_out + 0.20
        ]

    if not matched:
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(whisper_units, clip_data, total_dur)
        raw_t = clip_data.get("transcript", "") or clip_data.get("content", "")
        return [{
            "sentence_ids": [],
            "t_first": t_first,
            "t_last": t_last,
            "prev_sentence_end": prev_end,
            "next_sentence_start": next_start,
            "next_word_hard_wall": None,
            "words": [],
            "transcript": raw_t,
            "asr_transcript": raw_t,
            "trimmed_head": bool(clip_data.get("trimmed_head", False)),
            "trimmed_tail": bool(clip_data.get("trimmed_tail", False)),
            "composite_role": parent_role,
            "is_composite_splice": parent_is_splice,
        }]

    # Filter out off-screen crew sentences (is_target_speaker == False)
    target_matched = [s for s in matched if s.get("is_target_speaker", True)]
    if target_matched:
        matched = target_matched

    # Support deterministic --cached-json replay when t_first/t_last sub-span bounds already exist
    cached_t_first = clip_data.get("t_first")
    cached_t_last = clip_data.get("t_last")
    cached_pre_window_extra_bounds: list[tuple[float, float]] = []
    if (
        cached_t_first is not None
        and cached_t_last is not None
        and (parent_is_splice or parent_role != "none" or clip_data.get("trimmed_head") or clip_data.get("trimmed_tail"))
    ):
        tf_val = float(cached_t_first)
        tl_val = float(cached_t_last)
        windowed_matched = []
        for s in matched:
            s_words = list(s.get("words", []))
            if not s_words:
                if float(s.get("end", 0.0)) >= tf_val - 0.05 and float(s.get("start", 0.0)) <= tl_val + 0.05:
                    windowed_matched.append(dict(s))
                continue
            kept_w = []
            for w in s_words:
                ws = float(w.get("start", 0.0))
                we = float(w.get("end", ws))
                if we < tf_val - 0.03 or ws > tl_val + 0.03:
                    cached_pre_window_extra_bounds.append((ws, we))
                    continue
                w_c = dict(w)
                if ws < tf_val - 0.005 and we > tf_val + 0.005:
                    cached_pre_window_extra_bounds.append((ws, tf_val))
                    w_c["start"] = round(tf_val, 3)
                if we > tl_val + 0.005 and ws < tl_val - 0.005:
                    cached_pre_window_extra_bounds.append((tl_val, we))
                    w_c["end"] = round(tl_val, 3)
                kept_w.append(w_c)
            if kept_w:
                s_c = dict(s)
                s_c["words"] = kept_w
                s_c["start"] = kept_w[0]["start"]
                s_c["end"] = kept_w[-1]["end"]
                s_c["text"] = join_words_for_display([x.get("word", "") for x in kept_w])
                s_c["trimmed_head"] = kept_w[0]["start"] > float(s_words[0].get("start", 0.0)) + 0.005
                s_c["trimmed_tail"] = kept_w[-1]["end"] < float(s_words[-1].get("end", 0.0)) - 0.005
                windowed_matched.append(s_c)
        if windowed_matched:
            matched = windowed_matched

    orig_first_sid = matched[0].get("id")
    orig_last_sid = matched[-1].get("id")

    clip_transcript = (clip_data.get("transcript", "") or clip_data.get("content", "")).strip()
    if clip_transcript and not cached_pre_window_extra_bounds:
        span_groups = _extract_matched_word_spans_by_transcript(matched, clip_transcript)
    else:
        keys_all: set[tuple[int, int]] = set()
        for s in matched:
            sid = int(s.get("id", 0))
            for w_i in range(len(s.get("words", []))):
                keys_all.add((sid, w_i))
        span_groups = [{
            "sentences": matched,
            "transcript": clip_transcript,
            "composite_role": None,
            "is_composite_splice": False,
            "retained_keys": keys_all,
            "extra_bounds": cached_pre_window_extra_bounds,
        }]

    grouped_units = []
    for sg in span_groups:
        sg_sents = sg["sentences"]
        if not sg_sents:
            continue
        if sg_sents[0].get("id") != orig_first_sid:
            sg_sents[0] = dict(sg_sents[0], trimmed_head=True)
        if sg_sents[-1].get("id") != orig_last_sid:
            sg_sents[-1] = dict(sg_sents[-1], trimmed_tail=True)
        if clip_data.get("trimmed_head"):
            sg_sents[0] = dict(sg_sents[0], trimmed_head=True)
        if clip_data.get("trimmed_tail"):
            sg_sents[-1] = dict(sg_sents[-1], trimmed_tail=True)

        fine_chunks = []
        for s in sg_sents:
            fine_chunks.extend(_split_sentence_words_by_internal_gap(s, max_word_gap=max_word_gap))
        if not fine_chunks:
            continue

        sg_role = sg["composite_role"] or (parent_role if parent_role != "none" else "none")
        sg_splice = bool(sg["is_composite_splice"]) or parent_is_splice or (sg_role == "continuation")

        local_groups = []
        cur_group = {
            "sentence_ids": [fine_chunks[0]["sentence_id"]],
            "t_first": fine_chunks[0]["t_first"],
            "t_last": fine_chunks[0]["t_last"],
            "texts": [fine_chunks[0]["text"]],
            "words": list(fine_chunks[0].get("words", [])),
            "trimmed_head": bool(fine_chunks[0].get("trimmed_head", False)),
            "trimmed_tail": bool(fine_chunks[0].get("trimmed_tail", False)),
            "group_transcript": sg["transcript"],
            "composite_role": sg_role,
            "is_composite_splice": sg_splice,
            "retained_keys": sg["retained_keys"],
            "extra_bounds": sg["extra_bounds"],
        }

        for ch in fine_chunks[1:]:
            gap = ch["t_first"] - cur_group["t_last"]
            prev_sid = cur_group["sentence_ids"][-1]
            cur_sid = ch["sentence_id"]
            skipped_sentence = cur_sid > prev_sid + 1
            trimmed_boundary = bool(cur_group.get("trimmed_tail")) or bool(ch.get("trimmed_head"))

            if gap >= max_internal_gap or skipped_sentence or trimmed_boundary:
                local_groups.append(cur_group)
                cur_group = {
                    "sentence_ids": [cur_sid],
                    "t_first": ch["t_first"],
                    "t_last": ch["t_last"],
                    "texts": [ch["text"]],
                    "words": list(ch.get("words", [])),
                    "trimmed_head": bool(ch.get("trimmed_head", False)),
                    "trimmed_tail": bool(ch.get("trimmed_tail", False)),
                    "group_transcript": sg["transcript"],
                    "composite_role": "none",
                    "is_composite_splice": False,
                    "retained_keys": sg["retained_keys"],
                    "extra_bounds": sg["extra_bounds"],
                }
            else:
                if cur_sid not in cur_group["sentence_ids"]:
                    cur_group["sentence_ids"].append(cur_sid)
                cur_group["t_last"] = ch["t_last"]
                cur_group["texts"].append(ch["text"])
                cur_group["words"].extend(ch.get("words", []))
                cur_group["trimmed_tail"] = bool(ch.get("trimmed_tail", False))

        local_groups.append(cur_group)
        for lg_idx, lg in enumerate(local_groups):
            lg["is_single_in_span_group"] = (len(local_groups) == 1)
            if len(local_groups) > 1:
                if lg_idx == 0 and sg_role == "prefix":
                    lg["composite_role"] = "none"
                    local_groups[-1]["composite_role"] = "prefix"
                elif lg_idx > 0 and sg_role == "continuation":
                    lg["composite_role"] = "none"
                    lg["is_composite_splice"] = False
            grouped_units.append(lg)

    if not grouped_units:
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(whisper_units, clip_data, total_dur)
        return [{
            "sentence_ids": [],
            "t_first": t_first,
            "t_last": t_last,
            "prev_sentence_end": prev_end,
            "next_sentence_start": next_start,
            "next_word_hard_wall": None,
            "words": [],
            "transcript": clip_transcript,
            "asr_transcript": clip_transcript,
            "trimmed_head": bool(clip_data.get("trimmed_head", False)),
            "trimmed_tail": bool(clip_data.get("trimmed_tail", False)),
            "composite_role": parent_role,
            "is_composite_splice": parent_is_splice,
            "llm_verified_window": bool(clip_data.get("llm_verified_window", False)),
        }]

    sub_units = []
    for idx, g in enumerate(grouped_units):
        t_f = float(g["t_first"])
        t_l = float(g["t_last"])
        g_sids = set(g["sentence_ids"])
        g_retained_keys: set[tuple[int, int]] = g["retained_keys"]

        external_bounds: list[tuple[float, float]] = list(g.get("extra_bounds", []))
        for u in whisper_units:
            u_id = int(u.get("id", 0))
            u_words = u.get("words") or []
            if u_words:
                for w_i, w in enumerate(u_words):
                    ws = float(w.get("start", 0.0))
                    we = float(w.get("end", ws))
                    # Words outside this sub-unit's retained word span act as physical hard walls
                    if (u_id, w_i) not in g_retained_keys or we <= t_f - 0.005 or ws >= t_l + 0.005:
                        external_bounds.append((ws, we))
            elif u_id not in g_sids:
                external_bounds.append((float(u.get("start", 0.0)), float(u.get("end", 0.0))))

        prev_ends = [b_end for (_, b_end) in external_bounds if b_end <= t_f + 0.02]
        if idx > 0:
            prev_ends.append(float(grouped_units[idx - 1]["t_last"]))
        prev_end = round(max(prev_ends), 3) if prev_ends else None

        next_starts = [b_start for (b_start, _) in external_bounds if b_start >= t_l - 0.02]
        if idx + 1 < len(grouped_units):
            next_starts.append(float(grouped_units[idx + 1]["t_first"]))
        next_start = round(min(next_starts), 3) if next_starts else None

        sub_text = join_words_for_display([t for t in g["texts"] if t])
        grp_tr = g.get("group_transcript", "")
        primary_transcript = (grp_tr if (g.get("is_single_in_span_group") and grp_tr) else sub_text) or grp_tr
        t_tail = bool(g.get("trimmed_tail", False))
        c_role = g.get("composite_role", "none")
        is_splice = bool(g.get("is_composite_splice", False)) or (c_role == "continuation")

        sub_units.append({
            "sentence_ids": list(g["sentence_ids"]),
            "t_first": round(t_f, 3),
            "t_last": round(t_l, 3),
            "prev_sentence_end": prev_end,
            "next_sentence_start": next_start,
            "next_word_hard_wall": next_start if (t_tail or c_role == "prefix") else None,
            "words": list(g.get("words", [])),
            "transcript": primary_transcript,
            "asr_transcript": sub_text or primary_transcript,
            "trimmed_head": bool(g.get("trimmed_head", False)),
            "trimmed_tail": t_tail,
            "composite_role": c_role,
            "is_composite_splice": is_splice,
            "llm_verified_window": bool(clip_data.get("llm_verified_window", False)),
        })

    return sub_units


def coalesce_adjacent_sub_units(
    expanded_units: list[dict],
    max_internal_gap: float = 0.40,
) -> list[dict]:
    """
    Coalesce consecutive sub-units across clip boundaries when no sentence ID is skipped,
    neither boundary was trimmed or marked as a composite splice, and the pause is below max_internal_gap.

    ASD-STE100:
    Prevent artificial jump-cuts when the LLM splits a continuous take across two clips,
    while never re-merging across a boundary where a speech stumble or pick-up splice was cut.
    """
    if len(expanded_units) <= 1:
        return list(expanded_units)

    coalesced = [dict(expanded_units[0])]
    for unit in expanded_units[1:]:
        prev = coalesced[-1]
        prev_sids = prev.get("sentence_ids") or []
        curr_sids = unit.get("sentence_ids") or []

        gap = float(unit.get("t_first", 0.0)) - float(prev.get("t_last", 0.0))
        is_contiguous_sid = (
            bool(prev_sids)
            and bool(curr_sids)
            and 0 <= (int(curr_sids[0]) - int(prev_sids[-1])) <= 1
        )
        has_trimmed_boundary = (
            bool(prev.get("trimmed_tail"))
            or bool(unit.get("trimmed_head"))
            or bool(unit.get("is_composite_splice"))
            or prev.get("composite_role") == "prefix"
            or unit.get("composite_role") == "continuation"
        )

        if is_contiguous_sid and not has_trimmed_boundary and 0.0 <= gap < max_internal_gap:
            merged_sids = list(prev_sids)
            for sid in curr_sids:
                if sid not in merged_sids:
                    merged_sids.append(sid)
            prev["sentence_ids"] = merged_sids
            prev["t_last"] = float(unit["t_last"])
            prev["next_sentence_start"] = unit.get("next_sentence_start")
            prev["next_word_hard_wall"] = unit.get("next_word_hard_wall")
            prev["trimmed_tail"] = bool(unit.get("trimmed_tail", False))
            prev["llm_verified_window"] = bool(prev.get("llm_verified_window") and unit.get("llm_verified_window"))
            if "words" in prev or "words" in unit:
                prev["words"] = list(prev.get("words") or []) + list(unit.get("words") or [])
            prev_text = (prev.get("transcript") or "").strip()
            curr_text = (unit.get("transcript") or "").strip()
            if curr_text and curr_text not in prev_text:
                prev["transcript"] = join_words_for_display([prev_text, curr_text])
            prev_w_text = (prev.get("asr_transcript") or prev.get("whisper_transcript") or "").strip()
            curr_w_text = (unit.get("asr_transcript") or unit.get("whisper_transcript") or "").strip()
            if curr_w_text and curr_w_text not in prev_w_text:
                prev["asr_transcript"] = join_words_for_display([prev_w_text, curr_w_text])
            elif prev_w_text:
                prev["asr_transcript"] = prev_w_text
            prev.pop("whisper_transcript", None)
        else:
            u_copy = dict(unit)
            if "asr_transcript" not in u_copy and "whisper_transcript" in u_copy:
                u_copy["asr_transcript"] = u_copy.pop("whisper_transcript")
            coalesced.append(u_copy)

    if "asr_transcript" not in coalesced[0] and "whisper_transcript" in coalesced[0]:
        coalesced[0]["asr_transcript"] = coalesced[0].pop("whisper_transcript")

    return coalesced


def calculate_active_script_window(
    whisper_units: list[dict],
    script_text: str,
    total_duration: float,
    padding_seconds: float = 20.0,
) -> tuple[float, float, list[dict]]:
    """
    Align spoken clauses from a reference script against Whisper sentences to locate
    the active recording window in a multi-episode or long camera roll.

    ASD-STE100:
    This function matches script clauses with transcript sentences.
    It returns the start time, end time, and sentences inside the active window.
    """
    if not whisper_units or not script_text or not script_text.strip():
        return 0.0, total_duration, whisper_units

    raw_lines = re.split(r"[\n。！？!?；;]+", script_text)
    script_clauses = []
    for line in raw_lines:
        cleaned = re.sub(r"^\s*[#*>\-\d.]+\s*", "", line)
        cleaned = re.sub(r"（.*?）|\(.*?\)|【.*?】|\[.*?\]", "", cleaned)
        norm = normalize_text(cleaned)
        if len(norm) >= 6:
            script_clauses.append(norm)

    if not script_clauses:
        return 0.0, total_duration, whisper_units

    matched_unit_indices = set()
    matched_clause_count = 0

    norm_units = [(idx, normalize_text(u.get("text", ""))) for idx, u in enumerate(whisper_units)]

    for clause in script_clauses:
        clause_matched = False
        for idx, u_norm in norm_units:
            if len(u_norm) < 4:
                continue
            if clause in u_norm or (len(u_norm) >= 6 and u_norm in clause):
                matched_unit_indices.add(idx)
                clause_matched = True
                continue
            matcher = difflib.SequenceMatcher(None, clause, u_norm)
            longest = matcher.find_longest_match(0, len(clause), 0, len(u_norm))
            if longest.size >= 6 or (min(len(clause), len(u_norm)) >= 8 and matcher.ratio() >= 0.58):
                matched_unit_indices.add(idx)
                clause_matched = True
        if clause_matched:
            matched_clause_count += 1

    # Require at least 40% script clause coverage to activate windowing
    coverage = matched_clause_count / max(1, len(script_clauses))
    if not matched_unit_indices or coverage < 0.40:
        return 0.0, total_duration, whisper_units

    sorted_indices = sorted(matched_unit_indices)
    first_idx = sorted_indices[0]
    last_idx = sorted_indices[-1]

    win_start = max(0.0, round(float(whisper_units[first_idx].get("start", 0.0)) - padding_seconds, 2))
    win_end = min(total_duration, round(float(whisper_units[last_idx].get("end", total_duration)) + padding_seconds, 2))

    # If the window already covers >= 85% of the video, keep the full video
    if (win_end - win_start) >= total_duration * 0.85:
        return 0.0, total_duration, whisper_units

    filtered_units = [
        u for u in whisper_units
        if float(u.get("end", 0.0)) >= win_start and float(u.get("start", 0.0)) <= win_end
    ]
    return win_start, win_end, (filtered_units or whisper_units)


def _are_sentences_retake_related(text_a: str, text_b: str) -> bool:
    """
    Check if two sentences belong to the same retake cluster.
    """
    if is_earlier_sentence_ng_retake(text_a, text_b) or is_earlier_sentence_ng_retake(text_b, text_a):
        return True
    na = normalize_text(text_a)
    nb = normalize_text(text_b)
    if len(na) >= 4 and len(nb) >= 4 and na[:4] == nb[:4]:
        return True
    return False


def _backtrack_before_retake_cluster(
    whisper_units: list[dict],
    candidate_idx: int,
    min_idx: int,
) -> int:
    """
    Backtrack the split index so that a retake cluster is never split across two chunks.

    ASD-STE100:
    This function checks if sentences after the split repeat sentences before the split.
    If a repeat exists, it moves the split index backward before the first take.
    """
    idx = candidate_idx
    while idx > min_idx:
        after_slice = whisper_units[idx + 1 : min(len(whisper_units), idx + 5)]
        before_start = max(min_idx, idx - 4)
        earliest_overlap_idx = None

        for b_idx in range(before_start, idx + 1):
            b_text = whisper_units[b_idx].get("text", "")
            for a_unit in after_slice:
                a_text = a_unit.get("text", "")
                if _are_sentences_retake_related(b_text, a_text):
                    earliest_overlap_idx = b_idx
                    break
            if earliest_overlap_idx is not None:
                break

        if earliest_overlap_idx is not None and earliest_overlap_idx > min_idx:
            # Step back to the sentence immediately before Take 1 of the retake cluster
            idx = earliest_overlap_idx - 1
        else:
            break

    return idx


def detect_chunk_boundaries(
    whisper_units: list[dict],
    total_duration: float,
    target_chunk_duration: float = 480.0,
    min_pause_duration: float = 2.0,
    max_chunk_duration: float = 660.0,
    window_start: float | None = None,
) -> list[tuple[float, float, list[dict]]]:
    """
    Split the timeline into logical chunks at natural speech pauses (>= 2.0s).
    Backtracks before any retake cluster so Take 1 and Take N stay in the same chunk.

    ASD-STE100:
    This function finds pauses between spoken sentences.
    It groups sentences into chunks between 6 and 11 minutes long.

    Returns:
        List of tuples: (chunk_start_sec, chunk_end_sec, chunk_sentences)
    """
    if not whisper_units:
        start_t = 0.0 if window_start is None else window_start
        return [(start_t, total_duration, [])]

    first_unit_start = float(whisper_units[0].get("start", 0.0))
    chunk_start = window_start if window_start is not None else (0.0 if first_unit_start < 30.0 else max(0.0, first_unit_start - 5.0))
    effective_span = total_duration - chunk_start

    if effective_span <= max_chunk_duration:
        return [(chunk_start, total_duration, whisper_units)]

    chunks = []
    start_idx = 0
    n_units = len(whisper_units)

    while start_idx < n_units:
        # Check if remaining duration fits in a single final chunk
        if (total_duration - chunk_start) <= max_chunk_duration:
            chunks.append((chunk_start, total_duration, whisper_units[start_idx:]))
            break

        split_idx = None
        min_backtrack_idx = start_idx

        for i in range(start_idx, n_units - 1):
            s_curr_end = float(whisper_units[i].get("end", 0.0))
            chunk_elapsed = s_curr_end - chunk_start
            if chunk_elapsed >= target_chunk_duration * 0.65 and min_backtrack_idx == start_idx:
                min_backtrack_idx = i

            s_next_start = float(whisper_units[i + 1].get("start", 0.0))
            pause = s_next_start - s_curr_end

            if chunk_elapsed >= target_chunk_duration and pause >= min_pause_duration:
                safe_idx = _backtrack_before_retake_cluster(whisper_units, i, min_backtrack_idx)
                safe_pause = float(whisper_units[safe_idx + 1].get("start", 0.0)) - float(whisper_units[safe_idx].get("end", 0.0))
                if safe_idx == i or safe_pause >= 0.40:
                    split_idx = safe_idx
                    break

            if chunk_elapsed >= max_chunk_duration:
                # Backtrack within [min_backtrack_idx, i] to find the largest non-retake pause
                best_idx = i
                best_score = -1.0
                search_start = max(start_idx + 1, min_backtrack_idx)
                for cand in range(search_start, i + 1):
                    safe_cand = _backtrack_before_retake_cluster(whisper_units, cand, search_start)
                    cand_pause = (
                        float(whisper_units[safe_cand + 1].get("start", 0.0))
                        - float(whisper_units[safe_cand].get("end", 0.0))
                    )
                    if cand_pause > best_score:
                        best_score = cand_pause
                        best_idx = safe_cand
                split_idx = best_idx
                break

        if split_idx is None or split_idx >= n_units - 1:
            chunks.append((chunk_start, total_duration, whisper_units[start_idx:]))
            break

        s_curr_end = float(whisper_units[split_idx].get("end", 0.0))
        s_next_start = float(whisper_units[split_idx + 1].get("start", 0.0))
        pause = max(0.0, s_next_start - s_curr_end)
        split_point = round(s_curr_end + (pause / 2.0), 2)

        chunk_sentences = whisper_units[start_idx : split_idx + 1]
        chunks.append((chunk_start, split_point, chunk_sentences))

        chunk_start = split_point
        start_idx = split_idx + 1

    return chunks

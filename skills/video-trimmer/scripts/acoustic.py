"""Acoustic analysis: CPS calculation, post-pause onset clamping, overlap pivot selection, and boundary refinement."""

import logging
import math
import re

import numpy as np

from .constants import (
    COMPOSITE_GAIN_MATCH_MAX_DB,
    COMPOSITE_GAIN_MATCH_THRESHOLD_DB,
    COMPOSITE_MIN_PREFIX_SEC,
    COMPOSITE_ONSET_MAX_INTERNAL_GAP_SEC,
    COMPOSITE_SEAM_IN_MARGIN_SEC,
    COMPOSITE_SEAM_OUT_MARGIN_SEC,
    HANGOVER_STEPS,
    SILENCE_THRESHOLD_DBFS,
)

logger = logging.getLogger(__name__)

# Linear amplitude threshold converted from SILENCE_THRESHOLD_DBFS (-40 dBFS ~= 0.01).
_SILENCE_THRESHOLD_LINEAR = 10 ** (SILENCE_THRESHOLD_DBFS / 20)

# Maintain hysteresis ratio (3.5 : 1.8) between speech onset and silence decay thresholds.
_SPEECH_SILENCE_RATIO = 3.5 / 1.8


def calculate_clip_cps(transcript, duration):
    """Calculate presenter speaking rate in Characters/Syllables Per Second (CPS)."""
    if not transcript or duration <= 0:
        return 3.0, 0
    zh = len(re.findall(r'[一-鿿]', transcript))
    en_chars = len(re.findall(r'[a-zA-Z0-9]', transcript))
    syllables = zh + int(en_chars * 0.6)
    if syllables == 0:
        syllables = len(re.sub(r'\s+', '', transcript))
    dur = max(0.5, duration)
    return round(syllables / dur, 2), syllables


def _sliding_window_rms(audio, starts: np.ndarray, win: int) -> np.ndarray:
    """Compute RMS values at sample indices `starts` with window length `win` via cumulative sum."""
    if len(starts) == 0 or win <= 0 or len(audio) < win:
        return np.empty(0, dtype=np.float64)
    s0 = int(starts[0])
    s1 = min(len(audio), int(starts[-1]) + win)
    seg = np.asarray(audio[s0:s1], dtype=np.float64)
    csum = np.empty(len(seg) + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(seg * seg, out=csum[1:])
    rel = starts - s0
    msq = (csum[rel + win] - csum[rel]) / float(win)
    return np.sqrt(np.maximum(0.0, msq))


def _estimate_local_noise_floor(audio, sr, center_sec: float) -> float:
    """Estimate local ambient RMS noise floor in a 2.2s window around center_sec."""
    hop = max(1, int(0.01 * sr))
    win = max(1, int(0.03 * sr))
    idx_center = int(center_sec * sr)
    idx_start = max(0, idx_center - int(1.0 * sr))
    idx_end = min(len(audio), idx_center + int(1.2 * sr))
    if idx_end - idx_start <= win:
        return _SILENCE_THRESHOLD_LINEAR * 0.15
    starts = np.arange(idx_start, idx_end - win, hop, dtype=np.int64)
    all_rms = _sliding_window_rms(audio, starts, win)
    return float(np.percentile(all_rms, 10)) if len(all_rms) > 0 else _SILENCE_THRESHOLD_LINEAR * 0.15


def clamp_post_pause_word_onset(
    audio,
    sr: int,
    word_start: float,
    word_end: float,
    prev_word_end: float | None = None,
    gap_before: float = 0.0,
    is_trimmed_head: bool = False,
) -> float:
    """
    Clamp an inflated post-pause word start timestamp to the true speech onset.

    ASD-STE100:
    1. Activate only when is_trimmed_head is True or gap_before >= 0.15s.
    2. Locate the main voiced core inside [word_start, word_end] and walk backward across
       consonant dips up to COMPOSITE_ONSET_MAX_INTERNAL_GAP_SEC (120ms).
    3. Stop before pre-pickup breaths, sighs, or lip clicks separated by a silence valley.
    """
    w_s = float(word_start)
    w_e = float(word_end)
    if prev_word_end is not None:
        w_s = max(w_s, float(prev_word_end))
    if not (is_trimmed_head or gap_before >= 0.15) or (w_e - w_s) < 0.14 or len(audio) == 0:
        return round(w_s, 3)

    win = max(1, int(0.02 * sr))
    noise_floor = _estimate_local_noise_floor(audio, sr, (w_s + w_e) / 2.0)
    low_thresh = max(_SILENCE_THRESHOLD_LINEAR * 0.85, noise_floor * 1.6)
    core_thresh = max(_SILENCE_THRESHOLD_LINEAR * _SPEECH_SILENCE_RATIO, noise_floor * 3.2)

    times = np.arange(w_s, max(w_s + 0.01, w_e - 0.02), 0.01)
    if len(times) < 4:
        return round(w_s, 3)

    indices = (times * sr).astype(np.int64)
    rms_vals = np.zeros(len(indices), dtype=np.float64)
    valid_mask = (indices >= 0) & ((indices + win) <= len(audio))
    if np.any(valid_mask):
        rms_vals[valid_mask] = _sliding_window_rms(audio, indices[valid_mask], win)

    # Locate the rightmost sustained voiced core inside the word
    core_idx = None
    for i in range(len(rms_vals) - 1, -1, -1):
        if rms_vals[i] >= core_thresh:
            core_idx = i
            break
    if core_idx is None:
        return round(w_s, 3)

    max_gap_steps = max(1, int(round(COMPOSITE_ONSET_MAX_INTERNAL_GAP_SEC / 0.01)))
    onset_idx = core_idx
    low_run = 0
    for i in range(core_idx, -1, -1):
        if rms_vals[i] < low_thresh:
            low_run += 1
            if low_run > max_gap_steps:
                break
        else:
            low_run = 0
            onset_idx = i

    clamped = max(w_s, float(times[onset_idx]) - 0.015)
    clamped = min(clamped, max(w_s, w_e - 0.060))
    if prev_word_end is not None:
        clamped = max(clamped, float(prev_word_end))
    return round(clamped, 3)


def _segment_voiced_db(audio, sr: int, start_sec: float, end_sec: float) -> float | None:
    """Compute RMS loudness in dBFS over voiced 20ms frames within [start_sec, end_sec]."""
    if len(audio) == 0 or end_sec <= start_sec:
        return None
    win = max(1, int(0.02 * sr))
    hop = max(1, int(0.01 * sr))
    i_start = max(0, int(start_sec * sr))
    i_end = min(len(audio), int(end_sec * sr))
    if i_end - i_start < win:
        return None
    starts = np.arange(i_start, i_end - win + 1, hop, dtype=np.int64)
    frames = _sliding_window_rms(audio, starts, win)
    if len(frames) == 0:
        return None
    voiced = frames[frames >= _SILENCE_THRESHOLD_LINEAR * 0.9]
    target = voiced if len(voiced) > 0 else frames
    mean_rms = float(np.sqrt(np.mean(np.square(target))))
    if mean_rms <= 1e-7:
        return None
    return 20.0 * math.log10(mean_rms)


def find_optimal_overlap_pivot(
    audio,
    sr: int,
    prefix_words: list[dict],
    cont_words: list[dict],
    overlap_word_count: int,
    min_prefix_sec: float = COMPOSITE_MIN_PREFIX_SEC,
) -> int:
    """
    Select the lowest-cost word-boundary pivot index `p` in [0, overlap_word_count].

    ASD-STE100:
    1. Pivot `p` keeps `prefix_words[:len(prefix_words) - overlap_word_count + p]` in the prefix clip
       and `cont_words[p:]` in the continuation clip.
    2. Evaluate inter-word pause duration, boundary RMS energy, and local 200ms loudness continuity.
    3. Bias slightly toward p=0 so the latest take speaks the entire restarted phrase unless a mid-overlap
       pause yields a cleaner acoustic seam.
    """
    if overlap_word_count <= 0 or not prefix_words or not cont_words:
        return 0

    n_pref = len(prefix_words)
    n_cont = len(cont_words)
    best_p = 0
    best_cost = float("inf")
    win = max(1, int(0.02 * sr))

    for p in range(0, overlap_word_count + 1):
        pref_keep = n_pref - overlap_word_count + p
        if pref_keep <= 0 or p >= n_cont:
            continue
        left_w = prefix_words[:pref_keep]
        right_w = cont_words[p:]
        left_dur = float(left_w[-1]["end"]) - float(left_w[0]["start"])
        right_dur = float(right_w[-1]["end"]) - float(right_w[0]["start"])
        if left_dur < min_prefix_sec or right_dur < min_prefix_sec:
            continue

        t_left_cut = float(left_w[-1]["end"])
        t_right_cut = float(right_w[0]["start"])

        pause_bonus = 0.0
        if pref_keep < n_pref:
            pause_bonus += min(0.25, max(0.0, float(prefix_words[pref_keep]["start"]) - t_left_cut))
        if p > 0:
            pause_bonus += min(0.25, max(0.0, t_right_cut - float(cont_words[p - 1]["end"])))

        boundary_rms = 0.0
        if len(audio) > win:
            idx_l = min(max(0, int(t_left_cut * sr)), len(audio) - win)
            idx_r = min(max(0, int(t_right_cut * sr)), len(audio) - win)
            rms_l = float(np.sqrt(np.mean(audio[idx_l : idx_l + win] ** 2)))
            rms_r = float(np.sqrt(np.mean(audio[idx_r : idx_r + win] ** 2)))
            boundary_rms = (rms_l + rms_r) * 5.0

        db_l = _segment_voiced_db(audio, sr, max(float(left_w[0]["start"]), t_left_cut - 0.22), t_left_cut)
        db_r = _segment_voiced_db(audio, sr, t_right_cut, min(float(right_w[-1]["end"]), t_right_cut + 0.22))
        db_jump = abs(db_l - db_r) * 0.12 if (db_l is not None and db_r is not None) else 0.0

        anchor_bias = -0.15 if p == 0 else 0.0
        cost = boundary_rms + db_jump - (pause_bonus * 1.8) + anchor_bias
        if cost < best_cost:
            best_cost = cost
            best_p = p

    return best_p


def compute_composite_gain_db(
    audio,
    sr: int,
    prefix_bounds: tuple[float, float],
    cont_bounds: tuple[float, float],
) -> float:
    """
    Calculate bounded gain compensation (in dB) for a composite continuation clip.

    ASD-STE100:
    Compare voiced RMS dBFS between the prefix clip and the continuation clip.
    Return 0.0 when the difference is below COMPOSITE_GAIN_MATCH_THRESHOLD_DB (1.5 dB);
    otherwise apply 75% gain compensation clamped within [-4.5 dB, +4.5 dB].
    """
    pref_db = _segment_voiced_db(audio, sr, float(prefix_bounds[0]), float(prefix_bounds[1]))
    cont_db = _segment_voiced_db(audio, sr, float(cont_bounds[0]), float(cont_bounds[1]))
    if pref_db is None or cont_db is None:
        return 0.0
    diff_db = pref_db - cont_db
    if abs(diff_db) < COMPOSITE_GAIN_MATCH_THRESHOLD_DB:
        return 0.0
    gain = max(-COMPOSITE_GAIN_MATCH_MAX_DB, min(COMPOSITE_GAIN_MATCH_MAX_DB, diff_db * 0.75))
    return round(gain, 2)


def refine_speech_bounds_locked(
    audio,
    sr,
    t_first,
    t_last,
    total_dur,
    transcript="",
    pacing="auto",
    prev_sentence_end=None,
    next_sentence_start=None,
    tight_out_seam: bool = False,
    tight_in_seam: bool = False,
):
    """
    Refine clip cut-in and cut-out points around word-level ground-truth timestamps.

    ASD-STE100:
    1. Track the local noise floor and forward speech decay from t_last without truncating final phonemes.
    2. Clamp forward and backward acoustic searches within adjacent word/sentence hard walls
       (prev_sentence_end and next_sentence_start) to avoid capturing excised stumbles or noise.
    3. Use compact 40ms seam margins when tight_out_seam or tight_in_seam is enabled on composite splices.
    """
    t_first = float(t_first)
    t_last = float(t_last)

    # 1. Calculate presenter CPS
    raw_dur = max(0.5, t_last - t_first)
    cps, _ = calculate_clip_cps(transcript, raw_dur)

    # 2. Estimate local ambient noise floor (10ms hop, 30ms window)
    win = max(1, int(0.03 * sr))
    noise_floor = _estimate_local_noise_floor(audio, sr, t_last)

    # Combine absolute dBFS threshold with local noise floor fallback for noisy environments
    silence_thresh = max(_SILENCE_THRESHOLD_LINEAR, noise_floor * 1.8)
    speech_thresh = max(_SILENCE_THRESHOLD_LINEAR * _SPEECH_SILENCE_RATIO, noise_floor * 3.5)

    # 3. Track out-point forward from t_last (never cut earlier than t_last)
    silent_count = 0
    true_speech_end = t_last

    max_forward_scan = 0.08 if tight_out_seam else 1.2
    search_end_limit = min(total_dur - 0.02, t_last + max_forward_scan)
    if next_sentence_start is not None:
        search_end_limit = min(search_end_limit, float(next_sentence_start))

    f_steps = np.arange(t_last, max(t_last, search_end_limit), 0.01)
    for t in f_steps:
        idx = int(t * sr)
        if idx + win >= len(audio):
            break
        r = float(np.sqrt(np.mean(audio[idx : idx + win] ** 2)))
        peak = float(np.max(np.abs(audio[idx : idx + win])))

        # Stop before sudden off-screen impulse or crew voice after a short silence gap
        if silent_count >= 5 and (r > speech_thresh * 2 or peak > 0.30):
            true_speech_end = max(t_last, float(t) - (silent_count * 0.01) + 0.02)
            break

        if r < silence_thresh:
            silent_count += 1
            if silent_count >= HANGOVER_STEPS:
                true_speech_end = max(t_last, float(t) - (HANGOVER_STEPS * 0.01) + 0.03)
                break
        else:
            silent_count = 0
            true_speech_end = float(t)

    if next_sentence_start is not None:
        true_speech_end = min(true_speech_end, float(next_sentence_start))

    # 4. Calculate dynamic lead-in and settle-out buffers from presenter CPS
    if pacing == "compact" or cps >= 5.5:
        settle_out = 0.06
        breath_in = 0.06
    elif pacing == "breathing" or cps <= 3.0:
        settle_out = 0.16
        breath_in = 0.14
    else:
        alpha = (cps - 3.0) / (5.5 - 3.0)
        settle_out = 0.16 - alpha * (0.16 - 0.06)
        breath_in = 0.14 - alpha * (0.14 - 0.06)

    if tight_out_seam:
        settle_out = min(settle_out, COMPOSITE_SEAM_OUT_MARGIN_SEC)
    if tight_in_seam:
        breath_in = min(breath_in, COMPOSITE_SEAM_IN_MARGIN_SEC)

    target_out = min(float(total_dur), true_speech_end + settle_out)
    if next_sentence_start is not None:
        target_out = min(target_out, float(next_sentence_start))
    final_out = round(target_out, 3)
    if next_sentence_start is not None and final_out > float(next_sentence_start):
        final_out = round(float(next_sentence_start) - 0.0005, 3)

    # 5. Snap cut-in point to vocal onset and trim pre-speech dead air
    in_idx = int(t_first * sr)
    in_rms = float(np.sqrt(np.mean(audio[in_idx : in_idx + win] ** 2))) if in_idx + win < len(audio) else 0.0
    true_speech_start = t_first

    if in_rms < speech_thresh:
        # ASR word start fell inside pre-speech silence; scan forward to true vocal onset
        search_forward_e = min(t_last - 0.15, t_first + 2.0)
        if search_forward_e > t_first:
            for t in np.arange(t_first, search_forward_e, 0.01):
                idx = int(t * sr)
                if idx + win >= len(audio):
                    break
                r = float(np.sqrt(np.mean(audio[idx : idx + win] ** 2)))
                if r >= speech_thresh:
                    true_speech_start = float(t)
                    break
    else:
        # Speaker is already phonating at t_first; scan backward up to 150ms without crossing prev_sentence_end
        max_back = 0.06 if tight_in_seam else 0.15
        search_back_limit = max(0.0, t_first - max_back)
        if prev_sentence_end is not None:
            search_back_limit = max(search_back_limit, float(prev_sentence_end))
        for t in np.arange(t_first, search_back_limit, -0.01):
            idx = int(t * sr)
            if idx < 0:
                break
            r = float(np.sqrt(np.mean(audio[idx : idx + win] ** 2)))
            if r < silence_thresh:
                true_speech_start = float(t) + 0.01
                break
            true_speech_start = float(t)

    target_in = max(0.0, true_speech_start - breath_in)
    if prev_sentence_end is not None:
        target_in = max(target_in, float(prev_sentence_end))
    final_in = round(target_in, 3)
    if prev_sentence_end is not None and final_in < float(prev_sentence_end):
        final_in = round(float(prev_sentence_end) + 0.0005, 3)

    in_m = round(t_first - final_in, 3)
    out_m = round(final_out - t_last, 3)

    return final_in, final_out, cps, in_m, out_m


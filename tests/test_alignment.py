import unittest
import tests  # noqa: F401
import pytest

from pathlib import Path
import tempfile
from scripts.gemini_client import build_prompt
from scripts.transcribe import (
    align_clip_with_whisper,
    coalesce_adjacent_sub_units,
    resolve_clip_sub_units,
)


def _sentence(id_, start, end, text, words):
    return {"id": id_, "start": start, "end": end, "text": text, "words": words}


def _words_for(text, start, dur):
    """均勻切分字級時間戳，供合成測試資料使用。"""
    n = len(text)
    w = dur / n
    return [
        {"word": ch, "start": round(start + i * w, 3), "end": round(start + (i + 1) * w, 3)}
        for i, ch in enumerate(text)
    ]


def _build_units():
    return [
        _sentence(1, 0.0, 1.0, "早安大家好", _words_for("早安大家好", 0.0, 1.0)),
        _sentence(2, 1.2, 2.4, "今天天氣晴朗溫暖", _words_for("今天天氣晴朗溫暖", 1.2, 1.2)),
        _sentence(3, 2.6, 3.6, "謝謝收看再見", _words_for("謝謝收看再見", 2.6, 1.0)),
    ]


class TestAlignClipWithWhisper(unittest.TestCase):
    def test_empty_whisper_units_returns_clip_bounds_and_no_neighbors(self):
        clip = {"source_in": 5.0, "source_out": 8.0}
        t_first, t_last, prev_end, next_start = align_clip_with_whisper([], clip, total_dur=10.0)
        assert (t_first, t_last, prev_end, next_start) == (5.0, 8.0, None, None)

    def test_id_match_without_transcript_uses_word_bounds(self):
        units = _build_units()
        clip = {"start_sentence_id": 2, "end_sentence_id": 2}
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(units, clip, total_dur=10.0)
        assert t_first == pytest.approx(1.2)
        assert t_last == pytest.approx(2.4)
        assert prev_end == pytest.approx(1.0)    # 上一句 (id=1) 的結尾
        assert next_start == pytest.approx(2.6)  # 下一句 (id=3) 的起點

    def test_id_match_with_transcript_uses_char_level_diff_alignment(self):
        units = _build_units()
        clip = {"start_sentence_id": 2, "end_sentence_id": 2, "transcript": "今天天氣晴朗溫暖"}
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(units, clip, total_dur=10.0)
        assert t_first == pytest.approx(1.2, abs=0.05)
        assert t_last == pytest.approx(2.4, abs=0.05)
        assert prev_end == pytest.approx(1.0)
        assert next_start == pytest.approx(2.6)

    def test_first_sentence_has_no_prev_neighbor(self):
        units = _build_units()
        clip = {"start_sentence_id": 1, "end_sentence_id": 1}
        _, _, prev_end, next_start = align_clip_with_whisper(units, clip, total_dur=10.0)
        assert prev_end is None
        assert next_start == pytest.approx(1.2)

    def test_last_sentence_has_no_next_neighbor(self):
        units = _build_units()
        clip = {"start_sentence_id": 3, "end_sentence_id": 3}
        _, _, prev_end, next_start = align_clip_with_whisper(units, clip, total_dur=10.0)
        assert prev_end == pytest.approx(2.4)
        assert next_start is None

    def test_time_window_fallback_when_no_ids_provided(self):
        units = _build_units()
        clip = {"source_in": 1.1, "source_out": 2.5, "transcript": "今天天氣晴朗溫暖"}
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(units, clip, total_dur=10.0)
        assert t_first == pytest.approx(1.2, abs=0.05)
        assert t_last == pytest.approx(2.4, abs=0.05)

    def test_no_candidate_segments_returns_raw_window(self):
        units = _build_units()
        clip = {"source_in": 100.0, "source_out": 105.0, "transcript": "無關內容"}
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(units, clip, total_dur=200.0)
        assert (t_first, t_last, prev_end, next_start) == (100.0, 105.0, None, None)

    def test_target_speaker_in_point_lock_skips_non_target_words(self):
        """若首個句子的前置單詞為場外非目標人員 (is_target_speaker=False)，t_first 跳至首個主講人單詞"""
        words = [
            {"word": "Action", "start": 1.0, "end": 1.5, "is_target_speaker": False},
            {"word": "嗨", "start": 2.0, "end": 2.3, "is_target_speaker": True},
            {"word": "大家好", "start": 2.3, "end": 3.0, "is_target_speaker": True},
        ]
        sentence = _sentence(1, 1.0, 3.0, "Action 嗨 大家好", words)
        clip = {"start_sentence_id": 1, "end_sentence_id": 1, "transcript": "嗨 大家好"}
        t_first, t_last, _, _ = align_clip_with_whisper([sentence], clip, total_dur=10.0)
        assert t_first == pytest.approx(2.0, abs=0.05)
        assert t_last == pytest.approx(3.0, abs=0.05)

    def test_resolve_clip_sub_units_preserves_selected_sentences_and_splits_dead_air_pauses(self):
        """驗證 resolve_clip_sub_units 尊重 Gemini 選取的 sentence_ids 不做破壞性刪除，並在 >=0.40s 看稿停頓處拆開收緊"""
        units = [
            _sentence(1, 10.0, 12.0, "打開了潘朵拉的盒子對就是WiFi你大概也聽過", _words_for("打開了潘朵拉的盒子對就是WiFi你大概也聽過", 10.0, 2.0)),
            _sentence(2, 12.5, 14.5, "你大概也聽過WiFi訊號撞到人體會出現細微摔", _words_for("你大概也聽過WiFi訊號撞到人體會出現細微摔", 12.5, 2.0)),
            _sentence(3, 16.0, 19.0, "你大概也聽過WiFi訊號撞到人體會出現細微衰減與相位變化", _words_for("你大概也聽過WiFi訊號撞到人體會出現細微衰減與相位變化", 16.0, 3.0)),
        ]
        # Gemini 選取了 Sentence 1 與 Sentence 3 (跳過 NG Sentence 2)，且 Sentence 1 結尾與 Sentence 3 開頭為頂真修辭銜接
        clip = {"sentence_ids": [1, 3], "start_sentence_id": 1, "end_sentence_id": 3, "topic": "HOOK"}
        sub_units = resolve_clip_sub_units(units, clip, total_dur=30.0)

        # Sentence 1 絕不會被誤刪，且因與 Sentence 3 間隔長空白而被拆為 2 個緊湊子片段！
        assert len(sub_units) == 2
        assert sub_units[0]["t_first"] == pytest.approx(10.0, abs=0.05)
        assert sub_units[0]["t_last"] == pytest.approx(12.0, abs=0.05)
        assert sub_units[1]["t_first"] == pytest.approx(16.0, abs=0.05)
        assert sub_units[1]["t_last"] == pytest.approx(19.0, abs=0.05)

    def test_resolve_clip_sub_units_trims_trailing_stumble_via_llm_transcript(self):
        """Verify that when LLM selects Sentence 6 but omits a trailing false start in transcript, t_last snaps to the clean word boundary."""
        words = [
            {"word": "完美的監控工具", "start": 39.16, "end": 42.50},
            {"word": "其實已經在你家裡了", "start": 42.50, "end": 45.20},
            {"word": "那就是WiFi", "start": 45.20, "end": 47.10},
            {"word": "你大概也聽過細微摔", "start": 47.30, "end": 51.60},
        ]
        s6 = _sentence(
            6,
            39.16,
            51.60,
            "完美的監控工具其實已經在你家裡了那就是WiFi你大概也聽過細微摔",
            words,
        )
        clip = {
            "sentence_ids": [6],
            "start_sentence_id": 6,
            "end_sentence_id": 6,
            "transcript": "完美的監控工具其實已經在你家裡了，那就是WiFi。",
            "topic": "Opening",
        }
        sub_units = resolve_clip_sub_units([s6], clip, total_dur=60.0)
        assert len(sub_units) == 1
        assert sub_units[0]["t_first"] == pytest.approx(39.16, abs=0.02)
        assert sub_units[0]["t_last"] == pytest.approx(47.10, abs=0.02)

    def test_coalesce_adjacent_sub_units_merges_contiguous_clips_and_preserves_skipped_id_cuts(self):
        """Verify global cross-clip coalescing merges contiguous Sentence IDs (<0.40s) across clips without merging across skipped NG IDs."""
        expanded = [
            {
                "topic": "Part A",
                "sentence_ids": [6],
                "t_first": 39.16,
                "t_last": 43.20,
                "prev_sentence_end": 34.0,
                "next_sentence_start": 43.35,
                "transcript": "完美的監控工具其實已經在你家裡了",
            },
            {
                "topic": "Part B",
                "sentence_ids": [7],
                "t_first": 43.35,
                "t_last": 47.10,
                "prev_sentence_end": 43.20,
                "next_sentence_start": 47.30,
                "transcript": "那就是WiFi",
            },
            {
                "topic": "Part C (Skipped NG ID 8)",
                "sentence_ids": [9],
                "t_first": 51.60,
                "t_last": 58.00,
                "prev_sentence_end": 51.40,
                "next_sentence_start": None,
                "transcript": "你大概也聽過WiFi訊號撞到人體會出現細微衰減",
            },
        ]
        coalesced = coalesce_adjacent_sub_units(expanded, max_internal_gap=0.40)
        assert len(coalesced) == 2
        assert coalesced[0]["sentence_ids"] == [6, 7]
        assert coalesced[0]["t_first"] == pytest.approx(39.16)
        assert coalesced[0]["t_last"] == pytest.approx(47.10)
        assert coalesced[1]["sentence_ids"] == [9]
        assert coalesced[1]["t_first"] == pytest.approx(51.60)

    def test_dual_mode_prompt_formatting_mode_a_and_mode_b(self):
        """Verify build_prompt generates Mode A numbered script blocks with script_path and Mode B without script_path in 100% English."""
        import re

        prompt_candidates = [
            Path(__file__).resolve().parent.parent
            / "skills"
            / "video-trimmer"
            / "prompts"
            / "video_cut_prompt.md"
        ]
        en_units = [
            _sentence(1, 0.0, 1.0, "Hello everyone", _words_for("Hello everyone", 0.0, 1.0)),
            _sentence(2, 1.2, 2.4, "Today we test the system", _words_for("Today we test the system", 1.2, 1.2)),
        ]

        # Mode B (no script)
        prompt_b = build_prompt(prompt_candidates, script_path=None, whisper_units=en_units)
        assert "Mode B: Unscripted Intent-Window Take Arbitration (Active)" in prompt_b
        assert "repeating a key phrase three times for emphasis" in prompt_b
        assert not re.search(r"[\u4e00-\u9fff]", prompt_b)

        # Mode A (with script)
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tmp:
            tmp.write("# Section 1\nFirst paragraph of the shooting script.\nSecond paragraph of the shooting script.\n")
            tmp_path = Path(tmp.name)
        try:
            prompt_a = build_prompt(prompt_candidates, script_path=tmp_path, whisper_units=en_units)
            assert "Mode A: Monotonic Script-Anchored Take Arbitration (Active)" in prompt_a
            assert "[Script Block 01] First paragraph of the shooting script." in prompt_a
            assert "[Script Block 02] Second paragraph of the shooting script." in prompt_a
            assert not re.search(r"[\u4e00-\u9fff]", prompt_a)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_tc04_short_transcript_trimming_locks_exact_rightmost_word_bounds(self):
        """TC-04: Verify 1-to-2 char transcripts like '4%' are not rejected by a 3-char floor and lock to the final token."""
        words = [
            {"word": "4%", "start": 250.0, "end": 250.5},
            {"word": "呃", "start": 251.0, "end": 252.0},
            {"word": "對不起重來", "start": 252.2, "end": 255.0},
            {"word": "4%", "start": 257.4, "end": 258.0},
        ]
        s_short = _sentence(38, 250.0, 258.0, "4% 呃 對不起重來 4%", words)
        clip = {
            "sentence_ids": [38],
            "start_sentence_id": 38,
            "end_sentence_id": 38,
            "transcript": "4%",
            "topic": "Short Figure",
        }
        sub_units = resolve_clip_sub_units([s_short], clip, total_dur=300.0)
        assert len(sub_units) == 1
        assert sub_units[0]["t_first"] == pytest.approx(257.4, abs=0.02)
        assert sub_units[0]["t_last"] == pytest.approx(258.0, abs=0.02)
        assert sub_units[0]["trimmed_head"] is True

    def test_tc05_trimmed_boundary_prevents_coalesce_from_resurrecting_stumble(self):
        """TC-05: Verify coalesce_adjacent_sub_units never merges across a boundary where the LLM trimmed a tail stumble."""
        words_s6 = [
            {"word": "完美的監控工具那就是WiFi", "start": 39.16, "end": 47.10},
            {"word": "你大概也聽過細微摔", "start": 47.20, "end": 51.60},
        ]
        words_s7 = [
            {"word": "對就是你家路由器發出的訊號", "start": 47.25, "end": 55.00},
        ]
        s6 = _sentence(6, 39.16, 51.60, "完美的監控工具那就是WiFi你大概也聽過細微摔", words_s6)
        s7 = _sentence(7, 47.25, 55.00, "對就是你家路由器發出的訊號", words_s7)

        clip1 = {"sentence_ids": [6], "transcript": "完美的監控工具那就是WiFi", "topic": "Part 1"}
        clip2 = {"sentence_ids": [7], "transcript": "對就是你家路由器發出的訊號", "topic": "Part 2"}

        su1 = resolve_clip_sub_units([s6, s7], clip1, total_dur=60.0)
        su2 = resolve_clip_sub_units([s6, s7], clip2, total_dur=60.0)
        assert su1[0]["trimmed_tail"] is True
        assert su1[0]["t_last"] == pytest.approx(47.10, abs=0.02)

        # Even though Sentence 6 and 7 are contiguous IDs and gap (47.25 - 47.10 = 0.15s) < 0.40s,
        # coalesce_adjacent_sub_units must NOT merge them because su1.trimmed_tail is True.
        coalesced = coalesce_adjacent_sub_units(su1 + su2, max_internal_gap=0.40)
        assert len(coalesced) == 2
        assert coalesced[0]["t_last"] == pytest.approx(47.10, abs=0.02)
        assert coalesced[1]["t_first"] == pytest.approx(47.25, abs=0.02)

    def test_intra_sentence_repeat_trims_to_rightmost_repetition_and_keeps_gemini_transcript(self):
        """Verify A + A + B intra-sentence repeat trims to the second A + B and preserves Gemini's corrected transcript."""
        words = [
            {"word": "研究者推測", "start": 100.0, "end": 101.5},
            {"word": "資料本來就是壓縮", "start": 101.5, "end": 103.8},
            {"word": "研究者推測", "start": 104.0, "end": 105.5},
            {"word": "資料本來就是壓縮", "start": 105.5, "end": 107.8},
            {"word": "後的資訊", "start": 107.8, "end": 109.5},
        ]
        s15 = _sentence(15, 100.0, 109.5, "研究者推測資料本來就是壓縮研究者推測資料本來就是壓縮後的資訊", words)
        clip = {
            "sentence_ids": [15],
            "transcript": "研究者推測，資料本來就是壓縮後的資訊。",
            "topic": "Compression Explanation",
        }
        sub_units = resolve_clip_sub_units([s15], clip, total_dur=120.0)
        assert len(sub_units) == 1
        assert sub_units[0]["t_first"] == pytest.approx(104.0, abs=0.02)
        assert sub_units[0]["t_last"] == pytest.approx(109.5, abs=0.02)
        assert sub_units[0]["transcript"] == "研究者推測，資料本來就是壓縮後的資訊。"
        assert sub_units[0]["asr_transcript"] == "研究者推測資料本來就是壓縮後的資訊"

    def test_cjk_intra_token_proportional_interpolation_slices_multichar_word(self):
        """Interpolate word boundaries proportionally when a trim cut lands inside a multi-char CJK token."""
        s1 = {
            "id": 1,
            "start": 10.0,
            "end": 12.0,
            "text": "今天我們介紹核心架構",
            "words": [
                {"word": "今天我們", "start": 10.0, "end": 10.8},
                # 4-character CJK word spanning [10.8, 11.6]; keeping only "介紹" (2/4 chars) -> end = 11.2
                {"word": "介紹核心", "start": 10.8, "end": 11.6},
                {"word": "架構", "start": 11.6, "end": 12.0},
            ],
        }
        clip = {
            "sentence_ids": [1],
            "transcript": "今天我們介紹",
            "composite_role": "prefix",
            "topic": "Opening Prefix",
        }
        sub_units = resolve_clip_sub_units([s1], clip, total_dur=30.0)
        assert len(sub_units) == 1
        su = sub_units[0]
        assert su["t_first"] == pytest.approx(10.0, abs=0.01)
        assert su["t_last"] == pytest.approx(11.2, abs=0.02)
        assert su["next_word_hard_wall"] == pytest.approx(11.2, abs=0.02)
        assert su["composite_role"] == "prefix"

    def test_intra_sentence_mid_span_stumble_excision_splits_into_prefix_and_continuation(self):
        """Excise an internal stumble (A + B_ng + B_ok + C) into a Last-Take-Anchored prefix + continuation pair."""
        s1 = {
            "id": 1,
            "start": 20.0,
            "end": 26.0,
            "text": "今天我們來介紹核心系統呃核心系統的三個關鍵指標",
            "words": [
                {"word": "今天我們", "start": 20.00, "end": 20.70},
                {"word": "來介紹", "start": 20.70, "end": 21.30},
                {"word": "核心系統", "start": 21.32, "end": 22.10},
                {"word": "呃", "start": 22.12, "end": 22.45},
                {"word": "核心系統", "start": 22.60, "end": 23.40},
                {"word": "的三個", "start": 23.40, "end": 24.20},
                {"word": "關鍵指標", "start": 24.20, "end": 25.60},
            ],
        }
        clip = {
            "sentence_ids": [1],
            "transcript": "今天我們來介紹核心系統的三個關鍵指標。",
            "topic": "Key Metrics",
        }
        sub_units = resolve_clip_sub_units([s1], clip, total_dur=40.0)
        assert len(sub_units) == 2
        pref, cont = sub_units[0], sub_units[1]
        assert pref["composite_role"] == "prefix"
        assert pref["is_composite_splice"] is False
        assert pref["t_first"] == pytest.approx(20.00, abs=0.02)
        assert pref["t_last"] == pytest.approx(21.30, abs=0.02)
        assert pref["next_word_hard_wall"] == pytest.approx(21.32, abs=0.02)

        assert cont["composite_role"] == "continuation"
        assert cont["is_composite_splice"] is True
        assert cont["t_first"] == pytest.approx(22.60, abs=0.02)
        assert cont["t_last"] == pytest.approx(25.60, abs=0.02)
        assert cont["prev_sentence_end"] == pytest.approx(22.45, abs=0.02)

    def test_micro_prefix_below_viability_gate_is_rejected_in_favor_of_clean_restart(self):
        """Reject sub-0.55s / <4-char micro-prefixes so choppy single-word fragments are never emitted."""
        s1 = {
            "id": 1,
            "start": 5.0,
            "end": 9.0,
            "text": "今天呃今天我們來說明完整架構",
            "words": [
                {"word": "今天", "start": 5.00, "end": 5.30},
                {"word": "呃", "start": 5.32, "end": 5.55},
                {"word": "今天我們", "start": 5.70, "end": 6.50},
                {"word": "來說明", "start": 6.50, "end": 7.30},
                {"word": "完整架構", "start": 7.30, "end": 8.60},
            ],
        }
        clip = {
            "sentence_ids": [1],
            "transcript": "今天我們來說明完整架構。",
            "topic": "Clean Restart",
        }
        sub_units = resolve_clip_sub_units([s1], clip, total_dur=20.0)
        assert len(sub_units) == 1
        assert sub_units[0]["t_first"] == pytest.approx(5.70, abs=0.02)
        assert sub_units[0]["t_last"] == pytest.approx(8.60, abs=0.02)
        assert sub_units[0]["trimmed_head"] is True

    def test_cached_json_replay_preserves_composite_sub_span_bounds_and_flags(self):
        """Verify --cached-json replay keeps exact sub-span word bounds and composite metadata."""
        s2 = {
            "id": 2,
            "start": 30.0,
            "end": 35.0,
            "text": "不過沒關係核心系統運作非常穩定",
            "words": [
                {"word": "不過沒關係", "start": 30.00, "end": 31.10},
                {"word": "核心系統", "start": 31.25, "end": 32.20},
                {"word": "運作非常穩定", "start": 32.20, "end": 34.80},
            ],
        }
        cached_clip = {
            "sentence_ids": [2],
            "t_first": 31.25,
            "t_last": 34.80,
            "trimmed_head": True,
            "trimmed_tail": False,
            "composite_role": "continuation",
            "is_composite_splice": True,
            "transcript": "核心系統運作非常穩定。",
            "topic": "Cached Continuation",
        }
        sub_units = resolve_clip_sub_units([s2], cached_clip, total_dur=50.0)
        assert len(sub_units) == 1
        assert sub_units[0]["t_first"] == pytest.approx(31.25, abs=0.01)
        assert sub_units[0]["t_last"] == pytest.approx(34.80, abs=0.01)
        assert sub_units[0]["prev_sentence_end"] == pytest.approx(31.10, abs=0.01)
        assert sub_units[0]["composite_role"] == "continuation"
        assert sub_units[0]["is_composite_splice"] is True

    def test_homophone_first_word_preserved_without_truncating_opening(self):
        """Preserve t_first when the opening character differs only by an ASR/script homophone (e.g. 儘量 vs 盡量)."""
        s1 = _sentence(
            1,
            10.00,
            12.40,
            "儘量減少干擾與雜訊",
            [
                {"word": "儘", "start": 10.00, "end": 10.22},
                {"word": "量減少干擾與雜訊", "start": 10.24, "end": 12.40},
            ],
        )
        clip = {
            "sentence_ids": [1],
            "transcript": "盡量減少干擾與雜訊。",
            "topic": "Homophone Head",
        }
        sub_units = resolve_clip_sub_units([s1], clip, total_dur=30.0)
        assert len(sub_units) == 1
        assert sub_units[0]["t_first"] == pytest.approx(10.00, abs=0.01)
        assert sub_units[0]["trimmed_head"] is False

    def test_one_char_opening_word_not_dropped_when_sentence_has_mid_homophone(self):
        """Ensure sliding window never drops a 1-char opening word (e.g. 並) when a mid-sentence homophone exists."""
        s1 = _sentence(
            11,
            20.00,
            23.50,
            "並計算出反射路徑與訊號強度分佈",
            [
                {"word": "並", "start": 20.00, "end": 20.25},
                {"word": "計算出反射路徑與訊號強度", "start": 20.26, "end": 22.90},
                {"word": "分佈", "start": 22.92, "end": 23.50},
            ],
        )
        clip = {
            "sentence_ids": [11],
            "transcript": "並計算出反射路徑與信號強度",
            "topic": "Opening Conjunction Word",
        }
        sub_units = resolve_clip_sub_units([s1], clip, total_dur=40.0)
        assert len(sub_units) == 1
        assert sub_units[0]["t_first"] == pytest.approx(20.00, abs=0.01)
        assert sub_units[0]["t_last"] == pytest.approx(22.90, abs=0.02)

    def test_two_char_opening_stutter_skipped_even_with_downstream_homophone(self):
        """Skip a 2-char opening stutter (簡單簡單...) even when w_len == t_len + 2 and a downstream homophone exists."""
        s1 = _sentence(
            15,
            50.00,
            54.20,
            "簡單簡單模型就能還原無線訊號",
            [
                {"word": "簡單", "start": 50.00, "end": 50.45},
                {"word": "簡單", "start": 50.55, "end": 51.00},
                {"word": "模型就能還原無線訊號", "start": 51.02, "end": 54.20},
            ],
        )
        clip = {
            "sentence_ids": [15],
            "transcript": "簡單模型就能還原無線信號。",
            "topic": "Opening Stutter Skip",
        }
        sub_units = resolve_clip_sub_units([s1], clip, total_dur=60.0)
        assert len(sub_units) == 1
        assert sub_units[0]["t_first"] == pytest.approx(50.55, abs=0.01)
        assert sub_units[0]["t_last"] == pytest.approx(54.20, abs=0.01)
        assert sub_units[0]["trimmed_head"] is True

    def test_single_char_trailing_trim_on_long_sentence_not_bypassed_by_length_ratio(self):
        """Honor a 1-char trailing trim (e.g. removing orphan 對) even when len(tgt_norm)/len(whisper_str) > 0.95."""
        clean_body = "這項無線感測研究證明不需要安裝任何額外硬體就能在室內環境完成高精度定位與動作辨識"
        s1 = _sentence(
            1,
            5.00,
            13.80,
            clean_body + "對",
            [
                {"word": clean_body, "start": 5.00, "end": 13.10},
                {"word": "對", "start": 13.35, "end": 13.80},
            ],
        )
        clip = {
            "sentence_ids": [1],
            "transcript": clean_body + "。",
            "topic": "Long Sentence Single-Char Tail Trim",
        }
        sub_units = resolve_clip_sub_units([s1], clip, total_dur=20.0)
        assert len(sub_units) == 1
        assert sub_units[0]["t_last"] == pytest.approx(13.10, abs=0.01)
        assert sub_units[0]["trimmed_tail"] is True
        assert sub_units[0]["next_word_hard_wall"] == pytest.approx(13.349, abs=0.005)

    def test_leading_homophone_character_preserved_when_trailing_words_exist(self):
        """Preserve a 1-char leading ASR homophone (盡 vs 儘) instead of clipping the opening syllable at offset 1."""
        s1 = _sentence(
            20,
            60.00,
            65.20,
            "盡管如此這套架構依然具備極高的實務部署價值而且未來",
            [
                {"word": "盡", "start": 60.00, "end": 60.14},
                {"word": "管如此這套架構依然具備極高的實務部署價值", "start": 60.14, "end": 64.20},
                {"word": "而且未來", "start": 64.35, "end": 65.20},
            ],
        )
        clip = {
            "sentence_ids": [20],
            "transcript": "儘管如此，這套架構依然具備極高的實務部署價值。",
            "topic": "Leading Homophone Preservation",
        }
        sub_units = resolve_clip_sub_units([s1], clip, total_dur=80.0)
        assert len(sub_units) == 1
        assert sub_units[0]["t_first"] == pytest.approx(60.00, abs=0.01)
        assert sub_units[0]["t_last"] == pytest.approx(64.20, abs=0.01)
        assert sub_units[0]["trimmed_head"] is False
        assert sub_units[0]["trimmed_tail"] is True

    def test_multi_sentence_monotonic_alignment_excises_stutters_and_tail_stumbles(self):
        """Partition transcript monotonically across 3 sentences to excise internal stutters and trailing stumbles."""
        s1 = _sentence(
            31,
            100.00,
            104.80,
            "系統會將感測數據回傳給主機也就是說我們",
            [
                {"word": "系統會將感測數據回傳給主機", "start": 100.00, "end": 103.60},
                {"word": "也就是說我們", "start": 103.75, "end": 104.80},
            ],
        )
        s2 = _sentence(
            32,
            105.30,
            111.50,
            "也就是說也就是說我們的終端設備會公開廣播摘要任何人在範圍",
            [
                {"word": "也就是說", "start": 105.30, "end": 105.90},
                {"word": "也就是說我們的終端設備會公開廣播摘要", "start": 106.05, "end": 110.20},
                {"word": "任何人在範圍", "start": 110.35, "end": 111.50},
            ],
        )
        s3 = _sentence(
            33,
            112.00,
            115.60,
            "任何人在範圍內接收封包就能完成解析。",
            [
                {"word": "任何人在範圍內接收封包就能完成解析。", "start": 112.00, "end": 115.60},
            ],
        )
        clip = {
            "sentence_ids": [31, 32, 33],
            "transcript": "系統會將感測數據回傳給主機。也就是說，我們的終端設備會公開廣播摘要，任何人在範圍內接收封包就能完成解析。",
            "topic": "Multi-Sentence Stutter and Stumble Excision",
        }
        sub_units = resolve_clip_sub_units([s1, s2, s3], clip, total_dur=130.0)
        assert len(sub_units) == 3
        assert sub_units[0]["t_first"] == pytest.approx(100.00, abs=0.01)
        assert sub_units[0]["t_last"] == pytest.approx(103.60, abs=0.01)
        assert sub_units[0]["next_word_hard_wall"] == pytest.approx(103.749, abs=0.005)
        assert sub_units[1]["t_first"] == pytest.approx(106.05, abs=0.01)
        assert sub_units[1]["t_last"] == pytest.approx(110.20, abs=0.01)
        assert sub_units[1]["next_word_hard_wall"] == pytest.approx(110.349, abs=0.005)
        assert sub_units[2]["t_first"] == pytest.approx(112.00, abs=0.01)
        assert sub_units[2]["t_last"] == pytest.approx(115.60, abs=0.01)


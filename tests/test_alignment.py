import unittest

try:
    import pytest
except ImportError:
    import tests
    import pytest

from scripts.transcribe import align_clip_with_whisper


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


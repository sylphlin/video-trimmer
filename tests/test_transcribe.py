"""transcribe.py 的離線單元測試：手寫假 segments，不需要真的跑 Whisper。"""

from scripts.transcribe import (
    merge_whisper_segments_to_sentences,
    normalize_text,
)


def _seg(id_, start, end, text, words=None):
    return {"id": id_, "start": start, "end": end, "text": text, "words": words or []}


class TestNormalizeText:
    def test_strips_punctuation_and_lowercases(self):
        assert normalize_text("Hello, World!") == "helloworld"

    def test_keeps_chinese_characters(self):
        assert normalize_text("你好，世界！") == "你好世界"


class TestMergeWhisperSegmentsToSentences:
    def test_empty_input_returns_empty_list(self):
        assert merge_whisper_segments_to_sentences([]) == []

    def test_large_gap_splits_into_separate_sentences(self):
        segs = [
            _seg(1, 0.0, 1.0, "第一句"),
            _seg(2, 1.8, 2.5, "第二句"),  # gap = 0.8s >= max_gap(0.55s)
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2
        assert sentences[0]["text"] == "第一句"
        assert sentences[1]["text"] == "第二句"

    def test_small_gap_without_punctuation_merges_into_one_sentence(self):
        segs = [
            _seg(1, 0.0, 1.0, "第一句"),
            _seg(2, 1.1, 1.8, "接續內容"),  # gap = 0.1s, 無句尾標點閉合
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 1
        assert "第一句" in sentences[0]["text"]
        assert "接續內容" in sentences[0]["text"]

    def test_closure_punctuation_with_micro_gap_splits(self):
        segs = [
            _seg(1, 0.0, 1.0, "這是第一句。"),
            _seg(2, 1.25, 2.0, "這是第二句"),  # gap=0.25s >= 0.20s 且前句已標點閉合
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2

    def test_conjunction_at_next_start_prevents_split(self):
        segs = [
            _seg(1, 0.0, 1.0, "他說要來。"),
            _seg(2, 1.6, 2.0, "但是後來沒有"),  # gap=0.6s >= 0.55s，但下句為連詞開頭且 gap<0.90s
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 1

    def test_japanese_chinese_switch_forces_split(self):
        segs = [
            _seg(1, 0.0, 1.0, "這是中文"),
            _seg(2, 1.05, 1.5, "ありがとうございます"),
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2

    def test_word_level_timestamps_are_preserved_and_extended(self):
        words_a = [{"word": "你好", "start": 0.0, "end": 0.5}]
        words_b = [{"word": "嗎", "start": 1.1, "end": 1.4}]
        segs = [
            _seg(1, 0.0, 1.0, "你好", words=words_a),
            _seg(2, 1.1, 1.4, "嗎", words=words_b),
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 1
        assert sentences[0]["words"] == words_a + words_b

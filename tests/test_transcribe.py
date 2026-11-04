import unittest

from scripts.transcribe import (
    merge_whisper_segments_to_sentences,
    normalize_text,
)


def _seg(id_, start, end, text, words=None):
    return {"id": id_, "start": start, "end": end, "text": text, "words": words or []}


class TestNormalizeText(unittest.TestCase):
    def test_strips_punctuation_and_lowercases(self):
        assert normalize_text("Hello, World!") == "helloworld"

    def test_keeps_chinese_characters(self):
        assert normalize_text("你好，世界！") == "你好世界"


class TestMergeWhisperSegmentsToSentences(unittest.TestCase):
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

    def test_conjunction_after_pause_and_closure_splits_cleanly_for_llm(self):
        segs = [
            _seg(1, 0.0, 1.0, "他說要來。"),
            _seg(2, 1.6, 2.0, "但是後來沒有"),  # gap=0.6s >= 0.55s and closure punct -> split into 2 Sentence IDs
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2

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

    def test_target_speaker_change_forces_split(self):
        """場外人員雜音 (is_target_speaker=False) 與主講人 (is_target_speaker=True) 強制分句"""
        segs = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "Action", "is_target_speaker": False, "speaker_id": "SPEAKER_CREW"},
            {"id": 2, "start": 1.1, "end": 2.0, "text": "各位好", "is_target_speaker": True, "speaker_id": "SPEAKER_HOST"},
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2
        assert sentences[0]["is_target_speaker"] is False
        assert sentences[1]["is_target_speaker"] is True

    def test_speaker_turn_taking_forces_split(self):
        """訪談中說話者輪替 (SPEAKER_00 vs SPEAKER_01) 強制分句"""
        segs = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "請問你怎麼看", "is_target_speaker": True, "speaker_id": "SPEAKER_00"},
            {"id": 2, "start": 1.1, "end": 2.0, "text": "我覺得非常好", "is_target_speaker": True, "speaker_id": "SPEAKER_01"},
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2
        assert sentences[0]["speaker_id"] == "SPEAKER_00"
        assert sentences[1]["speaker_id"] == "SPEAKER_01"

    def test_long_pause_one_second_forces_split(self):
        """超過 1.0 秒長停頓強制物理切句，即使是同一說話者且帶有連詞"""
        segs = [
            {"id": 1, "start": 0.0, "end": 1.0, "text": "前面講得不太順。", "is_target_speaker": True, "speaker_id": "SPEAKER_00"},
            {"id": 2, "start": 2.1, "end": 3.0, "text": "但是重新來過", "is_target_speaker": True, "speaker_id": "SPEAKER_00"},
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2

    def test_retake_segments_are_never_merged_into_same_sentence(self):
        """講錯重講的相鄰 Whisper segments（即使停頓僅 0.25s）強制保持為獨立 Sentence ID"""
        segs = [
            _seg(1, 0.0, 2.0, "缺點是這套系統運作的前提"),
            _seg(2, 2.25, 4.5, "缺點是這套系統運作的前提是連接"),
            _seg(3, 4.70, 7.2, "缺點是這套系統運作的前提是連接比對"),
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 3
        assert sentences[0]["text"] == "缺點是這套系統運作的前提"
        assert sentences[1]["text"] == "缺點是這套系統運作的前提是連接"
        assert sentences[2]["text"] == "缺點是這套系統運作的前提是連接比對"

    def test_paused_in_sentence_restart_splits_at_physical_silence(self):
        """單一片段內若存在物理停頓 (>=0.18s) 且停頓後重複開頭前綴，於物理停頓處拆分為獨立 Sentence"""
        words = [
            {"word": "例如", "start": 0.0, "end": 0.2},
            {"word": "你", "start": 0.2, "end": 0.35},
            {"word": "自己", "start": 0.35, "end": 0.60},
            # 物理停頓 0.25s (>= 0.18s) 後重講「例如你...」
            {"word": "例如", "start": 0.85, "end": 1.05},
            {"word": "你", "start": 1.05, "end": 1.20},
            {"word": "帶著", "start": 1.20, "end": 1.45},
            {"word": "自己", "start": 1.45, "end": 1.70},
            {"word": "手機", "start": 1.70, "end": 2.00},
        ]
        segs = [_seg(1, 0.0, 2.0, "例如你自己 例如你帶著自己手機", words=words)]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2
        assert sentences[0]["text"] == "例如你自己"
        assert sentences[1]["text"] == "例如你帶著自己手機"

    def test_chirp3_multichar_cjk_words_split_at_natural_breath_pause(self):
        """Split multi-character CJK words at physical breath pauses (>= 0.35s) into separate Sentence IDs."""
        words = [
            {"word": "今天我們", "start": 0.000, "end": 0.620},
            {"word": "來介紹核心系統", "start": 0.630, "end": 1.540},
            # 0.42s breath pause before a partial pick-up or next clause with multi-char Chirp 3 tokens
            {"word": "核心系統的實測表現", "start": 1.960, "end": 3.120},
        ]
        segs = [_seg(1, 0.000, 3.120, "今天我們來介紹核心系統核心系統的實測表現", words=words)]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2
        assert sentences[0]["text"] == "今天我們來介紹核心系統"
        assert sentences[1]["text"] == "核心系統的實測表現"

    def test_is_viable_sub_span_enforces_duration_and_script_neutral_text_thresholds(self):
        from scripts.transcribe import is_viable_sub_span

        assert is_viable_sub_span("今天我們來介紹", 0.85) is True
        assert is_viable_sub_span("今天", 0.90) is False
        assert is_viable_sub_span("今天我們來介紹", 0.35) is False
        assert is_viable_sub_span("Today we introduce", 0.90) is True
        assert is_viable_sub_span("Today", 0.90) is False
        assert is_viable_sub_span("這個API", 0.70) is True

    def test_intra_segment_closure_punct_splits_orphan_confirmation_word(self):
        """Split intra-segment closure punctuation so trailing orphan confirmation words get their own Sentence ID."""
        words = [
            {"word": "打開了", "start": 0.00, "end": 0.45},
            {"word": "潘朵拉的盒子。", "start": 0.45, "end": 1.25},
            {"word": "對", "start": 1.32, "end": 1.60},
        ]
        segs = [_seg(1, 0.00, 1.60, "打開了潘朵拉的盒子。對", words=words)]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2
        assert sentences[0]["text"] == "打開了潘朵拉的盒子。"
        assert sentences[1]["text"] == "對"

    def test_clause_comma_with_micro_pause_splits_cleanly(self):
        """Split clauses ending in a comma when followed by a >= 0.18s micro-pause."""
        segs = [
            _seg(1, 0.00, 2.00, "第一階段完成訊號採集，"),
            _seg(2, 2.22, 4.10, "第二階段進行頻譜分析。"),
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2

    def test_conjunction_led_retakes_split_into_independent_sentence_ids(self):
        """Ensure repeated conjunction-led retakes across pauses are never welded into one giant Sentence ID."""
        segs = [
            _seg(1, 0.00, 1.80, "因為這次實驗環境複雜，"),
            _seg(2, 2.05, 3.90, "因為這次實驗環境有干擾，"),
            _seg(3, 4.15, 6.80, "因為這次實驗環境存在多重反射干擾。"),
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 3

    def test_inline_uppercase_acronym_without_pause_not_treated_as_slate(self):
        """Keep inline technical acronyms (zero pause) inside the surrounding sentence rather than splitting as a slate."""
        segs = [
            _seg(1, 0.00, 1.10, "系統會解析"),
            _seg(2, 1.12, 1.45, "CSI"),
            _seg(3, 1.47, 2.60, "通道狀態資訊"),
        ]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 1
        assert "CSI" in sentences[0]["text"]

    def test_comma_absorbed_pause_splits_clause_at_zero_interword_gap(self):
        """Split a clause comma when the comma token itself absorbs the >= 0.22s pause at zero inter-word gap."""
        words = [
            {"word": "第一階段", "start": 0.00, "end": 0.60},
            {"word": "完成採集,", "start": 0.60, "end": 1.60},
            {"word": "並透過", "start": 1.60, "end": 2.05},
        ]
        segs = [_seg(1, 0.00, 2.05, "第一階段完成採集,並透過", words=words)]
        sentences = merge_whisper_segments_to_sentences(segs)
        assert len(sentences) == 2
        assert sentences[0]["text"] == "第一階段完成採集,"
        assert sentences[1]["text"] == "並透過"

    def test_pre_boundary_hesitation_drag_splits_stutter_without_breaking_zero_gap_reduplication(self):
        """Split a stretched final hesitation syllable (>= 0.275s/char + >= 0.035s gap) while keeping zero-gap reduplication intact."""
        stutter_words = [
            {"word": "例", "start": 10.00, "end": 10.12},
            {"word": "如", "start": 10.12, "end": 10.24},
            {"word": "你", "start": 10.24, "end": 10.56},
            {"word": "例", "start": 10.62, "end": 10.74},
            {"word": "如", "start": 10.74, "end": 10.86},
            {"word": "你", "start": 10.86, "end": 11.00},
            {"word": "攜帶裝置進入測試區。", "start": 11.00, "end": 12.20},
        ]
        stutter_segs = [_seg(1, 10.00, 12.20, "例如你例如你攜帶裝置進入測試區。", words=stutter_words)]
        stutter_sentences = merge_whisper_segments_to_sentences(stutter_segs)
        assert len(stutter_sentences) == 2
        assert stutter_sentences[0]["text"] == "例如你"
        assert stutter_sentences[1]["text"] == "例如你攜帶裝置進入測試區。"

        redup_words = [
            {"word": "請大家", "start": 20.00, "end": 20.45},
            {"word": "仔", "start": 20.45, "end": 20.78},
            {"word": "仔", "start": 20.78, "end": 20.92},
            {"word": "細細觀察數據變化。", "start": 20.92, "end": 22.10},
        ]
        redup_segs = [_seg(1, 20.00, 22.10, "請大家仔仔細細觀察數據變化。", words=redup_words)]
        redup_sentences = merge_whisper_segments_to_sentences(redup_segs)
        assert len(redup_sentences) == 1


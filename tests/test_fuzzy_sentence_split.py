"""Unit tests for strip_filler_words, is_fuzzy_prefix_restart, and fuzzy sentence splitting in scripts/transcribe.py."""

import unittest

from scripts.transcribe import (
    is_fuzzy_prefix_restart,
    merge_whisper_segments_to_sentences,
    strip_filler_words,
)


class TestFuzzySentenceSplit(unittest.TestCase):
    def test_case_1_exact_clause_restart_with_accumulated_preceding_text(self):
        curr_1 = "完美的監控工具誕生了 德國卡爾斯魯爾理工學院的資安研究團"
        next_1 = "完美的監控工具誕生了"
        self.assertTrue(is_fuzzy_prefix_restart(curr_1, next_1))

    def test_case_2_asr_homophone_drift(self):
        curr_2 = "每個人的走路知識"
        next_2 = "每個人的走路姿態 骨架大小都會在空間中擾動"
        self.assertTrue(is_fuzzy_prefix_restart(curr_2, next_2))

    def test_case_3_leading_filler_particles(self):
        curr_3 = "完美的監控工具誕生了"
        next_3 = "好，那麼完美的監控工具誕生了，德國卡爾斯魯爾..."
        self.assertEqual(strip_filler_words("好，那麼完美的監控工具"), "完美的監控工具")
        self.assertTrue(is_fuzzy_prefix_restart(curr_3, next_3))

    def test_case_4_asr_typo_restart(self):
        curr_4 = "結果很嚇人 每位受試者留下40段證藏"
        next_4 = "結果很嚇人 每位受試者留下40段正常步行資料"
        self.assertTrue(is_fuzzy_prefix_restart(curr_4, next_4))

    def test_case_5_normal_continuous_sentence_must_not_split(self):
        curr_5 = "以前常用通道狀態資訊"
        next_5 = "CSI來分析，但通常得刷改特定網卡"
        self.assertFalse(is_fuzzy_prefix_restart(curr_5, next_5))

    def test_case_6_subject_repeated_in_second_half_must_not_decapitate_opening(self):
        """Verify that a normal sentence mentioning the same subject in the second clause does not split at word 4."""
        curr_6 = "這套系統"
        next_6 = "不需要改裝硬體，因為這套系統直接讀取路由器訊號"
        self.assertFalse(is_fuzzy_prefix_restart(curr_6, next_6))

    def test_merge_whisper_segments_splits_zero_gap_retake_restart(self):
        """Verify that Raw Seg 6 + Raw Seg 7 (NG) and Raw Seg 8 (Restart with gap=0.0) split into separate Sentence IDs."""
        raw_segments = [
            {
                "id": 6,
                "start": 34.50,
                "end": 36.82,
                "text": "完美的監控工具誕生了",
                "words": [
                    {"word": "完美的", "start": 34.50, "end": 35.40},
                    {"word": "監控工具", "start": 35.40, "end": 36.20},
                    {"word": "誕生了", "start": 36.20, "end": 36.82},
                ],
            },
            {
                "id": 7,
                "start": 36.82,
                "end": 39.16,
                "text": "德國卡爾斯魯爾理工學院的資安研究團",
                "words": [
                    {"word": "德國", "start": 36.82, "end": 37.30},
                    {"word": "卡爾斯魯爾理工學院的", "start": 37.30, "end": 38.50},
                    {"word": "資安研究團", "start": 38.50, "end": 39.16},
                ],
            },
            {
                "id": 8,
                "start": 39.16,
                "end": 42.78,
                "text": "完美的監控工具誕生了",
                "words": [
                    {"word": "完美的", "start": 39.16, "end": 40.90},
                    {"word": "監控工具", "start": 40.90, "end": 41.90},
                    {"word": "誕生了", "start": 41.90, "end": 42.78},
                ],
            },
        ]
        sentences = merge_whisper_segments_to_sentences(raw_segments)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]["orig_segment_ids"], [6, 7])
        self.assertEqual(sentences[1]["orig_segment_ids"], [8])
        self.assertAlmostEqual(sentences[1]["start"], 39.16, places=2)


if __name__ == "__main__":
    unittest.main()

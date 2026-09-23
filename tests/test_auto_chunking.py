"""Unit tests for active script windowing, auto-chunking with retake backtracking, and EDL merging."""

import unittest

from scripts.gemini_client import merge_chunked_edl
from scripts.transcribe import calculate_active_script_window, detect_chunk_boundaries


class TestAutoChunking(unittest.TestCase):
    def test_single_shot_preserved_under_limit(self):
        """Verify single-shot processing is preserved for videos at or below 660 seconds."""
        units = [
            {"sentence_id": 1, "start": 0.0, "end": 250.0, "text": "第一段完整的測試內容"},
            {"sentence_id": 2, "start": 253.0, "end": 520.0, "text": "第二段完整的測試內容"},
        ]
        chunks = detect_chunk_boundaries(units, total_duration=520.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], 0.0)
        self.assertEqual(chunks[0][1], 520.0)
        self.assertEqual(len(chunks[0][2]), 2)

    def test_splits_at_two_second_natural_pause(self):
        """Verify split boundaries fall inside silence pauses (>= 2.0s) after target_chunk_duration."""
        units = [
            {"sentence_id": 1, "start": 0.0, "end": 240.0, "text": "開場介紹智慧家庭技術發展"},
            {"sentence_id": 2, "start": 240.5, "end": 485.0, "text": "說明毫米波雷達與感測器配置"},
            # 2.4s natural pause here (>= 2.0s threshold)
            {"sentence_id": 3, "start": 487.4, "end": 750.0, "text": "進入下一個獨立主題討論隱私保護機制"},
            {"sentence_id": 4, "start": 751.0, "end": 920.0, "text": "總結未來應用場景與挑戰"},
        ]
        chunks = detect_chunk_boundaries(units, total_duration=920.0, target_chunk_duration=480.0, min_pause_duration=2.0)
        self.assertEqual(len(chunks), 2)
        # Split point should be halfway inside the [485.0, 487.4] pause -> 486.2s
        self.assertAlmostEqual(chunks[0][1], 486.2, places=2)
        self.assertAlmostEqual(chunks[1][0], 486.2, places=2)
        self.assertEqual([u["sentence_id"] for u in chunks[0][2]], [1, 2])
        self.assertEqual([u["sentence_id"] for u in chunks[1][2]], [3, 4])

    def test_backtracks_before_retake_cluster_without_exceeding_limit(self):
        """
        Verify that when a pause >= 2.0s occurs between Take 1 and Take 2 of the same sentence,
        detect_chunk_boundaries backtracks BEFORE Take 1 so that:
        1. Chunk 1 does not exceed max_chunk_duration.
        2. Both Take 1 and Take 2 stay together in Chunk 2.
        """
        units = [
            {"sentence_id": 1, "start": 0.0, "end": 410.0, "text": "前面七分鐘順利講完的第一大段內容"},
            # Safe pause (1.2s) before Take 1 starts
            {"sentence_id": 2, "start": 411.2, "end": 478.0, "text": "緊接著介紹硬體架構的連接方式"},
            # Pause 1.5s before Take 1 of the next paragraph
            {"sentence_id": 3, "start": 479.5, "end": 486.0, "text": "這套無線訊號感測系統最大的優勢是可以穿牆"},
            # Pause 2.6s (>= 2.0s) right after Take 1, followed by Take 2 of the SAME sentence!
            {"sentence_id": 4, "start": 488.6, "end": 500.0, "text": "這套無線訊號感測系統最大的優勢是可以穿牆偵測呼吸"},
            {"sentence_id": 5, "start": 501.0, "end": 880.0, "text": "最後總結實驗數據與臨床驗證結果"},
        ]
        chunks = detect_chunk_boundaries(
            units,
            total_duration=880.0,
            target_chunk_duration=480.0,
            min_pause_duration=2.0,
            max_chunk_duration=660.0,
        )
        self.assertEqual(len(chunks), 2)
        # Sentence 3 (Take 1) and Sentence 4 (Take 2) MUST both be in Chunk 2!
        chunk1_ids = [u["sentence_id"] for u in chunks[0][2]]
        chunk2_ids = [u["sentence_id"] for u in chunks[1][2]]
        self.assertEqual(chunk1_ids, [1, 2])
        self.assertEqual(chunk2_ids, [3, 4, 5])
        # Chunk 1 ends at the pause between sentence 2 and sentence 3 (478.75s < 480s), never exceeding limit
        self.assertAlmostEqual(chunks[0][1], 478.75, places=2)

    def test_calculate_active_script_window_crops_multi_episode_roll(self):
        """Verify script clause alignment isolates the target episode window with 20s padding."""
        whisper_units = [
            {"sentence_id": 1, "start": 100.0, "end": 400.0, "text": "第一集講的是區塊鏈金融科技與加密貨幣市場"},
            {"sentence_id": 2, "start": 874.0, "end": 920.0, "text": "歡迎回到節目今天我們要深入探討無線訊號穿牆感測技術"},
            {"sentence_id": 3, "start": 925.0, "end": 1362.0, "text": "研究團隊利用現有的家用路由器就能即時捕捉呼吸與心跳微動"},
            {"sentence_id": 4, "start": 1800.0, "end": 2300.0, "text": "第三集我們來聊電動車固態電池的量產時程"},
        ]
        script_text = """
        # 第二集：Wi-Fi 感測技術
        歡迎回到節目，今天我們要深入探討無線訊號穿牆感測技術。
        研究團隊利用現有的家用路由器，就能即時捕捉呼吸與心跳微動。
        """
        win_start, win_end, filtered = calculate_active_script_window(
            whisper_units=whisper_units,
            script_text=script_text,
            total_duration=2338.0,
            padding_seconds=20.0,
        )
        self.assertEqual(win_start, 854.0)
        self.assertEqual(win_end, 1382.0)
        self.assertEqual([u["sentence_id"] for u in filtered], [2, 3])

    def test_merge_chunked_edl_renumbers_monotonically(self):
        """Verify merging keeps all timestamps monotonic and renumbers clip_id sequentially."""
        chunk_results = [
            {
                "final_edl": [
                    {"clip_id": 1, "topic": "開場", "source_in": 12.0, "source_out": 45.0},
                    {"clip_id": 2, "topic": "中段", "source_in": 120.0, "source_out": 210.0},
                ]
            },
            {
                "final_edl": [
                    {"clip_id": 1, "topic": "下半場", "source_in": 510.0, "source_out": 620.0},
                ]
            },
        ]
        merged = merge_chunked_edl(chunk_results, project_title="Test Project")
        self.assertEqual(merged["project_title"], "Test Project")
        self.assertEqual(len(merged["final_edl"]), 3)
        self.assertEqual([c["clip_id"] for c in merged["final_edl"]], [1, 2, 3])
        self.assertEqual([c["source_in"] for c in merged["final_edl"]], [12.0, 120.0, 510.0])


if __name__ == "__main__":
    unittest.main()

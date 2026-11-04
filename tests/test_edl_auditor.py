"""Unit tests for edl_auditor.py (pre-render micro-window repair, timing sanitizer, and 8-dimension audit)."""

import tempfile
import unittest
from pathlib import Path

from scripts.edl_auditor import (
    _script_block_coverage_score,
    audit_edl_quality,
    deduplicate_and_sort_clips,
    detect_micro_window_anomalies,
    extract_script_blocks,
    generate_edl_audit_markdown,
    repair_edl_micro_windows,
    sanitize_refined_edl,
)
from scripts.gemini_client import format_script_blocks_for_prompt


def _sample_whisper_units():
    return [
        {"id": 1, "start": 5.0, "end": 10.0, "text": "大家好歡迎來到科學頻道", "is_target_speaker": True},
        {"id": 2, "start": 12.0, "end": 16.0, "text": "今天我們要聊量子力學的", "is_target_speaker": True},
        {"id": 3, "start": 17.0, "end": 24.0, "text": "今天我們要聊量子力學的核心概念與雙縫實驗", "is_target_speaker": True},
        {"id": 4, "start": 30.0, "end": 40.0, "text": "雙縫實驗證明了微觀粒子同時具備波動性與粒子性", "is_target_speaker": True},
        {"id": 5, "start": 45.0, "end": 55.0, "text": "最後請記得訂閱我們的頻道並開啟小鈴鐺", "is_target_speaker": True},
    ]


class TestScriptParserAndCoverage(unittest.TestCase):
    def test_script_parser_filters_metadata_frontmatter_and_aligns_with_prompt_formatter(self):
        script_text = (
            "---\n"
            "title: Wi-Fi 感測技術\n"
            "author: Tech Team\n"
            "---\n"
            "# 第一段：開場\n"
            "標題：不用攝影機也能穿牆看見你？Wi-Fi 感測技術解密\n"
            "主題：無線訊號感測\n"
            "大綱：介紹 Wi-Fi CSI 原理與隱私影響\n"
            "內文：\n"
            "（鏡頭拉近）\n"
            "1. 你有想過家裡的無線路由器，其實可以變成隱形雷達嗎？\n"
            "內文：當無線電波在房間裡反射時，人體移動會改變訊號相位。\n"
        )
        blocks = extract_script_blocks(script_text)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["label"], "[Script Block 01]")
        self.assertIn("你有想過家裡的無線路由器", blocks[0]["raw_text"])
        self.assertEqual(blocks[1]["label"], "[Script Block 02]")
        self.assertEqual(blocks[1]["raw_text"], "當無線電波在房間裡反射時，人體移動會改變訊號相位。")

        formatted = format_script_blocks_for_prompt(script_text)
        formatted_lines = [line for line in formatted.splitlines() if line.strip()]
        self.assertEqual(len(formatted_lines), 2)
        self.assertTrue(formatted_lines[0].startswith("[Script Block 01]"))
        self.assertTrue(formatted_lines[1].startswith("[Script Block 02]"))

    def test_script_coverage_score_handles_asr_typos_and_rejects_short_ng_fragments(self):
        block_norm = "你有想過家裡的無線路由器其實可以變成隱形雷達嗎"
        # Full EDL transcript with minor ASR wording differences ("家裡面", "能") embedded in a long text
        long_selected_norm = (
            "大家好歡迎回到我們的科技專欄今天要分享一個非常特別的研究"
            "你有想過家裡面的無線路由器其實能變成隱形雷達嗎"
            "接下來我們來看看研究團隊是怎麼做到的以及背後的天線原理"
        )
        cov_good = _script_block_coverage_score(block_norm, long_selected_norm)
        self.assertGreaterEqual(cov_good, 0.80)

        # Short aborted NG fragment (only 6 chars spoken before stumbling)
        short_ng_norm = "你有想過家裡"
        cov_ng = _script_block_coverage_score(block_norm, short_ng_norm)
        self.assertLess(cov_ng, 0.35)


class TestMicroWindowAnomalyDetectionAndRepair(unittest.TestCase):
    def test_detects_potential_residual_retake_and_repairs_window(self):
        whisper_units = _sample_whisper_units()
        model_edl = {
            "project_title": "Test",
            "final_edl": [
                {"clip_id": 1, "topic": "開場", "sentence_ids": [1], "source_in": 5.0, "source_out": 10.0, "transcript": "大家好歡迎來到科學頻道"},
                {"clip_id": 2, "topic": "NG句", "sentence_ids": [2], "source_in": 12.0, "source_out": 16.0, "transcript": "今天我們要聊量子力學的"},
                {"clip_id": 3, "topic": "正式句", "sentence_ids": [3], "source_in": 17.0, "source_out": 24.0, "transcript": "今天我們要聊量子力學的核心概念與雙縫實驗"},
            ],
        }
        anomalies = detect_micro_window_anomalies(model_edl, whisper_units, total_dur=60.0)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["type"], "POTENTIAL_RESIDUAL_RETAKE")
        self.assertEqual(anomalies[0]["replace_range"], (1, 3))

        captured_calls = []

        def _fake_window_inference(win_units, win_start, win_end, start_offset, end_offset, anomaly):
            captured_calls.append((start_offset, end_offset, [u["id"] for u in win_units]))
            return {
                "final_edl": [
                    {
                        "clip_id": 1,
                        "topic": "正式句修復",
                        "sentence_ids": [3],
                        "source_in": 17.0,
                        "source_out": 24.0,
                        "transcript": "今天我們要聊量子力學的核心概念與雙縫實驗",
                    }
                ]
            }

        repaired_edl, repairs = repair_edl_micro_windows(
            model_edl=model_edl,
            whisper_units=whisper_units,
            total_dur=60.0,
            run_window_inference_fn=_fake_window_inference,
        )
        self.assertEqual(len(repairs), 1)
        self.assertEqual(len(captured_calls), 1)
        self.assertEqual(len(repaired_edl["final_edl"]), 2)
        self.assertEqual(repaired_edl["final_edl"][0]["sentence_ids"], [1])
        self.assertEqual(repaired_edl["final_edl"][1]["sentence_ids"], [3])

    def test_detects_and_rescues_missing_script_block_in_mode_a(self):
        whisper_units = _sample_whisper_units()
        script_text = (
            "大家好歡迎來到科學頻道\n"
            "今天我們要聊量子力學的核心概念與雙縫實驗\n"
            "雙縫實驗證明了微觀粒子同時具備波動性與粒子性\n"
            "最後請記得訂閱我們的頻道並開啟小鈴鐺\n"
        )
        # Omit sentence 4 (Script Block 03) from initial model_edl
        model_edl = {
            "project_title": "Mode A Test",
            "final_edl": [
                {"clip_id": 1, "topic": "開場", "sentence_ids": [1], "source_in": 5.0, "source_out": 10.0, "transcript": "大家好歡迎來到科學頻道"},
                {"clip_id": 2, "topic": "主旨", "sentence_ids": [3], "source_in": 17.0, "source_out": 24.0, "transcript": "今天我們要聊量子力學的核心概念與雙縫實驗"},
                {"clip_id": 3, "topic": "結尾", "sentence_ids": [5], "source_in": 45.0, "source_out": 55.0, "transcript": "最後請記得訂閱我們的頻道並開啟小鈴鐺"},
            ],
        }
        anomalies = detect_micro_window_anomalies(model_edl, whisper_units, total_dur=60.0, script_text=script_text)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["type"], "MISSING_SCRIPT_BLOCK")
        self.assertEqual(anomalies[0]["script_block"]["label"], "[Script Block 03]")

        def _fake_rescue_inference(win_units, win_start, win_end, start_offset, end_offset, anomaly):
            return {
                "final_edl": [
                    {
                        "clip_id": 1,
                        "topic": "雙縫實驗",
                        "sentence_ids": [4],
                        "source_in": 30.0,
                        "source_out": 40.0,
                        "transcript": "雙縫實驗證明了微觀粒子同時具備波動性與粒子性",
                    }
                ]
            }

        repaired_edl, repairs = repair_edl_micro_windows(
            model_edl=model_edl,
            whisper_units=whisper_units,
            total_dur=60.0,
            run_window_inference_fn=_fake_rescue_inference,
            script_text=script_text,
        )
        self.assertEqual(len(repairs), 1)
        self.assertEqual([c["sentence_ids"] for c in repaired_edl["final_edl"]], [[1], [3], [4], [5]])

    def test_deduplicate_and_sort_clips_enforces_last_take_wins_per_script_block(self):
        whisper_units = [
            {"id": 1, "start": 10.0, "end": 18.0, "text": "雙縫實驗證明了微觀粒子同時具備波動性與粒子性", "is_target_speaker": True},
            {"id": 2, "start": 35.0, "end": 43.0, "text": "雙縫實驗證明了微觀粒子同時具備波動性與粒子性", "is_target_speaker": True},
            {"id": 3, "start": 45.0, "end": 55.0, "text": "最後請記得訂閱我們的頻道並開啟小鈴鐺", "is_target_speaker": True},
        ]
        script_text = (
            "雙縫實驗證明了微觀粒子同時具備波動性與粒子性\n"
            "最後請記得訂閱我們的頻道並開啟小鈴鐺\n"
        )
        # Out-of-order and duplicate takes for Script Block 01 (id=1 at 10s and id=2 at 35s)
        raw_clips = [
            {"clip_id": 1, "topic": "結尾", "sentence_ids": [3], "source_in": 45.0, "source_out": 55.0, "transcript": "最後請記得訂閱我們的頻道並開啟小鈴鐺"},
            {"clip_id": 2, "topic": "雙縫Take2", "sentence_ids": [2], "source_in": 35.0, "source_out": 43.0, "transcript": "雙縫實驗證明了微觀粒子同時具備波動性與粒子性"},
            {"clip_id": 3, "topic": "雙縫Take1", "sentence_ids": [1], "source_in": 10.0, "source_out": 18.0, "transcript": "雙縫實驗證明了微觀粒子同時具備波動性與粒子性"},
        ]
        deduped = deduplicate_and_sort_clips(raw_clips, whisper_units, script_text=script_text)
        self.assertEqual([c["sentence_ids"] for c in deduped], [[2], [3]])


class TestSanitizeRefinedEdl(unittest.TestCase):
    def test_resolves_overlaps_restores_plosive_tail_and_merges_micro_clips(self):
        raw_refined = [
            {
                "clip_id": 1,
                "topic": "A",
                "sentence_ids": [1],
                "t_first": 5.0,
                "t_last": 10.10,
                "source_in": 4.92,
                "source_out": 10.00,  # Truncates t_last (10.10) -> should restore to >= 10.10
                "duration": 5.08,
                "cps": 4.0,
                "in_margin": 0.08,
                "out_margin": 0.0,
                "transcript": "第一段完整的台詞內容",
            },
            {
                "clip_id": 2,
                "topic": "B",
                "sentence_ids": [2],
                "t_first": 10.12,
                "t_last": 10.35,
                "source_in": 10.05,  # Overlaps Clip 1 source_out AND is a <0.45s micro-clip
                "source_out": 10.38,
                "duration": 0.33,
                "cps": 3.5,
                "in_margin": 0.07,
                "out_margin": 0.03,
                "transcript": "補充詞",
            },
        ]
        sanitized, stats = sanitize_refined_edl(raw_refined, total_dur=30.0)
        self.assertEqual(stats["plosive_tails_restored"], 1)
        self.assertEqual(stats["overlaps_resolved"], 1)
        self.assertEqual(stats["micro_clips_merged"], 1)
        self.assertEqual(len(sanitized), 1)
        self.assertEqual(sanitized[0]["sentence_ids"], [1, 2])
        self.assertGreaterEqual(sanitized[0]["source_out"], 10.35)

    def test_prevents_negative_duration_on_out_of_order_or_contained_clips(self):
        # Simulate an out-of-order earlier clip [10.0, 20.0] appended after [50.0, 90.0]
        # and a nested inner clip [55.0, 65.0] contained inside [50.0, 90.0]
        raw_refined = [
            {
                "clip_id": 1,
                "topic": "Main",
                "sentence_ids": [5, 6],
                "t_first": 50.1,
                "t_last": 89.8,
                "source_in": 50.0,
                "source_out": 90.0,
                "duration": 40.0,
                "cps": 4.0,
                "transcript": "後段完整的主講內容",
            },
            {
                "clip_id": 2,
                "topic": "ContainedInner",
                "sentence_ids": [5],
                "t_first": 55.1,
                "t_last": 64.9,
                "source_in": 55.0,
                "source_out": 65.0,
                "duration": 10.0,
                "cps": 4.0,
                "transcript": "被包裹的重疊子片段",
            },
            {
                "clip_id": 3,
                "topic": "OutOfOrderEarlier",
                "sentence_ids": [1],
                "t_first": 10.1,
                "t_last": 19.9,
                "source_in": 10.0,
                "source_out": 20.0,
                "duration": 10.0,
                "cps": 4.0,
                "transcript": "前段順序顛倒的片段",
            },
        ]
        sanitized, _ = sanitize_refined_edl(raw_refined, total_dur=120.0)
        self.assertEqual(len(sanitized), 2)
        for i, c in enumerate(sanitized):
            self.assertGreater(c["source_out"], c["source_in"])
            self.assertGreater(c["duration"], 0.0)
            if i > 0:
                self.assertGreaterEqual(c["source_in"], sanitized[i - 1]["source_out"])


class TestAuditEdlQuality(unittest.TestCase):
    def test_clean_edl_passes_quality_gate_and_generates_markdown(self):
        whisper_units = _sample_whisper_units()
        refined_edl = [
            {
                "clip_id": 1,
                "topic": "開場",
                "sentence_ids": [1],
                "t_first": 5.0,
                "t_last": 10.0,
                "source_in": 4.92,
                "source_out": 10.12,
                "duration": 5.20,
                "cps": 4.2,
                "in_margin": 0.08,
                "out_margin": 0.12,
                "transcript": "大家好歡迎來到科學頻道",
            },
            {
                "clip_id": 2,
                "topic": "主題",
                "sentence_ids": [3],
                "t_first": 17.0,
                "t_last": 24.0,
                "source_in": 16.92,
                "source_out": 24.12,
                "duration": 7.20,
                "cps": 4.5,
                "in_margin": 0.08,
                "out_margin": 0.12,
                "transcript": "今天我們要聊量子力學的核心概念與雙縫實驗",
            },
        ]
        report = audit_edl_quality(
            refined_edl=refined_edl,
            whisper_units=whisper_units,
            total_dur=60.0,
            video_path=Path("/tmp/raw_footage.mp4"),
        )
        verdict = report["agent_verdict"]
        self.assertTrue(verdict["pass_quality_gate"])
        self.assertEqual(verdict["suggested_action"], "DELIVER")
        self.assertEqual(verdict["fatal_violations"], [])

        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "report.md"
            generate_edl_audit_markdown(report, md_path)
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("## 1. Executive Summary & Agent Verdict", md_text)
            self.assertIn("`True`", md_text)

    def test_residual_retake_fails_quality_gate_with_one_shot_remediate(self):
        whisper_units = _sample_whisper_units()
        refined_edl = [
            {
                "clip_id": 1,
                "topic": "NG",
                "sentence_ids": [2],
                "t_first": 12.0,
                "t_last": 16.0,
                "source_in": 11.92,
                "source_out": 16.10,
                "duration": 4.18,
                "cps": 3.8,
                "in_margin": 0.08,
                "out_margin": 0.10,
                "transcript": "今天我們要聊量子力學的",
            },
            {
                "clip_id": 2,
                "topic": "OK",
                "sentence_ids": [3],
                "t_first": 17.0,
                "t_last": 24.0,
                "source_in": 16.92,
                "source_out": 24.10,
                "duration": 7.18,
                "cps": 4.2,
                "in_margin": 0.08,
                "out_margin": 0.10,
                "transcript": "今天我們要聊量子力學的核心概念與雙縫實驗",
            },
        ]
        report = audit_edl_quality(
            refined_edl=refined_edl,
            whisper_units=whisper_units,
            total_dur=60.0,
            video_path=Path("/tmp/raw_footage.mp4"),
        )
        verdict = report["agent_verdict"]
        self.assertFalse(verdict["pass_quality_gate"])
        self.assertEqual(verdict["suggested_action"], "ONE_SHOT_REMEDIATE")
        self.assertTrue(any("residual retake" in v for v in verdict["fatal_violations"]))

    def test_tc01_tail_head_retake_detected_in_anomaly_scanner_and_audit_report(self):
        """TC-01: Detect tail-to-head stumble overlap (including trailing typo) for Gemini micro-window and final audit."""
        whisper_units = [
            {
                "id": 6,
                "start": 33.0,
                "end": 51.6,
                "text": "完美的監控工具其實已經在你家裡了對就是WiFi你大概也聽過WiFi訊號撞到人體會產生細微摔",
                "is_target_speaker": True,
            },
            {
                "id": 7,
                "start": 51.7,
                "end": 62.0,
                "text": "對就是你家路由器發出的訊號你大概也聽過WiFi訊號撞到人體會出現細微衰減與相位變化",
                "is_target_speaker": True,
            },
        ]
        model_edl = {
            "project_title": "Tail-Head Test",
            "final_edl": [
                {
                    "clip_id": 1,
                    "topic": "Opening",
                    "sentence_ids": [6],
                    "source_in": 33.0,
                    "source_out": 51.6,
                    "transcript": "完美的監控工具其實已經在你家裡了對就是WiFi你大概也聽過WiFi訊號撞到人體會產生細微摔",
                },
                {
                    "clip_id": 2,
                    "topic": "Explanation",
                    "sentence_ids": [7],
                    "source_in": 51.7,
                    "source_out": 62.0,
                    "transcript": "對就是你家路由器發出的訊號你大概也聽過WiFi訊號撞到人體會出現細微衰減與相位變化",
                },
            ],
        }
        anomalies = detect_micro_window_anomalies(model_edl, whisper_units, total_dur=70.0)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["type"], "TAIL_HEAD_RETAKE")
        self.assertEqual(anomalies[0]["replace_range"], (0, 2))

        report = audit_edl_quality(
            refined_edl=[
                dict(model_edl["final_edl"][0], duration=18.6, cps=4.0, in_margin=0.08, out_margin=0.10),
                dict(model_edl["final_edl"][1], duration=10.3, cps=4.0, in_margin=0.08, out_margin=0.10),
            ],
            whisper_units=whisper_units,
            total_dur=70.0,
            video_path=Path("/tmp/raw_footage.mp4"),
        )
        self.assertFalse(report["agent_verdict"]["pass_quality_gate"])
        self.assertTrue(any("ERR_TAIL_HEAD_RETAKE" in v for v in report["agent_verdict"]["fatal_violations"]))

    def test_tc02_intra_clip_repeat_detected_in_anomaly_scanner_and_audit_report(self):
        """TC-02: Detect intra-clip immediate repeat (A + A + B) for Gemini micro-window and final audit."""
        whisper_units = [
            {
                "id": 15,
                "start": 100.0,
                "end": 112.0,
                "text": "研究者推測BFI本來就是壓縮研究者推測BFI本來就是壓縮後的資訊",
                "is_target_speaker": True,
            }
        ]
        model_edl = {
            "project_title": "Intra-Repeat Test",
            "final_edl": [
                {
                    "clip_id": 1,
                    "topic": "BFI",
                    "sentence_ids": [15],
                    "source_in": 100.0,
                    "source_out": 112.0,
                    "transcript": "研究者推測BFI本來就是壓縮研究者推測BFI本來就是壓縮後的資訊",
                }
            ],
        }
        anomalies = detect_micro_window_anomalies(model_edl, whisper_units, total_dur=150.0)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["type"], "INTRA_CLIP_REPEAT")
        self.assertEqual(anomalies[0]["replace_range"], (0, 1))

        report = audit_edl_quality(
            refined_edl=[dict(model_edl["final_edl"][0], duration=12.0, cps=3.5, in_margin=0.08, out_margin=0.10)],
            whisper_units=whisper_units,
            total_dur=150.0,
            video_path=Path("/tmp/raw_footage.mp4"),
        )
        self.assertFalse(report["agent_verdict"]["pass_quality_gate"])
        self.assertTrue(any("ERR_INTRA_CLIP_REPEAT" in v for v in report["agent_verdict"]["fatal_violations"]))

    def test_tc03_script_take_collision_detected_and_repaired(self):
        """TC-03: Detect partial-take + full-take collision on the same Mode A Script Block (>= 0.45 coverage)."""
        script_text = "這代表駭客不需要破解你的WiFi密碼，也不需要植入任何木馬程式，只要在門外接收公開訊號就能追蹤。\n"
        whisper_units = [
            {
                "id": 20,
                "start": 200.0,
                "end": 206.0,
                "text": "這代表駭客不需要破解你的WiFi密碼也不需要植入任何木馬",
                "is_target_speaker": True,
            },
            {
                "id": 21,
                "start": 207.0,
                "end": 216.0,
                "text": "這代表駭客不需要破解你的WiFi密碼也不需要植入任何木馬程式只要在門外接收公開訊號就能追蹤",
                "is_target_speaker": True,
            },
        ]
        model_edl = {
            "project_title": "Collision Test",
            "final_edl": [
                {
                    "clip_id": 1,
                    "topic": "Partial NG",
                    "sentence_ids": [20],
                    "source_in": 200.0,
                    "source_out": 206.0,
                    "transcript": "這代表駭客不需要破解你的WiFi密碼也不需要植入任何木馬",
                },
                {
                    "clip_id": 2,
                    "topic": "Full Take",
                    "sentence_ids": [21],
                    "source_in": 207.0,
                    "source_out": 216.0,
                    "transcript": "這代表駭客不需要破解你的WiFi密碼也不需要植入任何木馬程式只要在門外接收公開訊號就能追蹤",
                },
            ],
        }
        anomalies = detect_micro_window_anomalies(
            model_edl, whisper_units, total_dur=250.0, script_text=script_text
        )
        self.assertTrue(any(a["type"] in ("POTENTIAL_RESIDUAL_RETAKE", "SCRIPT_TAKE_COLLISION") for a in anomalies))

        report = audit_edl_quality(
            refined_edl=[
                dict(model_edl["final_edl"][0], duration=6.0, cps=4.0, in_margin=0.08, out_margin=0.10),
                dict(model_edl["final_edl"][1], duration=9.0, cps=4.5, in_margin=0.08, out_margin=0.10),
            ],
            whisper_units=whisper_units,
            total_dur=250.0,
            video_path=Path("/tmp/raw_footage.mp4"),
            script_text=script_text,
        )
        self.assertFalse(report["agent_verdict"]["pass_quality_gate"])
        self.assertTrue(any("ERR_SCRIPT_TAKE_COLLISION" in v for v in report["agent_verdict"]["fatal_violations"]))
        self.assertEqual(report["dimensions"]["4_script_coverage"]["script_take_collision_count"], 1)

    def test_composite_prefix_occlusion_keeps_missing_prefix_while_full_retake_wins(self):
        """Verify Partial Tail Pick-Up keeps Take 1's opening prefix while Full Retake drops Take 1 completely."""
        script_text = "今天我們來深入探討核心系統的效能與延遲表現非常出色。"
        whisper_units = [
            {
                "id": 1,
                "start": 10.0,
                "end": 14.0,
                "text": "今天我們來深入探討核心系統的效能",
                "words": [
                    {"word": "今天我們來深入探討", "start": 10.0, "end": 12.1},
                    {"word": "核心系統的效能", "start": 12.2, "end": 14.0},
                ],
            },
            {
                "id": 2,
                "start": 15.0,
                "end": 19.5,
                "text": "核心系統的效能與延遲表現非常出色。",
                "words": [
                    {"word": "核心系統的效能", "start": 15.0, "end": 16.8},
                    {"word": "與延遲表現非常出色。", "start": 16.8, "end": 19.5},
                ],
            },
        ]
        partial_pickup_clips = [
            {
                "clip_id": 1,
                "topic": "Opening Attempt",
                "sentence_ids": [1],
                "source_in": 10.0,
                "source_out": 14.0,
                "transcript": "今天我們來深入探討核心系統的效能",
            },
            {
                "clip_id": 2,
                "topic": "Tail Pick-up",
                "sentence_ids": [2],
                "source_in": 15.0,
                "source_out": 19.5,
                "transcript": "核心系統的效能與延遲表現非常出色。",
            },
        ]
        deduped = deduplicate_and_sort_clips(partial_pickup_clips, whisper_units, script_text=script_text)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["composite_role"], "prefix")
        self.assertEqual(deduped[0]["transcript"], "今天我們來深入探討")
        self.assertEqual(deduped[1]["composite_role"], "continuation")
        self.assertTrue(deduped[1]["is_composite_splice"])

        # Audit should pass with 0 script take collisions and 1 composite splice
        refined = [
            {
                "clip_id": 1,
                "topic": "Prefix",
                "sentence_ids": [1],
                "t_first": 10.0,
                "t_last": 12.1,
                "source_in": 9.92,
                "source_out": 12.14,
                "duration": 2.22,
                "cps": 4.05,
                "in_margin": 0.08,
                "out_margin": 0.04,
                "transcript": "今天我們來深入探討",
                "asr_transcript": "今天我們來深入探討",
                "composite_role": "prefix",
                "is_composite_splice": False,
                "trimmed_tail": True,
            },
            {
                "clip_id": 2,
                "topic": "Continuation",
                "sentence_ids": [2],
                "t_first": 15.0,
                "t_last": 19.5,
                "source_in": 14.96,
                "source_out": 19.60,
                "duration": 4.64,
                "cps": 3.45,
                "in_margin": 0.04,
                "out_margin": 0.10,
                "transcript": "核心系統的效能與延遲表現非常出色。",
                "asr_transcript": "核心系統的效能與延遲表現非常出色。",
                "composite_role": "continuation",
                "is_composite_splice": True,
            },
        ]
        report = audit_edl_quality(
            refined_edl=refined,
            whisper_units=whisper_units,
            total_dur=30.0,
            video_path=Path("/tmp/raw_footage.mp4"),
            script_text=script_text,
            asr_engine="whisper",
            language="zh-TW",
            whisper_model="small",
            punch_in_composite=True,
        )
        self.assertTrue(report["agent_verdict"]["pass_quality_gate"])
        self.assertEqual(report["dimensions"]["1_acoustic_lock"]["acoustic_lock_rate_pct"], 100.0)
        self.assertEqual(report["dimensions"]["4_script_coverage"]["script_take_collision_count"], 0)
        self.assertEqual(report["summary"]["composite_splices_count"], 1)

    def test_detect_micro_window_anomalies_flags_partial_script_block_backfill(self):
        """Flag PARTIAL_SCRIPT_BLOCK_BACKFILL when only a tail pick-up was selected and an earlier take has the missing opening."""
        from scripts.edl_auditor import detect_micro_window_anomalies

        script_text = "今天我們來深入探討整個分散式儲存架構的核心設計理念與容錯機制。"
        whisper_units = [
            {"id": 1, "start": 10.0, "end": 14.5, "text": "今天我們來深入探討整個分散式儲存架構"},
            {"id": 2, "start": 16.0, "end": 20.5, "text": "整個分散式儲存架構的核心設計理念與容錯機制。"},
        ]
        clips = [
            {
                "clip_id": 1,
                "topic": "Tail Only",
                "sentence_ids": [2],
                "source_in": 16.0,
                "source_out": 20.5,
                "transcript": "整個分散式儲存架構的核心設計理念與容錯機制。",
            }
        ]
        anomalies = detect_micro_window_anomalies({"final_edl": clips}, whisper_units, total_dur=30.0, script_text=script_text)
        self.assertTrue(any(a["type"] == "PARTIAL_SCRIPT_BLOCK_BACKFILL" for a in anomalies))

    def test_sanitize_refined_edl_respects_next_word_hard_wall_and_preserves_composite_splices(self):
        """Ensure plosive tail restoration never crosses next_word_hard_wall and micro-clip merge skips composite seams."""
        raw_refined = [
            {
                "clip_id": 1,
                "topic": "Prefix",
                "sentence_ids": [1],
                "t_first": 5.000,
                "t_last": 6.100,
                "source_in": 4.920,
                "source_out": 6.080,  # slightly before t_last, with hard wall at 6.115
                "duration": 1.160,
                "transcript": "今天我們介紹",
                "composite_role": "prefix",
                "trimmed_tail": True,
                "next_word_hard_wall": 6.115,
            },
            {
                "clip_id": 2,
                "topic": "Short Continuation",
                "sentence_ids": [1],
                "t_first": 6.300,
                "t_last": 6.680,
                "source_in": 6.260,
                "source_out": 6.700,  # 0.44s (< 0.45s), gap 0.145s (< 0.35s), but marked as composite splice
                "duration": 0.440,
                "transcript": "核心架構",
                "composite_role": "continuation",
                "is_composite_splice": True,
            },
        ]
        sanitized, stats = sanitize_refined_edl(raw_refined, total_dur=20.0)
        self.assertEqual(len(sanitized), 2)
        self.assertLessEqual(sanitized[0]["source_out"], 6.115)
        self.assertEqual(stats["micro_clips_merged"], 0)

    def test_script_whitelisted_phrase_and_mid_sentence_echo_never_trigger_false_tail_head_retake(self):
        """Ensure repeated script terms or mid-sentence echoes (longest.b > 6) are neither truncated nor flagged."""
        script_text = (
            "波束成形技術會把無線能量精準送到裝置。\n"
            "這種為了讓連線更穩的設計把訊號精準送到裝置，卻也留下了反射特徵。\n"
        )
        whisper_units = [
            {"id": 5, "start": 20.0, "end": 25.0, "text": "波束成形技術會把無線能量精準送到裝置", "is_target_speaker": True},
            {
                "id": 6,
                "start": 25.5,
                "end": 32.0,
                "text": "這種為了讓連線更穩的設計把訊號精準送到裝置卻也留下了反射特徵",
                "is_target_speaker": True,
            },
        ]
        clips = [
            {
                "clip_id": 1,
                "topic": "Beamforming",
                "sentence_ids": [5],
                "source_in": 20.0,
                "source_out": 25.0,
                "transcript": "波束成形技術會把無線能量精準送到裝置。",
            },
            {
                "clip_id": 2,
                "topic": "Reflection",
                "sentence_ids": [6],
                "source_in": 25.5,
                "source_out": 32.0,
                "transcript": "這種為了讓連線更穩的設計把訊號精準送到裝置，卻也留下了反射特徵。",
            },
        ]
        deduped = deduplicate_and_sort_clips(clips, whisper_units, script_text=script_text)
        self.assertEqual(len(deduped), 2)
        self.assertIn("精準送到裝置", deduped[0]["transcript"])
        self.assertFalse(deduped[0].get("trimmed_tail", False))

        anomalies = detect_micro_window_anomalies({"final_edl": deduped}, whisper_units, total_dur=40.0, script_text=script_text)
        self.assertEqual(anomalies, [])

        refined = [
            dict(deduped[0], t_first=20.0, t_last=25.0, duration=5.0, cps=3.8, in_margin=0.08, out_margin=0.10),
            dict(deduped[1], t_first=25.5, t_last=32.0, duration=6.5, cps=4.6, in_margin=0.08, out_margin=0.10),
        ]
        report = audit_edl_quality(
            refined_edl=refined,
            whisper_units=whisper_units,
            total_dur=40.0,
            video_path=Path("/tmp/raw_footage.mp4"),
            script_text=script_text,
        )
        self.assertTrue(report["agent_verdict"]["pass_quality_gate"])

    def test_composite_prefix_occlusion_extends_pre_overlap_subject_and_preserves_continuation_head(self):
        """Trim dangling pre-overlap subject on composite prefix when continuation restarts with fuzzy subject + verb."""
        whisper_units = [
            {"id": 15, "start": 40.0, "end": 46.0, "text": "取得原始數據之後研究團隊將信號轉換", "is_target_speaker": True},
            {"id": 16, "start": 46.8, "end": 53.0, "text": "研究團隊把信號轉換成高解析度頻譜圖", "is_target_speaker": True},
        ]
        clips = [
            {
                "clip_id": 1,
                "topic": "Data",
                "sentence_ids": [15],
                "source_in": 40.0,
                "source_out": 46.0,
                "composite_role": "prefix",
                "transcript": "取得原始數據之後，研究團隊將信號轉換",
            },
            {
                "clip_id": 2,
                "topic": "Spectrogram",
                "sentence_ids": [16],
                "source_in": 46.8,
                "source_out": 53.0,
                "composite_role": "continuation",
                "is_composite_splice": True,
                "transcript": "研究團隊把信號轉換成高解析度頻譜圖。",
            },
        ]
        deduped = deduplicate_and_sort_clips(clips, whisper_units)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["transcript"], "取得原始數據之後")
        self.assertNotIn("研究團隊", deduped[0]["transcript"])
        self.assertFalse(deduped[1]["trimmed_head"])

    def test_deduplicate_preserves_parallelism_and_llm_verified_window_downgrades_audit_to_warning(self):
        """Do not blindly delete non-overlapping parallel clips in Python; honor llm_verified_window in audit."""
        whisper_units = [
            {"id": 1, "start": 10.0, "end": 15.0, "text": "我們可以看到第一組實驗數據表現非常穩定", "is_target_speaker": True},
            {"id": 2, "start": 15.6, "end": 21.0, "text": "我們可以看到第二組實驗數據同樣非常出色", "is_target_speaker": True},
        ]
        initial_edl = {
            "final_edl": [
                {"clip_id": 1, "topic": "Group 1", "sentence_ids": [1], "source_in": 10.0, "source_out": 15.0, "transcript": "我們可以看到第一組實驗數據表現非常穩定"},
                {"clip_id": 2, "topic": "Group 2", "sentence_ids": [2], "source_in": 15.6, "source_out": 21.0, "transcript": "我們可以看到第二組實驗數據同樣非常出色"},
            ]
        }

        # Gemini micro-window review listens to audio and intentionally retains both parallel sentences
        def _keep_both_parallel(win_units, win_start, win_end, start_offset, end_offset, anomaly):
            return {"final_edl": initial_edl["final_edl"]}

        repaired_edl, repairs = repair_edl_micro_windows(
            model_edl=initial_edl,
            whisper_units=whisper_units,
            total_dur=30.0,
            run_window_inference_fn=_keep_both_parallel,
        )
        self.assertEqual(len(repairs), 1)
        self.assertEqual(len(repaired_edl["final_edl"]), 2)
        self.assertTrue(all(c.get("llm_verified_window") for c in repaired_edl["final_edl"]))

        refined = [
            dict(repaired_edl["final_edl"][0], t_first=10.0, t_last=15.0, duration=5.0, cps=3.8, in_margin=0.08, out_margin=0.10),
            dict(repaired_edl["final_edl"][1], t_first=15.6, t_last=21.0, duration=5.4, cps=3.5, in_margin=0.08, out_margin=0.10),
        ]
        report = audit_edl_quality(
            refined_edl=refined,
            whisper_units=whisper_units,
            total_dur=30.0,
            video_path=Path("/tmp/raw_footage.mp4"),
        )
        self.assertTrue(report["agent_verdict"]["pass_quality_gate"])
        self.assertEqual(report["agent_verdict"]["fatal_violations"], [])
        self.assertTrue(any("LLM-verified window" in w for w in report["agent_verdict"]["warnings"]))

    def test_detect_micro_window_anomalies_coalesces_overlapping_multi_clip_defects(self):
        """Merge overlapping adjacent anomalies and punctuated intra-clip stutters into a single unified window."""
        whisper_units = [
            {"id": 14, "start": 80.0, "end": 85.0, "text": "前段結論非常明確因為系統架構", "is_target_speaker": True},
            {"id": 15, "start": 85.4, "end": 92.0, "text": "因為啊，因為系統架構已經完成模組化設計並將訊號轉換", "is_target_speaker": True},
            {"id": 16, "start": 92.5, "end": 98.0, "text": "將訊號轉換成標準向量格式輸出", "is_target_speaker": True},
        ]
        model_edl = {
            "final_edl": [
                {"clip_id": 1, "topic": "A", "sentence_ids": [14], "source_in": 80.0, "source_out": 85.0, "transcript": "前段結論非常明確因為系統架構"},
                {"clip_id": 2, "topic": "B", "sentence_ids": [15], "source_in": 85.4, "source_out": 92.0, "transcript": "因為啊，因為系統架構已經完成模組化設計並將訊號轉換"},
                {"clip_id": 3, "topic": "C", "sentence_ids": [16], "source_in": 92.5, "source_out": 98.0, "transcript": "將訊號轉換成標準向量格式輸出"},
            ]
        }
        anomalies = detect_micro_window_anomalies(model_edl, whisper_units, total_dur=120.0)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["replace_range"], (0, 3))
        self.assertIn("TAIL_HEAD_RETAKE", anomalies[0]["sub_types"])
        self.assertIn("INTRA_CLIP_REPEAT", anomalies[0]["sub_types"])

    def test_directional_opening_retake_preserves_long_prefix_before_tail_head_restart(self):
        """Treat A+B followed by B+C as a tail-head overlap rather than deleting A+B as an opening retake."""
        from scripts.edl_auditor import _detect_tail_head_overlap, _is_opening_retake_pair

        text_prev = "每個人的步行姿態與骨架大小都會在空間中擾動無線電波，當空間裡的設備不斷回報通道摘要時就會留下特徵。"
        text_next = "當空間裡的設備不斷回報通道摘要，你的動作就會被記錄成專屬特徵向量。"
        self.assertFalse(_is_opening_retake_pair(text_prev, text_next))
        overlap = _detect_tail_head_overlap(text_prev, text_next)
        self.assertIsNotNone(overlap)
        self.assertTrue(overlap.startswith("當空間裡的設備不斷回報"))

    def test_occlude_expanded_sub_units_drops_opening_retakes_and_slices_multichar_tail_head_overlap(self):
        """Drop expanded opening-retake sub-units and slice multi-char CJK words on tail-head overlap."""
        from scripts.edl_auditor import occlude_expanded_sub_units

        expanded = [
            {
                "parent_clip": {"clip_id": 1},
                "sentence_ids": [10],
                "t_first": 10.0,
                "t_last": 14.8,
                "prev_sentence_end": 9.2,
                "next_sentence_start": 15.2,
                "next_word_hard_wall": None,
                "words": [
                    {"word": "終端裝置將結果回傳給主機，", "start": 10.0, "end": 13.0},
                    {"word": "也就是說我們", "start": 13.2, "end": 14.8},
                ],
                "transcript": "終端裝置將結果回傳給主機，也就是說我們",
                "asr_transcript": "終端裝置將結果回傳給主機，也就是說我們",
                "trimmed_head": False,
                "trimmed_tail": False,
                "composite_role": "none",
                "is_composite_splice": False,
                "llm_verified_window": False,
            },
            {
                "parent_clip": {"clip_id": 2},
                "sentence_ids": [11],
                "t_first": 15.2,
                "t_last": 16.5,
                "prev_sentence_end": 14.8,
                "next_sentence_start": 17.0,
                "next_word_hard_wall": None,
                "words": [
                    {"word": "也就是說我們的設備", "start": 15.2, "end": 16.5},
                ],
                "transcript": "也就是說我們的設備",
                "asr_transcript": "也就是說我們的設備",
                "trimmed_head": False,
                "trimmed_tail": False,
                "composite_role": "none",
                "is_composite_splice": False,
                "llm_verified_window": False,
            },
            {
                "parent_clip": {"clip_id": 2},
                "sentence_ids": [12],
                "t_first": 17.0,
                "t_last": 22.0,
                "prev_sentence_end": 16.5,
                "next_sentence_start": 23.0,
                "next_word_hard_wall": None,
                "words": [
                    {"word": "也就是說我們的設備會公開廣播通道摘要。", "start": 17.0, "end": 22.0},
                ],
                "transcript": "也就是說我們的設備會公開廣播通道摘要。",
                "asr_transcript": "也就是說我們的設備會公開廣播通道摘要。",
                "trimmed_head": False,
                "trimmed_tail": False,
                "composite_role": "none",
                "is_composite_splice": False,
                "llm_verified_window": False,
            },
        ]
        occluded = occlude_expanded_sub_units(expanded)
        self.assertEqual(len(occluded), 2)
        self.assertEqual(occluded[0]["sentence_ids"], [10])
        self.assertAlmostEqual(occluded[0]["t_last"], 13.0, places=2)
        self.assertAlmostEqual(occluded[0]["next_word_hard_wall"], 13.199, places=2)
        self.assertIn("asr_transcript", occluded[0])
        self.assertEqual(occluded[1]["sentence_ids"], [12])
        self.assertAlmostEqual(occluded[1]["t_first"], 17.0, places=2)

    def test_parse_gemini_json_text_handles_comma_separated_objects_and_bare_arrays(self):
        """Wrap bare JSON arrays and comma-separated JSON objects into {'final_edl': [...]}."""
        from scripts.video_trimmer import _parse_gemini_json_text

        comma_objs = '{"clip_id": 1, "sentence_ids": [1]}, {"clip_id": 2, "sentence_ids": [2]}'
        parsed_a = _parse_gemini_json_text(comma_objs)
        self.assertEqual(len(parsed_a["final_edl"]), 2)

        bare_list = '[{"clip_id": 1, "sentence_ids": [3]}]'
        parsed_b = _parse_gemini_json_text(bare_list)
        self.assertEqual(parsed_b["final_edl"][0]["sentence_ids"], [3])

    def test_script_phrase_multiplicity_flags_duplicate_clause_and_exempts_legitimate_repetition(self):
        """Flag single-occurrence script clauses spoken in two clips, while exempting legitimately repeated script clauses."""
        script = "系統會將資料精準送到裝置，而主機為了把資料更精準送到裝置，會先向周邊裝置詢問狀態。"
        units = [
            {"id": 1, "start": 0.0, "end": 4.0, "text": "系統會將資料精準送到裝置，", "is_target_speaker": True},
            {"id": 2, "start": 4.2, "end": 8.0, "text": "而主機為了把資料更精準送到裝置，", "is_target_speaker": True},
            {"id": 3, "start": 8.2, "end": 11.5, "text": "會先向周邊裝置詢問狀態，", "is_target_speaker": True},
            {"id": 4, "start": 12.0, "end": 15.5, "text": "會先向周邊裝置詢問狀態。", "is_target_speaker": True},
        ]

        # Legitimate repetition ("資料精準送到裝置" / "精準送到裝置" appears twice in script, even with 1-char ASR homophone)
        clean_edl = {
            "final_edl": [
                {"clip_id": 1, "sentence_ids": [1], "transcript": "系統會將資料精準送到裝置，", "asr_transcript": "系統會將資料精準送到裝置"},
                {"clip_id": 2, "sentence_ids": [2, 4], "transcript": "而主機為了把資料更精準送到裝置，會先向周邊裝置詢問狀態。", "asr_transcript": "而主機為了把資寮更精準送到裝置會先向周邊裝置詢問狀態"},
            ]
        }
        anomalies_clean = detect_micro_window_anomalies(clean_edl, units, total_dur=20.0, script_text=script)
        mult_clean = [a for a in anomalies_clean if "SCRIPT_PHRASE_MULTIPLICITY" in a.get("sub_types", [a["type"]])]
        self.assertEqual(len(mult_clean), 0)

        # Duplicate retake of single-occurrence clause ("會先向周邊裝置詢問狀態" appears once in script, twice in EDL)
        dup_edl = {
            "final_edl": [
                {"clip_id": 1, "sentence_ids": [1, 2, 3], "transcript": "系統會將資料精準送到裝置，而主機為了把資料更精準送到裝置，會先向周邊裝置詢問狀態，", "asr_transcript": "系統會將資料精準送到裝置而主機為了把資料更精準送到裝置會先向周邊裝置詢問狀態"},
                {"clip_id": 2, "sentence_ids": [4], "transcript": "會先向周邊裝置詢問狀態。", "asr_transcript": "會先向周邊裝置詢問狀態"},
            ]
        }
        anomalies_dup = detect_micro_window_anomalies(dup_edl, units, total_dur=20.0, script_text=script)
        mult_dup = [a for a in anomalies_dup if "SCRIPT_PHRASE_MULTIPLICITY" in a.get("sub_types", [a["type"]])]
        self.assertEqual(len(mult_dup), 1)

    def test_unscripted_extraneous_span_flags_asr_leak_and_false_start_but_exempts_particles(self):
        """Flag unscripted >= 6-char false starts and ASR alignment leaks, while exempting short spoken particles and homophones."""
        script = "控制器收到封包之後，就能即時計算通道狀態並完成同步。"
        units = [
            {"id": 1, "start": 0.0, "end": 5.0, "text": "啊控制器收到封包之後呢，", "is_target_speaker": True},
            {"id": 2, "start": 5.2, "end": 10.0, "text": "就能即時計算通道狀態並完成同步。", "is_target_speaker": True},
            {"id": 3, "start": 10.5, "end": 16.0, "text": "我們先重講這一段控制器收到封包之後，", "is_target_speaker": True},
        ]

        # Short spoken particles ("啊", "呢") and homophone ("即實" vs "即時") -> 0 anomalies
        clean_edl = {
            "final_edl": [
                {"clip_id": 1, "sentence_ids": [1, 2], "transcript": "啊控制器收到封包之後呢，就能即時計算通道狀態並完成同步。", "asr_transcript": "啊控制器收到封包之後呢就能即實計算通道狀態並完成同步"},
            ]
        }
        anomalies_clean = detect_micro_window_anomalies(clean_edl, units, total_dur=20.0, script_text=script)
        ext_clean = [a for a in anomalies_clean if "UNSCRIPTED_EXTRANEOUS_SPAN" in a.get("sub_types", [a["type"]])]
        self.assertEqual(len(ext_clean), 0)

        # Unscripted >= 6-char false-start prefix ("我們先重講這一段") -> flagged
        bad_edl = {
            "final_edl": [
                {"clip_id": 1, "sentence_ids": [3, 2], "transcript": "控制器收到封包之後，就能即時計算通道狀態並完成同步。", "asr_transcript": "我們先重講這一段控制器收到封包之後就能即時計算通道狀態並完成同步"},
            ]
        }
        anomalies_bad = detect_micro_window_anomalies(bad_edl, units, total_dur=20.0, script_text=script)
        ext_bad = [a for a in anomalies_bad if "UNSCRIPTED_EXTRANEOUS_SPAN" in a.get("sub_types", [a["type"]])]
        self.assertEqual(len(ext_bad), 1)

    def test_closed_loop_post_alignment_verification_catches_physical_span_leak(self):
        """Verify _expand_and_occlude_edl_clips + _sub_units_to_aligned_edl exposes off-by-one physical word leaks before GCS cleanup."""
        from scripts.video_trimmer import _expand_and_occlude_edl_clips, _sub_units_to_aligned_edl

        script = "第一階段負責資料擷取，第二階段負責特徵轉換。"
        w_units = [
            {
                "id": 1,
                "start": 0.0,
                "end": 3.5,
                "text": "等一下我們先重講這一段剛剛講太快了，",
                "is_target_speaker": True,
                "words": [{"word": "等一下我們先重講這一段剛剛講太快了，", "start": 0.0, "end": 3.5}],
            },
            {
                "id": 2,
                "start": 4.0,
                "end": 7.0,
                "text": "第一階段負責資料擷取，",
                "is_target_speaker": True,
                "words": [{"word": "第一階段負責資料擷取，", "start": 4.0, "end": 7.0}],
            },
            {
                "id": 3,
                "start": 7.2,
                "end": 10.5,
                "text": "第二階段負責特徵轉換。",
                "is_target_speaker": True,
                "words": [{"word": "第二階段負責特徵轉換。", "start": 7.2, "end": 10.5}],
            },
        ]
        # Off-by-one Sentence ID hallucination: LLM wrote clean script text for ID 2, but pointed sentence_ids at NG ID 1
        raw_edl = {
            "final_edl": [
                {"clip_id": 1, "topic": "Stage 1", "sentence_ids": [1], "transcript": "第一階段負責資料擷取，"},
                {"clip_id": 2, "topic": "Stage 2", "sentence_ids": [3], "transcript": "第二階段負責特徵轉換。"},
            ]
        }
        pre_anomalies = detect_micro_window_anomalies(raw_edl, w_units, total_dur=12.0, script_text=script)
        self.assertEqual(len(pre_anomalies), 0)

        preview_units = _expand_and_occlude_edl_clips(raw_edl, w_units, total_dur=12.0, script_text=script)
        aligned_edl = _sub_units_to_aligned_edl(raw_edl, preview_units)
        post_anomalies = detect_micro_window_anomalies(aligned_edl, w_units, total_dur=12.0, script_text=script)
        post_types = {t for a in post_anomalies for t in a.get("sub_types", [a["type"]])}
        self.assertIn("UNSCRIPTED_EXTRANEOUS_SPAN", post_types)

    def test_extend_omitted_adjacent_short_clauses_recovers_head_and_tail_clauses(self):
        """Extend sentence_ids to include unclaimed opening or closing short clauses explicitly written in transcript."""
        from scripts.video_trimmer import _expand_and_occlude_edl_clips

        w_units = [
            {
                "id": 10,
                "start": 40.00,
                "end": 41.90,
                "text": "盡管如此,這份報告揭露的是",
                "is_target_speaker": True,
                "words": [
                    {"word": "盡管如此,", "start": 40.00, "end": 40.70},
                    {"word": "這份報告揭露的是", "start": 40.75, "end": 41.90},
                ],
            },
            {
                "id": 11,
                "start": 43.12,
                "end": 43.84,
                "text": "盡管如此,",
                "is_target_speaker": True,
                "words": [{"word": "盡管如此,", "start": 43.12, "end": 43.84}],
            },
            {
                "id": 12,
                "start": 43.88,
                "end": 49.36,
                "text": "這份報告揭露的是極具威脅的系統缺口啊,更麻煩的是標準",
                "is_target_speaker": True,
                "words": [
                    {"word": "這份", "start": 43.88, "end": 44.30},
                    {"word": "報告揭露的是極具威脅的系統缺口啊,", "start": 44.32, "end": 47.84},
                    {"word": "更麻煩的是標準", "start": 48.10, "end": 49.36},
                ],
            },
            {
                "id": 20,
                "start": 60.00,
                "end": 64.00,
                "text": "整個模組已經通過壓力測試,",
                "is_target_speaker": True,
                "words": [{"word": "整個模組已經通過壓力測試,", "start": 60.00, "end": 64.00}],
            },
            {
                "id": 21,
                "start": 64.15,
                "end": 65.05,
                "text": "表現非常穩定。",
                "is_target_speaker": True,
                "words": [{"word": "表現非常穩定。", "start": 64.15, "end": 65.05}],
            },
        ]

        # 1. Fresh EDL output (no t_first/t_last yet): recovers head ID 11 and tail ID 21 while trimming ID 12 tail stumble
        raw_edl = {
            "final_edl": [
                {
                    "clip_id": 1,
                    "topic": "Security Gap",
                    "sentence_ids": [12],
                    "transcript": "儘管如此，這份報告揭露的是極具威脅的系統缺口啊，",
                },
                {
                    "clip_id": 2,
                    "topic": "Stress Test",
                    "sentence_ids": [20],
                    "transcript": "整個模組已經通過壓力測試，表現非常穩定。",
                },
            ]
        }
        expanded_fresh = _expand_and_occlude_edl_clips(raw_edl, w_units, total_dur=70.0)
        self.assertEqual(len(expanded_fresh), 2)
        self.assertEqual(expanded_fresh[0]["sentence_ids"], [11, 12])
        self.assertAlmostEqual(expanded_fresh[0]["t_first"], 43.12, places=2)
        self.assertAlmostEqual(expanded_fresh[0]["t_last"], 47.84, places=2)
        self.assertTrue(expanded_fresh[0]["trimmed_tail"])
        self.assertFalse(expanded_fresh[0]["trimmed_head"])

        self.assertEqual(expanded_fresh[1]["sentence_ids"], [20, 21])
        self.assertAlmostEqual(expanded_fresh[1]["t_first"], 60.00, places=2)
        self.assertAlmostEqual(expanded_fresh[1]["t_last"], 65.05, places=2)

        # 2. Cached JSON replay (t_first/t_last and trimmed_tail=True already stored): still recovers ID 11 as one unified clip
        cached_edl = {
            "final_edl": [
                {
                    "clip_id": 1,
                    "topic": "Security Gap",
                    "sentence_ids": [12],
                    "start_sentence_id": 12,
                    "end_sentence_id": 12,
                    "t_first": 43.88,
                    "t_last": 47.84,
                    "source_in": 43.80,
                    "source_out": 47.90,
                    "trimmed_head": False,
                    "trimmed_tail": True,
                    "transcript": "儘管如此，這份報告揭露的是極具威脅的系統缺口啊，",
                    "asr_transcript": "這份報告揭露的是極具威脅的系統缺口啊,",
                }
            ]
        }
        expanded_cached = _expand_and_occlude_edl_clips(cached_edl, w_units, total_dur=70.0)
        self.assertEqual(len(expanded_cached), 1)
        self.assertEqual(expanded_cached[0]["sentence_ids"], [11, 12])
        self.assertAlmostEqual(expanded_cached[0]["t_first"], 43.12, places=2)
        self.assertAlmostEqual(expanded_cached[0]["t_last"], 47.84, places=2)

    def test_extend_omitted_adjacent_short_clauses_rejects_ng_takes_long_gaps_and_claimed_ids(self):
        """Never absorb longer NG sentences, pauses > 0.50s, claimed IDs, or composite/trimmed edges."""
        w_units = [
            # Case A: ID 1 is a longer aborted NG sentence (len 11 vs missing_head len 4), even within a short pause
            {"id": 1, "start": 10.00, "end": 12.00, "text": "盡管如此,這份報告揭露的是", "is_target_speaker": True},
            {"id": 2, "start": 12.20, "end": 16.00, "text": "這份報告揭露的是核心系統架構。", "is_target_speaker": True},
            # Case B: ID 3 has pause gap 0.85s (> 0.50s) before ID 4
            {"id": 3, "start": 20.00, "end": 20.70, "text": "換句話說,", "is_target_speaker": True},
            {"id": 4, "start": 21.55, "end": 25.00, "text": "硬體延遲已經降到最低標準。", "is_target_speaker": True},
            # Case C: ID 5 is already claimed by another clip; ID 7 is across a composite continuation seam
            {"id": 5, "start": 30.00, "end": 30.60, "text": "總而言之,", "is_target_speaker": True},
            {"id": 6, "start": 30.70, "end": 34.50, "text": "整體吞吐量提升了兩倍以上。", "is_target_speaker": True},
            {"id": 7, "start": 40.00, "end": 40.60, "text": "也就是說,", "is_target_speaker": True},
            {"id": 8, "start": 40.70, "end": 44.50, "text": "備援機制可以即時接手運算。", "is_target_speaker": True},
        ]
        clips = [
            {"clip_id": 1, "sentence_ids": [2], "transcript": "儘管如此，這份報告揭露的是核心系統架構。"},
            {"clip_id": 2, "sentence_ids": [4], "transcript": "換句話說，硬體延遲已經降到最低標準。"},
            {"clip_id": 3, "sentence_ids": [5], "transcript": "總而言之，"},
            {"clip_id": 4, "sentence_ids": [6], "transcript": "總而言之，整體吞吐量提升了兩倍以上。"},
            {
                "clip_id": 5,
                "sentence_ids": [8],
                "composite_role": "continuation",
                "is_composite_splice": True,
                "transcript": "也就是說，備援機制可以即時接手運算。",
            },
        ]
        deduped = deduplicate_and_sort_clips(clips, w_units)
        by_sids = [c["sentence_ids"] for c in deduped]
        self.assertEqual(by_sids, [[2], [4], [5], [6], [8]])


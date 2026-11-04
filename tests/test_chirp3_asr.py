"""Unit tests for Cloud Speech-to-Text V2 (Chirp 3) and dual-engine ASR caching."""

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.chirp3_asr import (
    _compute_chunk_windows,
    _group_words_into_segments,
    _suppress_repetition_loops,
    join_words_for_display,
    resolve_chirp3_language_codes,
)
from scripts.exceptions import VideoTrimmerError
from scripts.transcribe import transcribe_video_words


class TestChirp3AsrHelpers(unittest.TestCase):
    """Test Chirp 3 language resolution, chunking, loop suppression, and word grouping."""

    def test_resolve_chirp3_language_codes_explicit_and_script_hints(self):
        self.assertEqual(resolve_chirp3_language_codes("zh-TW"), (["cmn-Hant-TW", "en-US"], "zh-TW"))
        self.assertEqual(resolve_chirp3_language_codes("en"), (["en-US"], "en"))
        self.assertEqual(resolve_chirp3_language_codes("auto"), (["auto"], "zh-TW"))

        trad_hint = "今天我們來介紹最新的系統架構與實測數據。"
        self.assertEqual(resolve_chirp3_language_codes("auto", text_hint=trad_hint), (["cmn-Hant-TW", "en-US"], "zh-TW"))

        simp_hint = "今天我们来介绍这个系统的实时处理架构与评测结果。"
        self.assertEqual(resolve_chirp3_language_codes("auto", text_hint=simp_hint), (["cmn-Hans-CN", "en-US"], "zh-CN"))

        ja_hint = "本日は新しいシステム構成について説明します。"
        self.assertEqual(resolve_chirp3_language_codes("auto", text_hint=ja_hint), (["ja-JP", "en-US"], "ja"))

        en_hint = "Today we present the new multi-camera editing architecture and benchmarks."
        self.assertEqual(resolve_chirp3_language_codes("auto", text_hint=en_hint), (["en-US"], "en"))

    def test_compute_chunk_windows_splits_with_overlap(self):
        self.assertEqual(_compute_chunk_windows(540.0), [(0.0, 540.0)])
        windows = _compute_chunk_windows(2400.0)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0], (0.0, 1080.0))
        self.assertEqual(windows[1], (1075.0, 2160.0))
        self.assertEqual(windows[2], (2155.0, 2400.0))

    def test_suppress_repetition_loops_collapses_stalled_ngrams(self):
        words = [
            {"word": "Thank", "start": 0.0, "end": 0.3},
            {"word": "you", "start": 0.3, "end": 0.5},
            {"word": "very", "start": 0.5, "end": 0.8},
            {"word": "Thank", "start": 0.9, "end": 1.2},
            {"word": "you", "start": 1.2, "end": 1.4},
            {"word": "very", "start": 1.4, "end": 1.7},
            {"word": "Thank", "start": 1.8, "end": 2.1},
            {"word": "you", "start": 2.1, "end": 2.3},
            {"word": "very", "start": 2.3, "end": 2.6},
            {"word": "much.", "start": 2.7, "end": 3.1},
        ]
        cleaned = _suppress_repetition_loops(words)
        joined = join_words_for_display([w["word"] for w in cleaned])
        self.assertEqual(joined, "Thank you very much.")

    def test_group_words_into_segments_preserves_millisecond_precision(self):
        words = [
            {"word": "研究者推測", "start": 1.123, "end": 1.845},
            {"word": "API", "start": 1.852, "end": 2.214},
            {"word": "本來就是壓縮。", "start": 2.220, "end": 3.108},
            {"word": "下一句開始。", "start": 3.620, "end": 4.415},
        ]
        segs = _group_words_into_segments(words, norm_lang="zh-TW")
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["id"], 1)
        self.assertEqual(segs[0]["start"], 1.123)
        self.assertEqual(segs[0]["end"], 3.108)
        self.assertEqual(segs[0]["text"], "研究者推測API本來就是壓縮。")
        self.assertEqual(segs[1]["id"], 2)
        self.assertEqual(segs[1]["start"], 3.620)


class TestDualEngineCachingAndFailFast(unittest.TestCase):
    """Test engine-tagged cache invalidation and fail-fast error handling."""

    def test_engine_tagged_cache_hit_mismatch_and_force_refresh(self):
        whisper_cached = {
            "asr_engine": "whisper",
            "model": "small",
            "language": "zh-TW",
            "segments": [
                {
                    "id": 1,
                    "start": 1.0,
                    "end": 2.5,
                    "text": "這是Whisper快取。",
                    "words": [{"word": "這是Whisper快取。", "start": 1.0, "end": 2.5}],
                }
            ],
        }
        chirp_segments = [
            {
                "id": 1,
                "start": 1.005,
                "end": 2.495,
                "text": "這是Chirp快取。",
                "words": [{"word": "這是Chirp快取。", "start": 1.005, "end": 2.495}],
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            video_path = tmp_dir / "raw_footage.mp4"
            video_path.write_bytes(b"fake")
            cache_json = tmp_dir / "raw_footage_words.json"
            cache_json.write_text(json.dumps(whisper_cached, ensure_ascii=False), encoding="utf-8")

            # 1. Requesting whisper reuses matching whisper-tagged _words.json cache
            with patch("scripts.transcribe.run_chirp3_transcription") as mock_chirp:
                raw_w, sents_w = transcribe_video_words(video_path, cache_json, asr_engine="whisper")
                mock_chirp.assert_not_called()
                self.assertEqual(raw_w[0]["text"], "這是Whisper快取。")
                self.assertEqual(len(sents_w), 1)
                self.assertTrue((tmp_dir / "raw_footage_sentences.json").exists())

            # 2. Requesting chirp3 invalidates whisper cache and writes chirp3-tagged _words.json
            with patch("scripts.transcribe.run_chirp3_transcription", return_value=(chirp_segments, "zh-TW")) as mock_chirp:
                raw_c, _ = transcribe_video_words(
                    video_path,
                    cache_json,
                    asr_engine="chirp3",
                    project="demo-proj",
                    gcs_bucket="demo-bucket",
                )
                mock_chirp.assert_called_once()
                self.assertEqual(raw_c[0]["start"], 1.005)
                saved = json.loads(cache_json.read_text(encoding="utf-8"))
                self.assertEqual(saved["asr_engine"], "chirp3")
                self.assertTrue((tmp_dir / "raw_footage_sentences.json").exists())

            # 3. Second chirp3 call hits cache unless force=True
            with patch("scripts.transcribe.run_chirp3_transcription", return_value=(chirp_segments, "zh-TW")) as mock_chirp:
                transcribe_video_words(video_path, cache_json, asr_engine="chirp3")
                mock_chirp.assert_not_called()

                transcribe_video_words(video_path, cache_json, asr_engine="chirp3", force=True)
                mock_chirp.assert_called_once()

    def test_untagged_or_malformed_cache_retranscribes_cleanly(self):
        chirp_segments = [
            {
                "id": 1,
                "start": 0.5,
                "end": 2.0,
                "text": "重新產生的新格式快取。",
                "words": [{"word": "重新產生的新格式快取。", "start": 0.5, "end": 2.0}],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            video_path = tmp_dir / "raw_footage.mp4"
            video_path.write_bytes(b"fake")
            words_json = tmp_dir / "raw_footage_words.json"
            sentences_json = tmp_dir / "raw_footage_sentences.json"
            words_json.write_text(json.dumps([{"id": 1, "text": "untagged"}]), encoding="utf-8")

            with patch("scripts.transcribe.run_chirp3_transcription", return_value=(chirp_segments, "zh-TW")) as mock_chirp:
                raw_c, sents_c = transcribe_video_words(
                    video_path,
                    words_json,
                    asr_engine="chirp3",
                    project="demo-proj",
                    gcs_bucket="demo-bucket",
                )
                mock_chirp.assert_called_once()
                self.assertEqual(len(raw_c), 1)
                self.assertEqual(len(sents_c), 1)
                saved = json.loads(words_json.read_text(encoding="utf-8"))
                self.assertEqual(saved["asr_engine"], "chirp3")
                self.assertTrue(sentences_json.exists())

    def test_chirp3_http_403_fails_fast_and_cleans_up_gcs_without_whisper_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            video_path = tmp_dir / "raw_footage.mp4"
            video_path.write_bytes(b"fake")
            cache_json = tmp_dir / "raw_footage_words.json"

            mock_creds = MagicMock(valid=True, token="fake-token")
            mock_blob = MagicMock()
            mock_cleanup_blob = MagicMock()
            mock_bucket = MagicMock()
            mock_bucket.blob.return_value = mock_blob
            mock_storage = MagicMock()
            mock_storage.bucket.return_value = mock_bucket
            mock_storage.list_blobs.return_value = [mock_cleanup_blob]

            http_403 = urllib.error.HTTPError(
                url="https://us-speech.googleapis.com/v2/projects/p/locations/us/recognizers/_:batchRecognize",
                code=403,
                msg="Forbidden",
                hdrs=None,
                fp=io.BytesIO(b'{"error":{"message":"Permission speech.recognizers.recognize denied"}}'),
            )

            def _fake_ffmpeg(cmd, *args, **kwargs):
                Path(cmd[-1]).write_bytes(b"mp3")
                return MagicMock(returncode=0, stderr="")

            with (
                patch("scripts.chirp3_asr.google.auth.default", return_value=(mock_creds, "demo-proj")),
                patch("scripts.chirp3_asr.get_gcs_client", return_value=mock_storage),
                patch("scripts.chirp3_asr.subprocess.run", side_effect=_fake_ffmpeg),
                patch("scripts.chirp3_asr._get_audio_duration", return_value=12.0),
                patch("scripts.chirp3_asr.urllib.request.urlopen", side_effect=http_403),
                patch("scripts.transcribe._extract_mono_wav") as mock_whisper_wav,
            ):
                with self.assertRaises(VideoTrimmerError) as ctx:
                    transcribe_video_words(
                        video_path=video_path,
                        out_json_path=cache_json,
                        asr_engine="chirp3",
                        project="demo-proj",
                        gcs_bucket="demo-bucket",
                    )
                self.assertIn("403", str(ctx.exception))
                mock_whisper_wav.assert_not_called()
                mock_cleanup_blob.delete.assert_called_once()

    def test_synchronize_composite_overlap_pivots_updates_word_bounds_and_transcripts(self):
        import numpy as np
        from scripts.video_trimmer import _synchronize_composite_overlap_pivots

        sr = 16000
        audio = np.zeros(sr * 8, dtype=np.float64)
        units = [
            {
                "composite_role": "prefix",
                "is_composite_splice": False,
                "t_first": 0.0,
                "t_last": 2.4,
                "prev_sentence_end": None,
                "next_sentence_start": 2.9,
                "next_word_hard_wall": 2.9,
                "trimmed_head": False,
                "trimmed_tail": True,
                "transcript": "今天我們介紹核心",
                "asr_transcript": "今天我們介紹核心",
                "words": [
                    {"word": "今天我們", "start": 0.0, "end": 0.8},
                    {"word": "介紹", "start": 0.8, "end": 1.5},
                    {"word": "核心", "start": 1.6, "end": 2.4},
                ],
            },
            {
                "composite_role": "continuation",
                "is_composite_splice": True,
                "t_first": 3.0,
                "t_last": 6.0,
                "prev_sentence_end": 2.8,
                "next_sentence_start": None,
                "next_word_hard_wall": None,
                "trimmed_head": False,
                "trimmed_tail": False,
                "transcript": "介紹核心系統架構。",
                "asr_transcript": "介紹核心系統架構。",
                "words": [
                    {"word": "介紹", "start": 3.0, "end": 3.6},
                    {"word": "核心", "start": 3.9, "end": 4.5},
                    {"word": "系統架構。", "start": 4.5, "end": 6.0},
                ],
            },
        ]
        _synchronize_composite_overlap_pivots(units, audio, sr)
        combined = units[0]["asr_transcript"] + units[1]["asr_transcript"]
        self.assertEqual(combined, "今天我們介紹核心系統架構。")
        self.assertLessEqual(units[0]["t_last"], 1.5)


if __name__ == "__main__":
    unittest.main()

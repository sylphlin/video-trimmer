"""exporters.py 的離線單元測試：假造 EDL list，驗證輸出 XML 為合法格式且時間碼換算正確。"""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.exporters import generate_edl_csv, generate_fcp7_xml, generate_fcpxml


def _sample_edl():
    return [
        {
            "clip_id": 1, "topic": "開場", "source_in": 10.0, "source_out": 15.0, "duration": 5.0,
            "cps": 4.0, "in_margin": 0.1, "out_margin": 0.15,
            "transcript": "大家好", "visual_check": "眼神就緒", "audio_check": "收音完整",
        },
        {
            "clip_id": 2, "topic": "重點段落說明", "source_in": 20.0, "source_out": 27.5, "duration": 7.5,
            "cps": 5.0, "in_margin": 0.08, "out_margin": 0.12,
            "transcript": "這是重點", "visual_check": "眼神就緒", "audio_check": "收音完整",
        },
    ]


class TestGenerateFcp7Xml(unittest.TestCase):
    def test_output_is_well_formed_and_parseable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "take1.mp4"
            video_path.write_bytes(b"")
            xml_path = tmp_path / "out.xml"

            generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0, output_xml_path=xml_path,
                               width=1920, height=1080, fps=24.0)

            root = ET.parse(xml_path).getroot()
            assert root.tag == "xmeml"
            assert len(root.findall(".//clipitem")) == len(_sample_edl()) * 2

    def test_timecode_conversion_is_frame_accurate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "take1.mp4"
            video_path.write_bytes(b"")
            xml_path = tmp_path / "out.xml"
            fps = 24.0

            generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0, output_xml_path=xml_path,
                               width=1920, height=1080, fps=fps)

            root = ET.parse(xml_path).getroot()
            first_clip = root.find(".//video//clipitem")
            assert int(first_clip.find("in").text) == round(10.0 * fps)
            assert int(first_clip.find("out").text) == round(15.0 * fps)

            second_clip = root.findall(".//video//clipitem")[1]
            assert int(second_clip.find("start").text) == round(15.0 * fps) - round(10.0 * fps)

    def test_ntsc_flag_reflects_fractional_vs_integer_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "take1.mp4"
            video_path.write_bytes(b"")
            xml_ntsc = tmp_path / "ntsc.xml"
            xml_pal = tmp_path / "pal.xml"

            generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0, output_xml_path=xml_ntsc,
                               width=1920, height=1080, fps=29.97)
            generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0, output_xml_path=xml_pal,
                               width=1920, height=1080, fps=25.0)

            root_ntsc = ET.parse(xml_ntsc).getroot()
            root_pal = ET.parse(xml_pal).getroot()
            assert root_ntsc.find(".//rate/timebase").text == "30"
            assert root_ntsc.find(".//rate/ntsc").text == "TRUE"
            assert root_pal.find(".//rate/timebase").text == "25"
            assert root_pal.find(".//rate/ntsc").text == "FALSE"


class TestGenerateFcpxml(unittest.TestCase):
    def test_output_is_well_formed_and_parseable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "take1.mp4"
            video_path.write_bytes(b"")
            fcpxml_path = tmp_path / "out.fcpxml"
            edl = _sample_edl()
            total_out_dur = sum(c["duration"] for c in edl)

            generate_fcpxml(edl, video_path, total_source_dur=30.0, total_out_dur=total_out_dur,
                             output_fcpxml_path=fcpxml_path, fps=23.976)

            root = ET.parse(fcpxml_path).getroot()
            assert root.tag == "fcpxml"
            assert len(root.findall(".//asset-clip")) == len(edl)

    def test_duration_fraction_matches_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "take1.mp4"
            video_path.write_bytes(b"")
            fcpxml_path = tmp_path / "out.fcpxml"
            edl = [{"clip_id": 1, "topic": "A", "source_in": 0.0, "source_out": 2.0, "duration": 2.0}]

            generate_fcpxml(edl, video_path, total_source_dur=2.0, total_out_dur=2.0,
                             output_fcpxml_path=fcpxml_path, fps=23.976)

            root = ET.parse(fcpxml_path).getroot()
            clip = root.find(".//asset-clip")
            num, den = clip.get("duration").rstrip("s").split("/")
            seconds = float(num) / float(den)
            frame_duration = 1001 / 24000
            self.assertAlmostEqual(seconds, 2.0, delta=frame_duration)

    def test_supports_multiple_frame_rates_and_cumulative_spine_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "take1.mp4"
            video_path.write_bytes(b"")
            fcpxml_60 = tmp_path / "out_60.fcpxml"
            edl = _sample_edl()

            generate_fcpxml(edl, video_path, total_source_dur=30.0, total_out_dur=12.5,
                             output_fcpxml_path=fcpxml_60, fps=60.0, width=3840, height=2160)

            root = ET.parse(fcpxml_60).getroot()
            fmt = root.find(".//format")
            assert fmt.get("frameDuration") == "100/6000s"
            assert fmt.get("width") == "3840"
            assert fmt.get("height") == "2160"
            clips = root.findall(".//asset-clip")
            assert clips[0].get("offset") == "0s"
            assert clips[1].get("offset") == "30000/6000s"


class TestGenerateEdlCsv(unittest.TestCase):
    def test_csv_contains_header_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "out.csv"
            generate_edl_csv(_sample_edl(), csv_path)

            content = csv_path.read_text(encoding="utf-8-sig")
            lines = content.strip().splitlines()
            assert lines[0].startswith("Clip_ID,Topic,Source_In")
            assert len(lines) == 1 + len(_sample_edl())
            assert "開場" in lines[1]

    def test_quotes_in_fields_are_escaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "out.csv"
            edl = [{
                "clip_id": 1, "topic": "A", "source_in": 0.0, "source_out": 1.0, "duration": 1.0,
                "cps": 3.0, "in_margin": 0.1, "out_margin": 0.1,
                "transcript": '他說"你好"', "visual_check": "ok", "audio_check": "ok",
            }]
            generate_edl_csv(edl, csv_path)
            content = csv_path.read_text(encoding="utf-8-sig")
            assert '""你好""' in content


class TestRenderCutVideo(unittest.TestCase):
    def test_build_render_cmd_uses_per_clip_fast_seeking_and_gop_30(self):
        from scripts.render import _build_render_cmd

        edl = _sample_edl()
        cmd = _build_render_cmd(
            edl,
            video_path="raw_footage.mp4",
            out_mp4_path="out_trimmed.mp4",
            crf=18,
            encoder="h264_videotoolbox",
            use_hwaccel=True,
        )
        cmd_str = " ".join(cmd)
        self.assertIn("-hwaccel videotoolbox -ss 10.000 -to 15.000 -i raw_footage.mp4", cmd_str)
        self.assertIn("-hwaccel videotoolbox -ss 20.000 -to 27.500 -i raw_footage.mp4", cmd_str)
        self.assertIn("afade=t=in:st=0:d=0.020:curve=iqsin", cmd_str)
        self.assertIn("d=0.020:curve=qsin", cmd_str)
        self.assertNotIn("curve=oqsin", cmd_str)
        self.assertIn("-c:v h264_videotoolbox -b:v 12M -g 30", cmd_str)
        self.assertIn("-movflags +faststart", cmd_str)

    def test_render_cut_video_falls_back_to_libx264_when_videotoolbox_fails(self):
        from unittest.mock import MagicMock, patch
        from scripts.render import render_cut_video

        fail_proc = MagicMock(returncode=1, stderr="vt failed")
        ok_proc = MagicMock(returncode=0, stderr="")
        with patch("scripts.render.subprocess.run", side_effect=[fail_proc, fail_proc, ok_proc]) as mock_run:
            render_cut_video(_sample_edl(), "raw_footage.mp4", "out_trimmed.mp4", crf=20)
            self.assertEqual(mock_run.call_count, 3)
            final_cmd = " ".join(mock_run.call_args_list[2][0][0])
            self.assertIn("-c:v libx264 -preset fast -crf 20 -g 30", final_cmd)

    def test_real_ffmpeg_filtergraph_execution(self):
        import shutil
        import subprocess
        from scripts.render import render_cut_video

        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_mp4 = tmp_path / "synthetic_input.mp4"
            out_mp4 = tmp_path / "synthetic_trimmed.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=320x240:r=30:d=1.5",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.5",
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", "15",
                    "-c:a", "aac",
                    str(src_mp4),
                ],
                check=True,
                capture_output=True,
            )
            edl = [
                {"clip_id": 1, "topic": "A", "source_in": 0.1, "source_out": 0.5, "duration": 0.4},
                {"clip_id": 2, "topic": "B", "source_in": 0.8, "source_out": 1.3, "duration": 0.5},
            ]
            render_cut_video(edl, src_mp4, out_mp4, crf=24)
            self.assertTrue(out_mp4.exists())
            self.assertGreater(out_mp4.stat().st_size, 0)


class TestResolveOutputDir(unittest.TestCase):
    def test_defaults_to_output_subdir_under_parent(self):
        from scripts.video_trimmer import _resolve_output_dir

        with tempfile.TemporaryDirectory() as tmp:
            parent_dir = Path(tmp)
            out_dir = _resolve_output_dir(None, parent_dir)
            self.assertEqual(out_dir, parent_dir.resolve() / "output")
            self.assertTrue(out_dir.is_dir())

    def test_explicit_output_dir_overrides_default(self):
        from scripts.video_trimmer import _resolve_output_dir

        with tempfile.TemporaryDirectory() as tmp:
            parent_dir = Path(tmp)
            custom_dir = parent_dir / "custom_deliverables"
            out_dir = _resolve_output_dir(str(custom_dir), parent_dir)
            self.assertEqual(out_dir, custom_dir.resolve())
            self.assertTrue(out_dir.is_dir())


class TestCompositeMarkersAndPunchIn(unittest.TestCase):
    """Verify composite pick-up markers, DTD-valid transform order, XML escaping, and FFmpeg punch-in."""

    def _composite_edl(self):
        return [
            {
                "clip_id": 1,
                "topic": 'Prefix "A&B"',
                "source_in": 1.0,
                "source_out": 2.2,
                "duration": 1.2,
                "composite_role": "prefix",
                "is_composite_splice": False,
                "gain_db": 0.0,
            },
            {
                "clip_id": 2,
                "topic": "Pick-up <C>",
                "source_in": 3.0,
                "source_out": 4.5,
                "duration": 1.5,
                "composite_role": "continuation",
                "is_composite_splice": True,
                "gain_db": -2.25,
            },
        ]

    def test_fcpxml_composite_marker_and_transform_follow_strict_v1_9_dtd_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "raw_footage.mp4"
            video_path.write_bytes(b"")
            fcpxml_path = tmp_path / "composite.fcpxml"

            generate_fcpxml(
                self._composite_edl(),
                video_path,
                total_source_dur=10.0,
                total_out_dur=2.7,
                output_fcpxml_path=fcpxml_path,
                fps=30.0,
                width=1920,
                height=1080,
                punch_in_composite=True,
            )

            root = ET.parse(fcpxml_path).getroot()
            clips = root.findall(".//asset-clip")
            self.assertEqual(len(clips), 2)
            self.assertEqual(list(clips[0]), [])
            child_tags = [child.tag for child in clips[1]]
            self.assertEqual(child_tags, ["adjust-transform", "marker"])
            self.assertEqual(clips[1].find("adjust-transform").get("scale"), "1.12 1.12")
            self.assertEqual(clips[1].find("marker").get("value"), "Composite Splice (Pick-up)")

    def test_fcp7_xml_composite_marker_and_basic_motion_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "raw_footage.mp4"
            video_path.write_bytes(b"")
            xml_path = tmp_path / "composite.xml"

            generate_fcp7_xml(
                self._composite_edl(),
                video_path,
                total_source_dur=10.0,
                output_xml_path=xml_path,
                width=1920,
                height=1080,
                fps=30.0,
                punch_in_composite=True,
            )

            root = ET.parse(xml_path).getroot()
            v_clips = root.findall(".//video//clipitem")
            self.assertEqual(len(v_clips), 2)
            self.assertIsNone(v_clips[0].find("marker"))
            self.assertIsNotNone(v_clips[1].find("marker"))
            self.assertEqual(v_clips[1].find("marker/name").text, "Composite Splice (Pick-up)")
            self.assertEqual(v_clips[1].find(".//parameter[parameterid='scale']/value").text, "112.0")

    def test_real_ffmpeg_punch_in_composite_and_gain_db_on_854x480(self):
        import shutil
        import subprocess
        from scripts.render import _build_render_cmd, render_cut_video

        edl = [
            {"clip_id": 1, "topic": "Prefix", "source_in": 0.1, "source_out": 0.5, "duration": 0.4, "is_composite_splice": False, "gain_db": 0.0},
            {"clip_id": 2, "topic": "Pickup", "source_in": 0.7, "source_out": 1.2, "duration": 0.5, "is_composite_splice": True, "gain_db": -2.25},
        ]
        cmd_str = " ".join(_build_render_cmd(edl, "raw_footage.mp4", "out.mp4", punch_in_composite=True, width=854, height=480))
        self.assertIn("crop=trunc(iw/1.1200/2)*2:trunc(ih/1.1200/2)*2", cmd_str)
        self.assertIn("volume=-2.25dB,", cmd_str)

        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_mp4 = tmp_path / "synthetic_854x480.mp4"
            out_mp4 = tmp_path / "synthetic_punchin.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=navy:s=854x480:r=30:d=1.5",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.5",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", "15",
                    "-c:a", "aac",
                    str(src_mp4),
                ],
                check=True,
                capture_output=True,
            )
            render_cut_video(edl, src_mp4, out_mp4, crf=24, punch_in_composite=True, width=854, height=480)
            self.assertTrue(out_mp4.exists())
            self.assertGreater(out_mp4.stat().st_size, 0)

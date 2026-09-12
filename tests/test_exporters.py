"""exporters.py 的離線單元測試：假造 EDL list，驗證輸出 XML 為合法格式且時間碼換算正確。"""

import xml.etree.ElementTree as ET

import pytest

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


class TestGenerateFcp7Xml:
    def test_output_is_well_formed_and_parseable(self, tmp_path):
        video_path = tmp_path / "take1.mp4"
        video_path.write_bytes(b"")
        xml_path = tmp_path / "out.xml"

        generate_fcp7_xml(_sample_edl(), video_path, total_source_dur=30.0, output_xml_path=xml_path,
                           width=1920, height=1080, fps=24.0)

        root = ET.parse(xml_path).getroot()
        assert root.tag == "xmeml"
        # 每個 clip 各在 video 與 audio track 產生一個 clipitem
        assert len(root.findall(".//clipitem")) == len(_sample_edl()) * 2

    def test_timecode_conversion_is_frame_accurate(self, tmp_path):
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
        # 第二段接續在第一段之後的時間軸位置 (start = 第一段的 duration frames)
        assert int(second_clip.find("start").text) == round(15.0 * fps) - round(10.0 * fps)


class TestGenerateFcpxml:
    def test_output_is_well_formed_and_parseable(self, tmp_path):
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

    def test_duration_fraction_matches_seconds(self, tmp_path):
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
        # FCPXML 以 23.976fps (24000/1001) 幀為最小量化單位，容許誤差為一格幀長 (~0.0417s)。
        frame_duration = 1001 / 24000
        assert seconds == pytest.approx(2.0, abs=frame_duration)


class TestGenerateEdlCsv:
    def test_csv_contains_header_and_rows(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        generate_edl_csv(_sample_edl(), csv_path)

        content = csv_path.read_text(encoding="utf-8-sig")
        lines = content.strip().splitlines()
        assert lines[0].startswith("Clip_ID,Topic,Source_In")
        assert len(lines) == 1 + len(_sample_edl())
        assert "開場" in lines[1]

    def test_quotes_in_fields_are_escaped(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        edl = [{
            "clip_id": 1, "topic": "A", "source_in": 0.0, "source_out": 1.0, "duration": 1.0,
            "cps": 3.0, "in_margin": 0.1, "out_margin": 0.1,
            "transcript": '他說"你好"', "visual_check": "ok", "audio_check": "ok",
        }]
        generate_edl_csv(edl, csv_path)
        content = csv_path.read_text(encoding="utf-8-sig")
        assert '""你好""' in content

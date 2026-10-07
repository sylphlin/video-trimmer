"""NLE timeline and cut-table exporters: FCP 7 XML (.xml), Apple FCPXML (.fcpxml), and CSV (.csv).

ASD-STE100:
Generate frame-accurate NLE timelines for Adobe Premiere Pro, DaVinci Resolve, and Apple Final Cut Pro
across standard cinema and broadcast frame rates (23.976 to 60 fps).
"""

from pathlib import Path


def _is_ntsc_rate(fps: float) -> bool:
    """Return True when fps is a fractional NTSC rate (23.976, 29.97, 59.94)."""
    return abs(fps - round(fps)) > 0.001


def _fcpxml_rate_spec(fps: float) -> tuple[int, int, str]:
    """
    Return (frame_num, frame_den, format_tag) for FCPXML rational frame durations.

    Each frame duration equals frame_num / frame_den seconds.
    """
    timebase = int(round(fps))
    if _is_ntsc_rate(fps):
        if timebase == 24:
            return 1001, 24000, "2398"
        if timebase == 30:
            return 1001, 30000, "2997"
        if timebase == 60:
            return 1001, 60000, "5994"
        return 1001, max(1000, timebase * 1000), f"{timebase}p"
    return 100, max(100, timebase * 100), f"{timebase}p"


def generate_fcp7_xml(edl, video_path, total_source_dur, output_xml_path, width=1920, height=1080, fps=23.976):
    """Generate a Final Cut Pro 7 XML (xmeml v4) timeline for Premiere Pro and DaVinci Resolve."""
    timebase = max(1, int(round(fps)))
    ntsc_flag = "TRUE" if _is_ntsc_rate(fps) else "FALSE"

    def s2f(sec):
        return int(round(sec * fps))

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE xmeml>',
        '<xmeml version="4">',
        '  <project>',
        f'    <name>VideoTrimmer_{video_path.stem}</name>',
        '    <children>',
        '      <sequence id="sequence-1">',
        f'        <name>{video_path.stem}_TrimmerCut</name>',
        f'        <rate><timebase>{timebase}</timebase><ntsc>{ntsc_flag}</ntsc></rate>',
        '        <media>',
        '          <video>',
        '            <format>',
        '              <samplecharacteristics>',
        f'                <rate><timebase>{timebase}</timebase><ntsc>{ntsc_flag}</ntsc></rate>',
        f'                <width>{width}</width>',
        f'                <height>{height}</height>',
        '                <pixelaspectratio>square</pixelaspectratio>',
        '              </samplecharacteristics>',
        '            </format>',
        '            <track>'
    ]

    timeline_cursor = 0
    for idx, clip in enumerate(edl):
        in_frame = s2f(clip["source_in"])
        out_frame = s2f(clip["source_out"])
        dur_frame = max(1, out_frame - in_frame)
        start_frame = timeline_cursor
        end_frame = timeline_cursor + dur_frame
        timeline_cursor = end_frame

        xml_lines.extend([
            f'              <clipitem id="clipitem-v{idx+1}">',
            f'                <name>Clip_{clip["clip_id"]:02d}_{clip["topic"][:15]}</name>',
            f'                <rate><timebase>{timebase}</timebase><ntsc>{ntsc_flag}</ntsc></rate>',
            f'                <in>{in_frame}</in>',
            f'                <out>{out_frame}</out>',
            f'                <start>{start_frame}</start>',
            f'                <end>{end_frame}</end>',
            f'                <file id="file-1">',
            f'                  <name>{video_path.name}</name>',
            f'                  <pathurl>file://localhost{video_path.resolve()}</pathurl>',
            f'                  <rate><timebase>{timebase}</timebase><ntsc>{ntsc_flag}</ntsc></rate>',
            f'                  <duration>{s2f(total_source_dur)}</duration>',
            '                </file>',
            '              </clipitem>'
        ])

    xml_lines.extend([
        '            </track>',
        '          </video>',
        '          <audio>',
        '            <track>'
    ])

    timeline_cursor = 0
    for idx, clip in enumerate(edl):
        in_frame = s2f(clip["source_in"])
        out_frame = s2f(clip["source_out"])
        dur_frame = max(1, out_frame - in_frame)
        start_frame = timeline_cursor
        end_frame = timeline_cursor + dur_frame
        timeline_cursor = end_frame

        xml_lines.extend([
            f'              <clipitem id="clipitem-a{idx+1}">',
            f'                <name>Clip_{clip["clip_id"]:02d}_{clip["topic"][:15]}</name>',
            f'                <rate><timebase>{timebase}</timebase><ntsc>{ntsc_flag}</ntsc></rate>',
            f'                <in>{in_frame}</in>',
            f'                <out>{out_frame}</out>',
            f'                <start>{start_frame}</start>',
            f'                <end>{end_frame}</end>',
            f'                <file id="file-1"/>',
            '              </clipitem>'
        ])

    xml_lines.extend([
        '            </track>',
        '          </audio>',
        '        </media>',
        '      </sequence>',
        '    </children>',
        '  </project>',
        '</xmeml>'
    ])

    Path(output_xml_path).write_text('\n'.join(xml_lines), encoding='utf-8')


def generate_fcpxml(
    edl,
    video_path,
    total_source_dur,
    total_out_dur,
    output_fcpxml_path,
    fps=23.976,
    width=1920,
    height=1080,
):
    """Generate an Apple Final Cut Pro FCPXML (v1.9) timeline across 23.976 to 60 fps."""
    frame_num, frame_den, rate_tag = _fcpxml_rate_spec(fps)
    exact_fps = frame_den / frame_num

    def sec_to_frames(sec):
        return int(round(sec * exact_fps))

    def frames_to_fraction(frames):
        return f"{frames * frame_num}/{frame_den}s"

    total_src_frac = frames_to_fraction(sec_to_frames(total_source_dur))

    clip_elements = []
    offset_frames = 0
    for clip in edl:
        clip_dur = clip.get("duration", float(clip["source_out"]) - float(clip["source_in"]))
        start_frames = sec_to_frames(clip["source_in"])
        dur_frames = max(1, sec_to_frames(clip_dur))
        offset_frac = "0s" if offset_frames == 0 else frames_to_fraction(offset_frames)
        start_frac = frames_to_fraction(start_frames)
        dur_frac = frames_to_fraction(dur_frames)
        offset_frames += dur_frames
        name = f"Clip_{clip['clip_id']:02d}_{clip['topic'][:15]}"
        clip_elements.append(
            f'            <asset-clip name="{name}" ref="r2" offset="{offset_frac}" start="{start_frac}" duration="{dur_frac}"/>'
        )

    seq_frames = offset_frames if offset_frames > 0 else sec_to_frames(total_out_dur)
    total_out_frac = frames_to_fraction(seq_frames)

    xml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE fcpxml>',
        '<fcpxml version="1.9">',
        '  <resources>',
        f'    <format id="r1" name="FFVideoFormat{height}p{rate_tag}" frameDuration="{frame_num}/{frame_den}s" width="{width}" height="{height}"/>',
        f'    <asset id="r2" name="{video_path.name}" src="file://localhost{video_path.resolve()}" start="0s" duration="{total_src_frac}" hasVideo="1" hasAudio="1" format="r1"/>',
        '  </resources>',
        '  <library>',
        f'    <event name="VideoTrimmer_{video_path.stem}">',
        f'      <project name="{video_path.stem}_TrimmerCut">',
        f'        <sequence format="r1" duration="{total_out_frac}">',
        '          <spine>'
    ]
    xml.extend(clip_elements)
    xml.extend([
        '          </spine>',
        '        </sequence>',
        '      </project>',
        '    </event>',
        '  </library>',
        '</fcpxml>'
    ])

    Path(output_fcpxml_path).write_text('\n'.join(xml), encoding='utf-8')


def generate_edl_csv(edl, csv_path):
    """Write the EDL cut table as a UTF-8 with BOM CSV file."""
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("Clip_ID,Topic,Source_In,Source_Out,Duration,CPS,In_Margin,Out_Margin,Transcript,Visual_Check,Audio_Check\n")
        for c in edl:
            tr = c["transcript"].replace('"', '""')
            vc = c["visual_check"].replace('"', '""')
            ac = c["audio_check"].replace('"', '""')
            f.write(f'{c["clip_id"]},"{c["topic"]}",{c["source_in"]:.2f},{c["source_out"]:.2f},{c["duration"]:.2f},{c["cps"]:.2f},{c["in_margin"]:.2f},{c["out_margin"]:.2f},"{tr}","{vc}","{ac}"\n')

"""影片探測與 FFmpeg 最終渲染。"""

import json
import logging
import subprocess

from .constants import CROSSFADE_DURATION_SEC
from .exceptions import FFmpegError

logger = logging.getLogger(__name__)


def probe_video(video_path):
    """取得影片時長、解析度與幀率"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=codec_type,width,height,r_frame_rate",
        "-of", "json", str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise FFmpegError(f"ffprobe 錯誤: {res.stderr}")
    info = json.loads(res.stdout)
    duration = float(info["format"]["duration"])
    v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    width = int(v_stream.get("width", 1920))
    height = int(v_stream.get("height", 1080))
    fps_parts = v_stream.get("r_frame_rate", "24000/1001").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 23.976
    return duration, width, height, fps


def render_cut_video(edl, video_path, out_mp4_path, crf=18):
    """使用 FFmpeg 依據精確時間碼進行高畫質轉碼拼接 (無縫零跳幀)"""
    n = len(edl)
    if n == 0:
        logger.warning("無入選片段可供渲染。")
        return

    filter_complex = []
    for i, c in enumerate(edl):
        dur = c["duration"]
        fade_d = min(CROSSFADE_DURATION_SEC, max(0.005, dur / 4))
        fade_out_st = max(0.0, dur - fade_d)
        filter_complex.append(
            f"[0:v]trim=start={c['source_in']:.3f}:end={c['source_out']:.3f},setpts=PTS-STARTPTS[v{i}]; "
            f"[0:a]atrim=start={c['source_in']:.3f}:end={c['source_out']:.3f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d={fade_d:.3f},afade=t=out:st={fade_out_st:.3f}:d={fade_d:.3f}[a{i}];"
        )
    concat_inputs = "".join([f"[v{i}][a{i}]" for i in range(n)])
    filter_complex.append(f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-filter_complex", "".join(filter_complex),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "320k",
        "-movflags", "+faststart",
        str(out_mp4_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"FFmpeg 渲染失敗: {proc.stderr}")

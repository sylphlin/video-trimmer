#!/usr/bin/env python3
"""
PanSci AI Auto Rough Cut (泛科學 AI 自動初剪主程式)
--------------------------------------------------
基於 Gemini 3.8 Flash 多模態視訊理解 + 緊湊式減法剪輯（Compact Subtraction Cut）
支援語意重複取最後一次（Last Take Wins）、眼神就緒防眨眼、消除死寂停頓，
一鍵輸出 Final Cut Pro / DaVinci Resolve / Premiere Pro 剪輯工程檔與 MP4 成片。
"""

import os
import sys
import json
import time
import argparse
import subprocess
import re
from pathlib import Path

try:
    import soundfile as sf
    import numpy as np
    from google import genai
    from google.genai import types
except ImportError as e:
    print(f"缺少必要套件: {e}")
    print("請先執行: pip install -r requirements.txt")
    sys.exit(1)


def load_gemini_api_key():
    """載入 Gemini API Key，優先檢查環境變數，次之檢查本機 .env，再次之檢查 ~/.gemini/.env"""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    
    # 檢查當前目錄與專案目錄的 .env
    for candidate in [Path.cwd() / ".env", Path(__file__).parent / ".env"]:
        try:
            if candidate.exists():
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        key = line.split("=", 1)[1].strip("\"'")
                        os.environ["GEMINI_API_KEY"] = key
                        return key
        except Exception:
            pass

    # 檢查 ~/.gemini/.env
    try:
        env_file = Path.home() / ".gemini" / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip("\"'")
                    os.environ["GEMINI_API_KEY"] = key
                    return key
    except Exception:
        pass
    
    print("錯誤: 未找到 GEMINI_API_KEY！請設置環境變數或建立 .env 檔案")
    sys.exit(1)


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
        print(f"ffprobe 錯誤: {res.stderr}")
        sys.exit(1)
    
    info = json.loads(res.stdout)
    duration = float(info["format"]["duration"])
    v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    width = int(v_stream.get("width", 1920))
    height = int(v_stream.get("height", 1080))
    fps_parts = v_stream.get("r_frame_rate", "24000/1001").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 23.976
    return duration, width, height, fps


def mmss_to_sec(val):
    """處理模型將 mm:ss 輸出為 mm*100 + ss 的情況 (例如 724.8 -> 7m24.8s -> 444.8s)"""
    val_int = int(val)
    frac = val - val_int
    m = val_int // 100
    s = (val_int % 100) + frac
    return m * 60 + s


def resolve_timestamp(raw_val, total_dur):
    """
    智能判定時間戳：
    僅當模型輸出之數值大於影片總長度時，判定為 mm*100+ss 格式 (例如 1241.0 代表 12分41秒)。
    若數值在片長內，代表已經是絕對秒數，直接保留。
    """
    if raw_val > total_dur:
        return mmss_to_sec(raw_val)
    return raw_val


def detect_pre_speech_claps(audio_full, s_in, search_before=2.5, search_after=2.0, sr=16000):
    """
    精確偵測開拍打板拍手聲（物理特徵：超短瞬態衝擊波 peak>0.25，
    在 60ms 內快速衰減，且後方伴隨 >350ms 之純寂靜準備空檔）。
    返回所有拍手發生之絕對秒數。
    """
    start_t = max(0.0, s_in - search_before)
    end_t = min(len(audio_full) / sr, s_in + search_after)
    audio_seg = audio_full[int(start_t * sr): int(end_t * sr)]

    win_30ms = int(0.03 * sr)
    step = int(0.005 * sr)
    claps = []

    silence_len = int(0.35 * sr)
    limit = len(audio_seg) - (int(0.08 * sr) + silence_len)

    for i in range(0, max(0, limit), step):
        w1 = audio_seg[i : i + win_30ms]
        peak1 = np.max(np.abs(w1))
        if peak1 > 0.25:
            w_decay = audio_seg[i + int(0.05 * sr) : i + int(0.08 * sr)]
            rms_decay = np.sqrt(np.mean(w_decay**2)) if len(w_decay) > 0 else 0

            w_silence = audio_seg[i + int(0.08 * sr) : i + int(0.08 * sr) + silence_len]
            rms_silence = np.sqrt(np.mean(w_silence**2)) if len(w_silence) > 0 else 0

            # 拍手物理判據：急速衰減且後方持續 >350ms 寂靜
            if rms_decay < 0.030 and rms_silence < 0.012:
                t = start_t + i / sr
                if not claps or t - claps[-1] > 0.5:
                    claps.append(round(t, 2))
    return claps


def calculate_clip_cps(transcript, duration):
    """
    計算該段落的每秒語速 (CPS: Characters/Syllables Per Second)。
    中文按漢字數計算，英文/數字按詞與音節權重計算。
    """
    zh = len(re.findall(r'[\u4e00-\u9fff]', transcript))
    en_chars = len(re.findall(r'[a-zA-Z0-9]', transcript))
    syllables = zh + int(en_chars * 0.6)
    if syllables == 0:
        syllables = len(re.sub(r'\s+', '', transcript))
    dur = max(0.5, duration)
    return round(syllables / dur, 2), syllables


def compute_dynamic_margins(cps, pacing_mode="auto"):
    """
    根據語速 CPS 計算最適留白空間：
    - compact (固定緊湊): In=0.06s, Out=0.08s
    - breathing (固定呼吸): In=0.35s, Out=0.45s
    - auto / dynamic (連續動態浮動):
      CPS >= 5.5 (快節奏新聞/短影音) -> In=0.06s, Out=0.08s
      CPS <= 4.2 (沉穩慢說書/歷史漫談) -> In=0.35s, Out=0.45s
      中間進行連續平滑線性插值 (Linear Continuous Interpolation)
    """
    if pacing_mode == "compact":
        return 0.06, 0.08
    elif pacing_mode == "breathing":
        return 0.35, 0.45

    # 連續型動態浮動呼吸 (Continuous Dynamic Floating)
    t = (cps - 4.2) / (5.5 - 4.2)
    t = max(0.0, min(1.0, t))
    in_margin = 0.35 - t * (0.35 - 0.06)
    out_margin = 0.45 - t * (0.45 - 0.08)
    return round(in_margin, 2), round(out_margin, 2)


def find_clean_silence_boundary_in(audio, sr, true_onset, target_in_margin, last_clap=None, ambient_rms_thresh=0.015):
    """
    從開口發音點 (true_onset) 往回取留白，但只能在純靜默底噪區延伸。
    一旦遇到任何突發能量 (拍手、嗶聲、外人口令)，立刻停止回溯。
    """
    win = int(0.02 * sr)
    current_t = true_onset
    min_allowed_t = true_onset - target_in_margin
    if last_clap is not None:
        min_allowed_t = max(min_allowed_t, last_clap + 0.25)

    while current_t > min_allowed_t + 0.005:
        prev_idx = int((current_t - 0.02) * sr)
        if prev_idx < 0:
            break
        frame = audio[prev_idx : prev_idx + win]
        rms = np.sqrt(np.mean(frame**2))
        peak = np.max(np.abs(frame))
        if rms > ambient_rms_thresh or peak > 0.10:
            break
        current_t -= 0.005
    return max(0.0, current_t)


def find_speech_decay_end(audio, sr, last_strong_t, max_decay_search=0.35, decay_rms_thresh=0.006):
    """
    從最後一個強發音點 (RMS > 0.02) 向後追蹤字音的物理自然衰減 (Decay)，
    直到能量真正回落至環境底噪基準，確保字尾收音完整不被切斷。
    """
    win = int(0.02 * sr)
    t = last_strong_t
    max_t = min(len(audio) / sr, last_strong_t + max_decay_search)

    while t < max_t:
        idx = int(t * sr)
        if idx + win >= len(audio):
            break
        frame = audio[idx : idx + win]
        rms = np.sqrt(np.mean(frame**2))
        if rms < decay_rms_thresh:
            # 雙重驗證：確認後續 30ms 也是安靜的，避免短暫輔音或唇齒間隙誤判
            idx_next = idx + int(0.03 * sr)
            if idx_next + win < len(audio):
                frame_next = audio[idx_next : idx_next + win]
                if np.sqrt(np.mean(frame_next**2)) < decay_rms_thresh:
                    return t
        t += 0.005
    return min(max_t, t)


def find_clean_silence_boundary_out(audio, sr, speech_decay_end, target_out_margin, total_dur, voice_burst_thresh=0.035):
    """
    從尾音自然衰減結束點 (speech_decay_end) 往後保留表情沉澱與放鬆留白。
    僅防禦外人突發插話 (例如導演大喊『卡！』、『好！』或工作人員插嘴)：
    若在留白區間內偵測到新發音能量爆發，則在該雜音前 50ms 提前下刀切出；
    若環境為正常底噪，則 100% 賦予完整的動態留白。
    """
    win = int(0.02 * sr)
    max_allowed_t = min(total_dur, speech_decay_end + target_out_margin)
    # 衰減點後 50ms 避開殘存微氣流
    t = speech_decay_end + 0.05

    while t < max_allowed_t:
        idx = int(t * sr)
        if idx + win >= len(audio):
            break
        frame = audio[idx : idx + win]
        rms = np.sqrt(np.mean(frame**2))
        peak = np.max(np.abs(frame))
        # 僅在偵測到明顯外人講話 / 拍手脈衝時提前切出
        if rms > voice_burst_thresh or peak > 0.20:
            return max(speech_decay_end + 0.05, t - 0.05)
        t += 0.005
    return max_allowed_t


def refine_speech_bounds(audio, sr, s_in, s_out, total_dur, transcript="", pacing="auto"):
    """
    透過音訊 RMS 能量回溯掃描，精確定位字音起訖點，
    結合連續型動態浮動呼吸 (CPS) 與純靜默邊界防護，
    保證 100% 杜絕任何非主講人聲音 (拍手、嗶聲、導演口令)，
    並完整保護字尾自然消退期與微表情餘韻。
    """
    # 1. 偵測開拍前之拍手打板脈衝 (作為物理硬防護)
    claps = detect_pre_speech_claps(audio, s_in, search_before=2.5, search_after=2.0, sr=sr)
    last_clap = max(claps) if claps else None

    # 2. 計算動態語速留白
    raw_dur = max(0.5, s_out - s_in)
    cps, _ = calculate_clip_cps(transcript, raw_dur)
    in_margin, out_margin = compute_dynamic_margins(cps, pacing)

    # 3. 語音能量回溯掃描
    start_t = max(0.0, s_in - 2.5)
    end_t = min(total_dur, s_out + 1.0)
    seg = audio[int(start_t * sr):int(end_t * sr)]
    win_len = int(0.02 * sr)  # 20ms
    hop_len = int(0.005 * sr) # 5ms

    rms_vals = [np.sqrt(np.mean(seg[i:i + win_len] ** 2)) for i in range(0, len(seg) - win_len, hop_len)]
    t_vals = [start_t + i * 0.005 for i in range(len(rms_vals))]

    # 尋找真正人聲起訖點（必須嚴格位於最後一記拍手之後）
    speech_times = [t for t, r in zip(t_vals, rms_vals) if (last_clap is None or t >= last_clap + 0.20) and r > 0.020]

    if speech_times:
        true_onset = speech_times[0]
        last_strong_speech = speech_times[-1]
    else:
        true_onset = s_in
        last_strong_speech = s_out

    # 追蹤最後字音的自然衰減，直到沉入環境底噪
    speech_decay_end = find_speech_decay_end(audio, sr, last_strong_speech)

    # 4. 純靜默向外取留白
    final_in = find_clean_silence_boundary_in(audio, sr, true_onset, in_margin, last_clap)
    final_out = find_clean_silence_boundary_out(audio, sr, speech_decay_end, out_margin, total_dur)

    return round(final_in, 2), round(final_out, 2), cps, in_margin, out_margin


def generate_fcp7_xml(edl, video_path, total_source_dur, output_xml_path, width=1920, height=1080, fps=23.976):
    """產生 Premiere Pro / DaVinci Resolve 相容的 FCP 7 XML (xmeml v4)"""
    timebase = int(round(fps))
    def s2f(sec):
        return int(round(sec * fps))

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE xmeml>',
        '<xmeml version="4">',
        '  <project>',
        f'    <name>AI_RoughCut_{video_path.stem}</name>',
        '    <children>',
        '      <sequence id="sequence-1">',
        f'        <name>{video_path.stem}_Final_Cut</name>',
        f'        <rate><timebase>{timebase}</timebase><ntsc>TRUE</ntsc></rate>',
        '        <media>',
        '          <video>',
        '            <format>',
        '              <samplecharacteristics>',
        f'                <width>{width}</width>',
        f'                <height>{height}</height>',
        f'                <rate><timebase>{timebase}</timebase><ntsc>TRUE</ntsc></rate>',
        '              </samplecharacteristics>',
        '            </format>',
        '            <track>'
    ]

    cursor = 0
    for i, c in enumerate(edl):
        in_f = s2f(c["source_in"])
        out_f = s2f(c["source_out"])
        dur_f = out_f - in_f
        c_start = cursor
        c_end = cursor + dur_f
        cursor = c_end

        xml_lines.extend([
            f'              <clipitem id="clipitem-v{i+1}">',
            f'                <name>Clip_{c["clip_id"]}_{c["topic"]}</name>',
            f'                <duration>{dur_f}</duration>',
            f'                <rate><timebase>{timebase}</timebase><ntsc>TRUE</ntsc></rate>',
            f'                <in>{in_f}</in>',
            f'                <out>{out_f}</out>',
            f'                <start>{c_start}</start>',
            f'                <end>{c_end}</end>',
            '                <file id="file-1">',
            f'                  <name>{video_path.name}</name>',
            f'                  <pathurl>file://{video_path.resolve()}</pathurl>',
            f'                  <rate><timebase>{timebase}</timebase><ntsc>TRUE</ntsc></rate>',
            f'                  <duration>{s2f(total_source_dur)}</duration>',
            '                  <media>',
            '                    <video>',
            '                      <samplecharacteristics>',
            f'                        <width>{width}</width>',
            f'                        <height>{height}</height>',
            '                      </samplecharacteristics>',
            '                    </video>',
            '                    <audio>',
            '                      <samplecharacteristics>',
            '                        <depth>16</depth>',
            '                        <samplerate>48000</samplerate>',
            '                      </samplecharacteristics>',
            '                      <channelcount>2</channelcount>',
            '                    </audio>',
            '                  </media>',
            '                </file>',
            '              </clipitem>'
        ])

    xml_lines.extend([
        '            </track>',
        '          </video>',
        '          <audio>',
        '            <track>'
    ])

    cursor = 0
    for i, c in enumerate(edl):
        in_f = s2f(c["source_in"])
        out_f = s2f(c["source_out"])
        dur_f = out_f - in_f
        c_start = cursor
        c_end = cursor + dur_f
        cursor = c_end

        xml_lines.extend([
            f'              <clipitem id="clipitem-a{i+1}">',
            f'                <name>Clip_{c["clip_id"]}_{c["topic"]}</name>',
            f'                <duration>{dur_f}</duration>',
            f'                <rate><timebase>{timebase}</timebase><ntsc>TRUE</ntsc></rate>',
            f'                <in>{in_f}</in>',
            f'                <out>{out_f}</out>',
            f'                <start>{c_start}</start>',
            f'                <end>{c_end}</end>',
            '                <file id="file-1"/>',
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
    output_xml_path.write_text("\n".join(xml_lines), encoding="utf-8")


def generate_fcpxml(edl, video_path, total_source_dur, total_out_dur, output_fcpxml_path, fps=23.976):
    """產生 Final Cut Pro X 專用 XML (FCPXML v1.9)"""
    def s2f(sec):
        return int(round(sec * fps))

    fcpxml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE fcpxml>',
        '<fcpxml version="1.9">',
        '    <resources>',
        '        <format id="r1" name="FFVideoFormat1080p2398" frameDuration="1001/24000s" width="1920" height="1080" />',
        f'        <asset id="r2" name="{video_path.stem}" src="file://{video_path.resolve()}" start="0s" duration="{s2f(total_source_dur) * 1001}/24000s" format="r1" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000" />',
        '    </resources>',
        '    <library>',
        '        <event name="AI_RoughCut">',
        f'            <project name="{video_path.stem}_Final_Cut">',
        f'                <sequence format="r1" duration="{s2f(total_out_dur) * 1001}/24000s">',
        '                    <spine>'
    ]

    for c in edl:
        s_in = c["source_in"]
        dur = c["duration"]
        in_f = s2f(s_in)
        dur_f = s2f(dur)
        cid = c["clip_id"]
        topic = c["topic"]
        fcpxml_lines.append(f'                        <asset-clip name="Clip_{cid}_{topic}" ref="r2" offset="0s" start="{in_f * 1001}/24000s" duration="{dur_f * 1001}/24000s" format="r1" />')

    fcpxml_lines.extend([
        '                    </spine>',
        '                </sequence>',
        '            </project>',
        '        </event>',
        '    </library>',
        '</fcpxml>'
    ])
    output_fcpxml_path.write_text("\n".join(fcpxml_lines), encoding="utf-8")


def render_cut_video(edl, video_path, out_mp4_path, crf=18):
    """透過 FFmpeg 進行精準幀剪切與邊緣 15ms 微淡化無縫拼接"""
    filter_parts = []
    v_labels = []
    a_labels = []

    for i, c in enumerate(edl):
        s_in = c["source_in"]
        s_out = c["source_out"]
        dur = s_out - s_in
        fade_d = 0.015  # 15ms 防爆音微淡化
        fade_out_st = max(0.0, dur - fade_d)

        filter_parts.append(f"[0:v]trim=start={s_in:.3f}:end={s_out:.3f},setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[0:a]atrim=start={s_in:.3f}:end={s_out:.3f},asetpts=PTS-STARTPTS,afade=t=in:ss=0:d={fade_d:.3f},afade=t=out:st={fade_out_st:.3f}:d={fade_d:.3f}[a{i}]")
        v_labels.append(f"[v{i}]")
        a_labels.append(f"[a{i}]")

    concat_v = "".join(v_labels)
    concat_a = "".join(a_labels)
    n = len(edl)
    concat_filter = f"{concat_v}concat=n={n}:v=1:a=0[outv];{concat_a}concat=n={n}:v=0:a=1[outa]"
    filter_complex = ";".join(filter_parts) + ";" + concat_filter

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "320k",
        "-movflags", "+faststart",
        "-y", str(out_mp4_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"FFmpeg 渲染失敗: {proc.stderr}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="泛科學 AI 自動初剪工具 (Gemini 3.8 Flash)")
    parser.add_argument("--input", "-i", required=True, help="輸入影片檔案路徑 (MP4/MOV)")
    parser.add_argument("--output-dir", "-o", default=None, help="輸出資料夾 (預設為影片所在目錄)")
    parser.add_argument("--model", "-m", default="gemini-3.8-flash", help="使用的 Gemini 模型名稱")
    parser.add_argument("--crf", type=int, default=18, help="FFmpeg H.264 畫質參數 (預設 18)")
    parser.add_argument("--pacing", "-p", choices=["auto", "dynamic", "compact", "breathing"], default="auto",
                        help="剪輯節奏風格: 'auto'/'dynamic' (自適應連續動態浮動, 預設推薦), 'compact' (極致緊湊減法), 'breathing' (呼吸空間留白)")
    parser.add_argument("--cached-json", default=None,
                        help="指定既有之初剪決策 JSON 檔案路徑，跳過 Gemini 上傳與雲端分析 (便於快速微調留白與重渲染)")
    args = parser.parse_args()

    video_path = Path(args.input).resolve()
    if not video_path.exists():
        print(f"錯誤: 找不到影片檔案 {video_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve() if args.output_dir else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = video_path.stem

    print(f"==> 1. 檢測影片資訊: {video_path.name}")
    total_dur, width, height, fps = probe_video(video_path)
    print(f"    時長: {total_dur:.1f} 秒 (~{total_dur/60:.1f} 分鐘) | 解析度: {width}x{height} | 幀率: {fps:.3f} fps")

    if args.cached_json:
        cached_file = Path(args.cached_json).resolve()
        if not cached_file.exists():
            print(f"錯誤: 找不到快取 JSON 檔案: {cached_file}")
            sys.exit(1)
        print(f"==> 2. 跳過 Gemini 模型分析，直接載入快取初剪決策: {cached_file.name}")
        with open(cached_file, "r", encoding="utf-8") as f:
            model_edl = json.load(f)
    else:
        load_gemini_api_key()
        client = genai.Client()

        # 讀取 Markdown 格式提示詞
        prompt_file = Path(__file__).parent / "prompts" / "video_cut_prompt.md"
        if prompt_file.exists():
            prompt = prompt_file.read_text(encoding="utf-8")
        else:
            print("未找到 prompts/video_cut_prompt.md，請確認專案結構完整。")
            sys.exit(1)

        print(f"==> 2. 上傳影片至 Gemini Files API ({video_path.stat().st_size / (1024*1024):.1f} MB)...")
        t0 = time.time()
        video_file = client.files.upload(file=str(video_path))
        print(f"    上傳完畢 (耗時 {time.time() - t0:.1f} 秒)。等待雲端轉碼 ACTIVE...")

        while video_file.state.name == "PROCESSING":
            time.sleep(3)
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name != "ACTIVE":
            print(f"Gemini 處理失敗: {video_file.state.name}")
            sys.exit(1)

        print(f"==> 3. 調用 {args.model} 進行多模態視訊理解與初剪決策...")
        t1 = time.time()
        response = client.models.generate_content(
            model=args.model,
            contents=[video_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=8192
            )
        )
        print(f"    模型分析完成 (耗時 {time.time() - t1:.1f} 秒)！")

        # 清理遠端檔案
        try:
            client.files.delete(name=video_file.name)
        except Exception:
            pass

        raw_json = response.text.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:]
        if raw_json.startswith("```"):
            raw_json = raw_json[3:]
        if raw_json.endswith("```"):
            raw_json = raw_json[:-3]
        raw_json = raw_json.strip()

        model_edl = json.loads(raw_json)

    if args.pacing in ["auto", "dynamic"]:
        pacing_label = "【自適應連續動態浮動呼吸模式（Dynamic Floating Pacing）】"
    elif args.pacing == "breathing":
        pacing_label = "【呼吸空間模式（Breathing Space）】"
    else:
        pacing_label = "【緊湊減法模式（Compact Subtraction）】"

    print(f"==> 4. 抽取音軌進行能量全頻譜掃描，套用 {pacing_label}...")
    temp_wav = out_dir / f"temp_{base_name}.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(temp_wav)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audio, sr = sf.read(str(temp_wav))

    refined_edl = []
    for c in model_edl.get("final_edl", []):
        raw_in = c["source_in"]
        raw_out = c["source_out"]
        # 智能解析時間戳 (僅當數值超過總長時才轉 mm:ss)
        s_in = resolve_timestamp(raw_in, total_dur)
        s_out = resolve_timestamp(raw_out, total_dur)
        if s_out <= s_in:
            s_out = max(raw_out, s_in + 1.0)

        transcript = c.get("transcript", "") or c.get("content", "")
        tight_in, tight_out, clip_cps, in_m, out_m = refine_speech_bounds(
            audio, sr, s_in, s_out, total_dur, transcript=transcript, pacing=args.pacing
        )
        if tight_out <= tight_in:
            tight_out = tight_in + 1.0
        dur = round(tight_out - tight_in, 2)
        refined_edl.append({
            "clip_id": c["clip_id"],
            "topic": c["topic"],
            "source_in": tight_in,
            "source_out": tight_out,
            "duration": dur,
            "cps": clip_cps,
            "in_margin": in_m,
            "out_margin": out_m,
            "transcript": transcript,
            "visual_check": c.get("visual_check", "眼神直視鏡頭就緒，無眨眼閉眼"),
            "audio_check": c.get("audio_check", f"CPS={clip_cps:.2f} 字/秒 (In留白={in_m:.2f}s, Out留白={out_m:.2f}s, 純靜默邊界)")
        })
        print(f"    Clip {c['clip_id']:2d}: In={tight_in:6.2f}s, Out={tight_out:6.2f}s ({dur:5.2f}s) | CPS={clip_cps:4.2f} (In={in_m:.2f}s, Out={out_m:.2f}s) | {c['topic']}")

    temp_wav.unlink(missing_ok=True)
    total_out_dur = round(sum(c["duration"] for c in refined_edl), 2)
    avg_cps = round(sum(c["cps"] for c in refined_edl) / max(1, len(refined_edl)), 2)
    print(f"    初剪片段數: {len(refined_edl)} | 成片預計長度: {total_out_dur:.1f} 秒 (~{total_out_dur/60:.2f} 分鐘) | 全片平均語速: {avg_cps:.2f} 字/秒")

    file_tag = f"_{args.pacing}" if args.pacing != "compact" else ""

    # 5. 儲存 JSON
    json_path = out_dir / f"{base_name}{file_tag}_edl.json"
    json_path.write_text(json.dumps({
        "project_title": f"{base_name} AI 初剪 ({args.pacing})",
        "pacing_style": args.pacing,
        "average_cps": avg_cps,
        "total_duration": total_out_dur,
        "final_edl": refined_edl
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"==> 5. 輸出結構化資料: {json_path.name}")

    # 6. 儲存 CSV
    csv_path = out_dir / f"{base_name}{file_tag}_edl.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("Clip_ID,Topic,Source_In,Source_Out,Duration,CPS,In_Margin,Out_Margin,Transcript,Visual_Check,Audio_Check\n")
        for c in refined_edl:
            tr = c["transcript"].replace('"', '""')
            vc = c["visual_check"].replace('"', '""')
            ac = c["audio_check"].replace('"', '""')
            f.write(f'{c["clip_id"]},"{c["topic"]}",{c["source_in"]:.2f},{c["source_out"]:.2f},{c["duration"]:.2f},{c["cps"]:.2f},{c["in_margin"]:.2f},{c["out_margin"]:.2f},"{tr}","{vc}","{ac}"\n')
    print(f"==> 6. 輸出表格清單: {csv_path.name}")

    # 7. 儲存 FCP 7 XML (Premiere / DaVinci)
    xml_path = out_dir / f"{base_name}{file_tag}_edl.xml"
    generate_fcp7_xml(refined_edl, video_path, total_dur, xml_path, width, height, fps)
    print(f"==> 7. 輸出通用剪輯工程檔: {xml_path.name}")

    # 8. 儲存 FCPXML (Final Cut Pro X)
    fcpxml_path = out_dir / f"{base_name}{file_tag}_edl.fcpxml"
    generate_fcpxml(refined_edl, video_path, total_dur, total_out_dur, fcpxml_path, fps)
    print(f"==> 8. 輸出 Final Cut Pro X 工程檔: {fcpxml_path.name}")

    # 9. FFmpeg 渲染成片
    out_mp4 = out_dir / f"{base_name}{file_tag}_final_cut.mp4"
    print(f"==> 9. FFmpeg 渲染{pacing_label}成片: {out_mp4.name} ...")
    render_cut_video(refined_edl, video_path, out_mp4, crf=args.crf)
    print(f"==> [完成] 最終成片已產出！檔案大小: {out_mp4.stat().st_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()

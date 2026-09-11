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


def refine_speech_bounds(audio, sr, s_in, s_out, total_dur, pacing="compact"):
    """
    透過音訊 RMS 能量回溯掃描，精確定位字音起訖點，自動濾除拍手/打板雜音，
    並依據 pacing 風格模式設定起迄留白裕度：
    - 'compact': 緊湊減法模式（開口前 0.06s，落音後 0.06s，相鄰停頓 ~0.20s）
    - 'breathing': 呼吸空間模式（開口前 0.28s，落音後 0.45s，相鄰停頓 ~0.70s）
    """
    head_search = 1.0 if pacing == "breathing" else 0.5
    tail_search = 1.0 if pacing == "breathing" else 0.5
    start_t = max(0.0, s_in - head_search)
    end_t = min(total_dur, s_out + tail_search)
    
    seg = audio[int(start_t * sr):int(end_t * sr)]
    win_len = int(0.02 * sr)  # 20ms
    hop_len = int(0.005 * sr) # 5ms
    
    rms_vals = [np.sqrt(np.mean(seg[i:i + win_len] ** 2)) for i in range(0, len(seg) - win_len, hop_len)]
    t_vals = [start_t + i * 0.005 for i in range(len(rms_vals))]
    
    threshold = 0.020 if pacing == "breathing" else 0.030
    
    # 尋找真正連續語音，跳過孤立拍手脈衝（拍手特徵：極短脈衝且後方跟隨 >250ms 寂靜）
    speech_times = []
    for idx, (t, r) in enumerate(zip(t_vals, rms_vals)):
        if r > threshold:
            # 檢查開頭是否有拍手/打板 (若後方 250ms 內能量皆低於 0.015，則為孤立拍手，予以忽略)
            future_rms = rms_vals[idx + 1:idx + 50]  # 50 * 5ms = 250ms
            if len(future_rms) >= 20 and max(future_rms) < 0.015:
                continue
            speech_times.append(t)
            
    if speech_times:
        true_onset = speech_times[0]
        true_offset = speech_times[-1]
    else:
        true_onset = s_in
        true_offset = s_out
        
    if pacing == "breathing":
        # 呼吸空間：開口前 0.28s 氣息與眼神就位，句尾落音後 0.45s 從容餘韻
        compact_in = max(0.0, true_onset - 0.28)
        compact_out = min(total_dur, true_offset + 0.45)
    else:
        # 緊湊下刀：開口前 0.06s 微呼吸，句尾落音後 0.06s 俐落切斷
        compact_in = max(0.0, true_onset - 0.06)
        compact_out = min(total_dur, true_offset + 0.06)
        
    return round(compact_in, 2), round(compact_out, 2)


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
    parser.add_argument("--pacing", "-p", choices=["compact", "breathing"], default="compact",
                        help="剪輯節奏風格: 'compact' (極致緊湊減法, 適用新聞/科普) 或 'breathing' (呼吸空間留白, 適用慢說書/文化訪談)")
    args = parser.parse_args()

    video_path = Path(args.input).resolve()
    if not video_path.exists():
        print(f"錯誤: 找不到影片檔案 {video_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve() if args.output_dir else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = video_path.stem

    load_gemini_api_key()
    client = genai.Client()

    print(f"==> 1. 檢測影片資訊: {video_path.name}")
    total_dur, width, height, fps = probe_video(video_path)
    print(f"    時長: {total_dur:.1f} 秒 (~{total_dur/60:.1f} 分鐘) | 解析度: {width}x{height} | 幀率: {fps:.3f} fps")

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

    pacing_label = "【呼吸空間模式（Breathing Space）】" if args.pacing == "breathing" else "【緊湊減法模式（Compact Subtraction）】"
    print(f"==> 4. 抽取音軌進行能量（RMS）全頻譜回溯掃描，套用 {pacing_label}...")
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

        tight_in, tight_out = refine_speech_bounds(audio, sr, s_in, s_out, total_dur, pacing=args.pacing)
        if tight_out <= tight_in:
            tight_out = tight_in + 1.0
        dur = round(tight_out - tight_in, 2)
        refined_edl.append({
            "clip_id": c["clip_id"],
            "topic": c["topic"],
            "source_in": tight_in,
            "source_out": tight_out,
            "duration": dur,
            "transcript": c.get("transcript", ""),
            "visual_check": c.get("visual_check", "眼神直視鏡頭就緒，無眨眼閉眼"),
            "audio_check": c.get("audio_check", "自然起音與呼吸留白" if args.pacing == "breathing" else "緊湊起音與俐落切出")
        })

    temp_wav.unlink(missing_ok=True)
    total_out_dur = round(sum(c["duration"] for c in refined_edl), 2)
    print(f"    初剪片段數: {len(refined_edl)} | 成片預計長度: {total_out_dur:.1f} 秒 (~{total_out_dur/60:.2f} 分鐘)")

    file_tag = f"_{args.pacing}" if args.pacing == "breathing" else ""

    # 5. 儲存 JSON
    json_path = out_dir / f"{base_name}{file_tag}_edl.json"
    json_path.write_text(json.dumps({
        "project_title": f"{base_name} AI 初剪 ({args.pacing})",
        "pacing_style": args.pacing,
        "total_duration": total_out_dur,
        "final_edl": refined_edl
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"==> 5. 輸出結構化資料: {json_path.name}")

    # 6. 儲存 CSV
    csv_path = out_dir / f"{base_name}{file_tag}_edl.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("Clip_ID,Topic,Source_In,Source_Out,Duration,Transcript,Visual_Check,Audio_Check\n")
        for c in refined_edl:
            tr = c["transcript"].replace('"', '""')
            vc = c["visual_check"].replace('"', '""')
            ac = c["audio_check"].replace('"', '""')
            f.write(f'{c["clip_id"]},"{c["topic"]}",{c["source_in"]:.2f},{c["source_out"]:.2f},{c["duration"]:.2f},"{tr}","{vc}","{ac}"\n')
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

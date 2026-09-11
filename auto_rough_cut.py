#!/usr/bin/env python3
"""
PanSci AI Auto Rough Cut (泛科學 AI 自動初剪主程式)
--------------------------------------------------
基於 Whisper 微觀字級聲學時間戳對齊 (Word-level Ground Truth)
+ Gemini 3.8 Flash 多模態視訊理解與選鏡決策 (Multimodal Take Selection)
+ 文字剪輯時間鎖定 (Text-Based Locked Time Cutting)
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
import difflib
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

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


def load_gemini_api_key():
    """載入 Gemini API Key，優先檢查環境變數，次之檢查本機 .env，再次之檢查 ~/.gemini/.env"""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    
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
    """計算片段口播語速 CPS (Characters/Syllables Per Second)"""
    if not transcript or duration <= 0:
        return 3.0, 0
    zh = len(re.findall(r'[\u4e00-\u9fff]', transcript))
    en_chars = len(re.findall(r'[a-zA-Z0-9]', transcript))
    syllables = zh + int(en_chars * 0.6)
    if syllables == 0:
        syllables = len(re.sub(r'\s+', '', transcript))
    dur = max(0.5, duration)
    return round(syllables / dur, 2), syllables


def merge_whisper_segments_to_sentences(segments, max_gap=0.55, max_sentence_dur=20.0):
    """
    將 Whisper 零碎的聲學 Segments 依據自然換氣停頓與語法標點合併為完整語意句子。
    完整保留底層每個單詞的字級時間戳 (word_timestamps)，
    讓大模型在更高層級的完整語意單元 (Sentence/Take) 上進行挑選，杜絕漏選半截話或語意腰斬。
    """
    if not segments:
        return []

    CLOSURE_PUNCT = ('。', '！', '？', '!', '?', '……', '...')
    CONJUNCTIONS = ('但是', '而且', '所以', '然而', '如果', '因為', '不過', '雖然', '或是', '或者')

    sentences = []
    curr = None

    for s in segments:
        s_text = s['text'].strip()
        s_start = s['start']
        s_end = s['end']
        s_words = s.get('words', [])

        if curr is None:
            curr = {
                'id': 1,
                'start': s_start,
                'end': s_end,
                'text': s_text,
                'words': list(s_words),
                'orig_segment_ids': [s['id']]
            }
            continue

        gap = s_start - curr['end']
        curr_text = curr['text'].strip()
        dur = s_end - curr['start']

        should_split = False
        # 1. 物理換氣/停頓明顯 (gap >= 0.55s)
        if gap >= max_gap:
            should_split = True
        # 2. 句子長度已很長且有適度微停頓
        elif dur >= max_sentence_dur and gap >= 0.25:
            should_split = True
        # 3. 句尾標點閉合且微停頓
        elif any(curr_text.endswith(p) for p in CLOSURE_PUNCT) and gap >= 0.20:
            should_split = True

        # 連詞防斷保護 (若下個片段開頭是連詞，且未發生極長停頓，強制黏合)
        if should_split and gap < 0.90:
            if any(s_text.startswith(c) for c in CONJUNCTIONS):
                should_split = False

        if should_split:
            sentences.append(curr)
            curr = {
                'id': len(sentences) + 1,
                'start': s_start,
                'end': s_end,
                'text': s_text,
                'words': list(s_words),
                'orig_segment_ids': [s['id']]
            }
        else:
            curr['end'] = s_end
            curr['text'] = curr['text'] + ' ' + s_text if not curr['text'].endswith((' ', '，', '。')) else curr['text'] + s_text
            curr['words'].extend(s_words)
            curr['orig_segment_ids'].append(s['id'])

    if curr is not None:
        sentences.append(curr)

    return sentences


def transcribe_video_whisper(video_path, out_json_path, model_name="small"):
    """
    第一階段：微觀聲學時間戳對齊 (Word-level Ground Truth)
    在調用大模型之前，先在本地以 Whisper 生成帶有毫秒級字級時間戳的結構化劇本。
    返回: (raw_segments, merged_sentences)
    """
    out_json = Path(out_json_path)
    sentences_json = out_json.parent / f"{video_path.stem}_whisper_sentences.json"

    raw_results = []
    if out_json.exists():
        print(f"    載入既有之 Whisper 聲學時間戳: {out_json.name}")
        with open(out_json, "r", encoding="utf-8") as f:
            raw_results = json.load(f)
    elif HAS_WHISPER:
        print(f"    本地調用 faster-whisper ({model_name}) 進行微觀字級時間戳轉錄...")
        temp_wav = out_json.parent / f"temp_{video_path.stem}_whisper.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", str(temp_wav)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(temp_wav), word_timestamps=True, language="zh")

        for i, s in enumerate(segments):
            words_data = []
            if s.words:
                for w in s.words:
                    words_data.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2)
                    })
            raw_results.append({
                "id": i + 1,
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text.strip(),
                "words": words_data
            })

        temp_wav.unlink(missing_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, ensure_ascii=False, indent=2)
        print(f"    Whisper 轉錄完畢，共解析出 {len(raw_results)} 個高精度時間戳片段。")
    else:
        print("    [提示] 未安裝 faster-whisper，跳過本地字級轉錄。")
        return [], []

    # 執行語意句子合併
    if sentences_json.exists():
        with open(sentences_json, "r", encoding="utf-8") as f:
            sentences = json.load(f)
        print(f"    載入既有之語意句子劇本: {sentences_json.name} ({len(sentences)} 句)")
    else:
        sentences = merge_whisper_segments_to_sentences(raw_results)
        with open(sentences_json, "w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)
        print(f"    語意合併完成：從 {len(raw_results)} 個聲學碎片濃縮為 {len(sentences)} 個完整語意句子！")

    return raw_results, sentences


def format_whisper_transcript_for_prompt(whisper_sentences):
    """將 Whisper 合併後的語意句子格式化為 Gemini 提示詞專用的文字剪輯清單"""
    lines = [
        "\n---",
        "## Whisper 語意句子時間戳劇本 (Ground-Truth Semantic Sentences)",
        "以下是由本地微觀語音模型對本片轉錄、並依據自然換氣停頓與標點合併為完整語意的「句子清單」。",
        "每個 Sentence 均為一個完整的表達單元，包含精確起訖秒數（start -> end）。",
        "請在輸出 final_edl 時，直接引用選定片段對應的 `start_sentence_id` 與 `end_sentence_id`（亦相容 `start_segment_id`/`end_segment_id`），",
        "並將 `source_in` 與 `source_out` 對齊該句子之起訖時間（嚴禁自行粗估時間碼）：\n"
    ]
    for s in whisper_sentences:
        lines.append(f"[Sentence ID: {s['id']:3d}] {s['start']:6.2f}s -> {s['end']:6.2f}s | {s['text']}")
    return "\n".join(lines)


def normalize_text(text):
    return re.sub(r'[^\w\u4e00-\u9fff]', '', text).lower()


def align_clip_with_whisper(whisper_units, clip_data, total_dur):
    """
    比照字幕方式：嚴格以 Whisper 物理時間為準，不猜測、不更動字尾！
    優先採用 start_sentence_id / end_sentence_id 或 start_segment_id / end_segment_id，
    次之採用文本子字串比對，再次之採用時間窗。
    """
    if not whisper_units:
        return clip_data.get("source_in", 0), clip_data.get("source_out", total_dur)

    start_id = clip_data.get("start_sentence_id")
    if start_id is None:
        start_id = clip_data.get("start_segment_id")

    end_id = clip_data.get("end_sentence_id")
    if end_id is None:
        end_id = clip_data.get("end_segment_id")

    # 1. 優先使用 ID
    if start_id is not None and end_id is not None:
        matched = [s for s in whisper_units if start_id <= s["id"] <= end_id]
        if matched:
            first_words = matched[0].get("words", [])
            last_words = matched[-1].get("words", [])
            t_first = first_words[0]["start"] if first_words else matched[0]["start"]
            t_last = last_words[-1]["end"] if last_words else matched[-1]["end"]
            return t_first, t_last

    raw_in = clip_data.get("source_in", 0)
    raw_out = clip_data.get("source_out", total_dur)
    transcript = clip_data.get("transcript", "")

    # 2. 透過時間窗附近篩選
    cand_segs = [s for s in whisper_units if not (s["end"] < raw_in - 2.5 or s["start"] > raw_out + 2.5)]
    if not cand_segs:
        return raw_in, raw_out

    # 3. 逐字元建立時間線並做 SequenceMatcher
    char_timeline = []
    for s in cand_segs:
        words = s.get("words", [])
        if words:
            for w in words:
                w_norm = normalize_text(w["word"])
                if not w_norm:
                    continue
                w_dur = (w["end"] - w["start"]) / len(w_norm)
                for idx, ch in enumerate(w_norm):
                    ch_start = w["start"] + idx * w_dur
                    char_timeline.append((ch, round(ch_start, 3), round(ch_start + w_dur, 3)))
        else:
            s_norm = normalize_text(s["text"])
            if s_norm:
                s_dur = (s["end"] - s["start"]) / len(s_norm)
                for idx, ch in enumerate(s_norm):
                    ch_start = s["start"] + idx * s_dur
                    char_timeline.append((ch, round(ch_start, 3), round(ch_start + s_dur, 3)))

    tgt_norm = normalize_text(transcript)
    whisper_str = ''.join([item[0] for item in char_timeline])

    if tgt_norm and whisper_str and char_timeline:
        matcher = difflib.SequenceMatcher(None, tgt_norm, whisper_str)
        blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
        if blocks:
            first_b = blocks[0]
            last_b = blocks[-1]
            t_first = char_timeline[first_b.b][1]
            t_last = char_timeline[min(len(char_timeline) - 1, last_b.b + last_b.size - 1)][2]
            return t_first, t_last

    # 若匹配落空，返回重疊 segment 範圍
    return cand_segs[0]["start"], cand_segs[-1]["end"]


def compute_dynamic_margins(cps, pacing="auto"):
    """
    依據語速 CPS (字/秒) 動態計算自然呼吸與收口留白：
    - 快語速 (CPS >= 5.5，如泛科學 6.0~8.0 字/秒)：
      In 留白 0.06s 微氣息，Out 留白 0.12s 俐落定格（緊湊有勁，無冷場拖沓）
    - 慢語速 (CPS <= 3.0，如工頭堅 2.5~3.0 字/秒)：
      In 留白 0.15s 定神氣息，Out 留白 0.28s 表情餘韻（沉穩舒緩）
    - 中間語速：連續線性插值
    """
    if pacing == "compact":
        return 0.06, 0.10
    elif pacing == "breathing":
        return 0.25, 0.35

    if cps >= 5.5:
        in_m = 0.06
        out_m = 0.12
    elif cps <= 3.0:
        in_m = 0.15
        out_m = 0.28
    else:
        alpha = (cps - 3.0) / (5.5 - 3.0)
        in_m = 0.15 - alpha * (0.15 - 0.06)
        out_m = 0.28 - alpha * (0.28 - 0.12)
    return round(in_m, 2), round(out_m, 2)


def refine_speech_bounds_locked(audio, sr, t_first, t_last, total_dur, transcript="", pacing="auto"):
    """
    文字剪輯時間鎖定標準 (Text-Based Locked Time Standard):
    - 嚴格尊重 Whisper 物理發音時間戳，絕不提前切斷字尾！
    - 依據 CPS 連續動態分配呼吸氣息與閉口留白 (快語速緊湊 0.06/0.12s，慢語速舒緩 0.15/0.28s)
    - 開頭：保留動態前置呼吸氣息，但以開拍前拍手打板點為絕對硬防護阻斷 (Physical Clap Guard)。
    - 結尾：在最後一個字發音結束後，保留動態俐落自然閉口定格，流暢緊湊不拖沓。
    """
    # 1. 偵測開拍前之拍手打板脈衝
    claps = detect_pre_speech_claps(audio, t_first, search_before=2.5, search_after=1.5, sr=sr)
    last_clap = max(claps) if claps else None

    # 2. 計算片段語速 (CPS) 與自適應動態留白
    raw_dur = max(0.5, t_last - t_first)
    cps, _ = calculate_clip_cps(transcript, raw_dur)
    in_margin, out_margin = compute_dynamic_margins(cps, pacing)

    # 3. 句首下刀點 (氣息留白，拍手防護)
    if last_clap is not None and last_clap < t_first:
        final_in = max(last_clap + 0.10, t_first - in_margin)
    else:
        final_in = max(0.0, t_first - in_margin)

    # 4. 句尾下刀點 (動態俐落收口定格)
    max_out = min(total_dur, t_last + out_margin)
    win_len = int(0.02 * sr)
    t = t_last + 0.03
    final_out = max_out

    while t < max_out:
        idx = int(t * sr)
        if idx + win_len >= len(audio):
            break
        frame = audio[idx : idx + win_len]
        peak = np.max(np.abs(frame))
        rms = np.sqrt(np.mean(frame**2))
        # 僅在偵測到極大能量突發（外人搶話/拍手）時提前停止
        if peak > 0.25 or rms > 0.040:
            final_out = max(t_last + 0.03, t - 0.03)
            break
        t += 0.005

    in_m = round(t_first - final_in, 2)
    out_m = round(final_out - t_last, 2)

    return round(final_in, 2), round(final_out, 2), cps, in_m, out_m


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
        f'                <rate><timebase>{timebase}</timebase><ntsc>TRUE</ntsc></rate>',
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
            f'                <rate><timebase>{timebase}</timebase><ntsc>TRUE</ntsc></rate>',
            f'                <in>{in_frame}</in>',
            f'                <out>{out_frame}</out>',
            f'                <start>{start_frame}</start>',
            f'                <end>{end_frame}</end>',
            f'                <file id="file-1">',
            f'                  <name>{video_path.name}</name>',
            f'                  <pathurl>file://localhost{video_path.resolve()}</pathurl>',
            f'                  <rate><timebase>{timebase}</timebase><ntsc>TRUE</ntsc></rate>',
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
            f'                <rate><timebase>{timebase}</timebase><ntsc>TRUE</ntsc></rate>',
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


def generate_fcpxml(edl, video_path, total_source_dur, total_out_dur, output_fcpxml_path, fps=23.976):
    """產生 Final Cut Pro X 相容的 FCPXML (v1.9)"""
    def sec_to_fraction(sec):
        frames = int(round(sec * 24000 / 1001))
        return f"{frames * 1001}/24000s"

    total_out_frac = sec_to_fraction(total_out_dur)
    total_src_frac = sec_to_fraction(total_source_dur)

    xml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE fcpxml>',
        '<fcpxml version="1.9">',
        '  <resources>',
        '    <format id="r1" name="FFVideoFormat1080p2398" frameDuration="1001/24000s" width="1920" height="1080"/>',
        f'    <asset id="r2" name="{video_path.name}" src="file://localhost{video_path.resolve()}" start="0s" duration="{total_src_frac}" hasVideo="1" hasAudio="1" format="r1"/>',
        '  </resources>',
        '  <library>',
        f'    <event name="AI_RoughCut_{video_path.stem}">',
        f'      <project name="{video_path.stem}_RoughCut">',
        f'        <sequence format="r1" duration="{total_out_frac}">',
        '          <spine>'
    ]

    for clip in edl:
        clip_dur = clip["duration"]
        start_frac = sec_to_fraction(clip["source_in"])
        dur_frac = sec_to_fraction(clip_dur)
        name = f"Clip_{clip['clip_id']:02d}_{clip['topic'][:15]}"
        xml.append(f'            <asset-clip name="{name}" ref="r2" offset="0s" start="{start_frac}" duration="{dur_frac}"/>')

    xml.extend([
        '          </spine>',
        '        </sequence>',
        '      </project>',
        '    </event>',
        '  </library>',
        '</fcpxml>'
    ])

    Path(output_fcpxml_path).write_text('\n'.join(xml), encoding='utf-8')


def render_cut_video(edl, video_path, out_mp4_path, crf=18):
    """使用 FFmpeg 依據精確時間碼進行高畫質轉碼拼接 (無縫零跳幀)"""
    n = len(edl)
    if n == 0:
        print("警告: 無入選片段可供渲染。")
        return

    filter_complex = []
    for i, c in enumerate(edl):
        filter_complex.append(
            f"[0:v]trim=start={c['source_in']:.3f}:end={c['source_out']:.3f},setpts=PTS-STARTPTS[v{i}]; "
            f"[0:a]atrim=start={c['source_in']:.3f}:end={c['source_out']:.3f},asetpts=PTS-STARTPTS[a{i}];"
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
        print(f"FFmpeg 渲染失敗: {proc.stderr}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="泛科學 AI 自動初剪工具 (Whisper Ground-Truth + Gemini 3.8 Flash)")
    parser.add_argument("--input", "-i", required=True, help="輸入影片檔案路徑 (MP4/MOV)")
    parser.add_argument("--output-dir", "-o", default=None, help="輸出資料夾 (預設為影片所在目錄)")
    parser.add_argument("--model", "-m", default="gemini-3.8-flash", help="使用的 Gemini 模型名稱")
    parser.add_argument("--crf", type=int, default=18, help="FFmpeg H.264 畫質參數 (預設 18)")
    parser.add_argument("--pacing", "-p", choices=["auto", "dynamic", "compact", "breathing"], default="auto",
                        help="剪輯節奏風格: 'auto'/'dynamic' (文字剪輯微氣息鎖定, 推薦預設)")
    parser.add_argument("--agentic", action="store_true", help="啟用 Gemini Agentic Video Understanding 動態探索模式")
    parser.add_argument("--suffix", default=None, help="自訂輸出檔案名稱標籤後綴 (預設為 agentic 或 static)")
    parser.add_argument("--script", "-s", default=None, help="可選的分鏡講稿或文本檔案路徑 (TXT/MD)")
    parser.add_argument("--cached-json", default=None,
                        help="指定既有之初剪決策 JSON 檔案路徑，跳過 Gemini 上傳與雲端分析")
    parser.add_argument("--skip-whisper", action="store_true", help="跳過本地 Whisper 轉錄，僅使用純能量檢測")
    args = parser.parse_args()

    video_path = Path(args.input).resolve()
    if not video_path.exists():
        print(f"錯誤: 找不到影片檔案 {video_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve() if args.output_dir else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = video_path.stem

    # 決定輸出檔案標籤
    tag = args.suffix if args.suffix else ("agentic" if args.agentic else "static")

    print(f"==> 1. 檢測影片資訊: {video_path.name}")
    total_dur, width, height, fps = probe_video(video_path)
    print(f"    時長: {total_dur:.1f} 秒 (~{total_dur/60:.1f} 分鐘) | 解析度: {width}x{height} | 幀率: {fps:.3f} fps")

    # 第一階段：Whisper 微觀聲學時間戳對齊 (Word-level Ground Truth) 與語意句子合併
    whisper_segs = []
    whisper_sentences = []
    if not args.skip_whisper:
        whisper_json = out_dir / f"{base_name}_whisper_raw.json"
        print(f"==> 2. 執行 Whisper 微觀字級聲學時間戳轉錄與語意句子合併...")
        whisper_segs, whisper_sentences = transcribe_video_whisper(video_path, whisper_json, model_name="small")

    whisper_units = whisper_sentences if whisper_sentences else whisper_segs

    # 第二階段：Gemini 宏觀多模態視訊理解與選鏡決策
    inference_metrics = {}
    if args.cached_json:
        cached_file = Path(args.cached_json).resolve()
        if not cached_file.exists():
            print(f"錯誤: 找不到快取 JSON 檔案: {cached_file}")
            sys.exit(1)
        print(f"==> 3. 跳過 Gemini 模型分析，直接載入快取初剪決策: {cached_file.name}")
        with open(cached_file, "r", encoding="utf-8") as f:
            model_edl = json.load(f)
    else:
        load_gemini_api_key()
        client = genai.Client()

        prompt_file = Path(__file__).parent / "prompts" / "video_cut_prompt.md"
        if not prompt_file.exists():
            prompt_file = Path(__file__).parent / "pansci-ai-rough-cut" / "prompts" / "video_cut_prompt.md"
        if prompt_file.exists():
            prompt = prompt_file.read_text(encoding="utf-8")
        else:
            print("未找到 prompts/video_cut_prompt.md，請確認專案結構完整。")
            sys.exit(1)

        # 附上分鏡講稿 (若有)
        if args.script:
            script_path = Path(args.script).resolve()
            if script_path.exists():
                prompt += f"\n\n---\n## 參考講稿 (Production Script)\n{script_path.read_text(encoding='utf-8')}\n"

        # 附上 Whisper 語意劇本
        if whisper_units:
            prompt += format_whisper_transcript_for_prompt(whisper_units)

        # 影片上傳或快取重用
        upload_cache_path = out_dir / f".{video_path.name}_upload_cache.json"
        video_file = None
        if upload_cache_path.exists():
            try:
                cached_info = json.loads(upload_cache_path.read_text(encoding="utf-8"))
                cached_name = cached_info.get("name")
                check_file = client.files.get(name=cached_name)
                if check_file.state.name == "ACTIVE":
                    video_file = check_file
                    print(f"    ✓ 重用快取之雲端視訊 (File ID: {video_file.name})")
            except Exception:
                pass

        if video_file is None:
            print(f"==> 3. 上傳影片至 Gemini Files API ({video_path.stat().st_size / (1024*1024):.1f} MB)...")
            t0 = time.time()
            video_file = client.files.upload(file=str(video_path))
            print(f"    上傳完畢 (耗時 {time.time() - t0:.1f} 秒)。等待雲端轉碼 ACTIVE...")

            while video_file.state.name == "PROCESSING":
                time.sleep(3)
                video_file = client.files.get(name=video_file.name)

            if video_file.state.name != "ACTIVE":
                print(f"Gemini 處理失敗: {video_file.state.name}")
                sys.exit(1)

            try:
                upload_cache_path.write_text(json.dumps({
                    "name": video_file.name,
                    "uri": video_file.uri,
                    "timestamp": time.time()
                }), encoding="utf-8")
            except Exception:
                pass

        mime_type = video_file.mime_type or "video/mp4"
        mode_str = "🤖 Agentic Video Understanding (Interactions API)" if args.agentic else "📺 Static Multimodal (靜態抽幀)"
        print(f"==> 4. 調用 {args.model} [模式: {mode_str}] 結合 Whisper 劇本進行文字剪輯與選鏡決策...")

        t1 = time.time()
        raw_json = ""
        prompt_tokens = 0
        candidates_tokens = 0
        thoughts_tokens = 0
        total_tokens = 0
        tool_tokens = 0

        if args.agentic:
            try:
                interaction = client.interactions.create(
                    model=args.model,
                    input=[
                        {"type": "video", "uri": video_file.uri, "processing": "agentic"},
                        {"type": "text", "text": prompt}
                    ]
                )
                raw_json = interaction.output_text or ""
                usage = getattr(interaction, "usage", None)
                if usage:
                    prompt_tokens = getattr(usage, "total_input_tokens", 0) or 0
                    candidates_tokens = getattr(usage, "total_output_tokens", 0) or 0
                    thoughts_tokens = getattr(usage, "total_thought_tokens", 0) or 0
                    total_tokens = getattr(usage, "total_tokens", 0) or 0
                    tool_tokens = getattr(usage, "total_tool_use_tokens", 0) or 0
            except Exception as e:
                print(f"    Interactions API 遭遇異常 ({e})，降級調用 models.generate_content...")
                video_part = types.Part(
                    file_data=types.FileData(file_uri=video_file.uri, mime_type=mime_type),
                    media_processing=types.MediaProcessing.AGENTIC
                )
                response = client.models.generate_content(
                    model=args.model,
                    contents=[video_part, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                        max_output_tokens=8192
                    )
                )
                if hasattr(response, "text") and response.text:
                    raw_json = response.text
                elif hasattr(response, "candidates") and response.candidates:
                    parts = getattr(response.candidates[0].content, "parts", [])
                    raw_json = "\n".join([p.text for p in parts if hasattr(p, "text") and p.text])
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
                    total_tokens = getattr(usage, "total_token_count", 0) or 0
                    thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0
        else:
            video_part = types.Part(
                file_data=types.FileData(file_uri=video_file.uri, mime_type=mime_type)
            )
            response = client.models.generate_content(
                model=args.model,
                contents=[video_part, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=8192
                )
            )
            if hasattr(response, "text") and response.text:
                raw_json = response.text
            elif hasattr(response, "candidates") and response.candidates:
                parts = getattr(response.candidates[0].content, "parts", [])
                raw_json = "\n".join([p.text for p in parts if hasattr(p, "text") and p.text])
            usage = getattr(response, "usage_metadata", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
                total_tokens = getattr(usage, "total_token_count", 0) or 0
                thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0

        duration = time.time() - t1
        print(f"    模型推論完成 (耗時 {duration:.1f} 秒)！")
        print(f"    📊 Token 統計: 輸入={prompt_tokens:,} | 輸出={candidates_tokens:,} | 思維={thoughts_tokens:,} | 工具探索={tool_tokens:,} | 總計={total_tokens:,}")

        inference_metrics = {
            "mode": "agentic" if args.agentic else "static",
            "duration_seconds": round(duration, 2),
            "prompt_tokens": prompt_tokens,
            "candidates_tokens": candidates_tokens,
            "thoughts_tokens": thoughts_tokens,
            "tool_use_tokens": tool_tokens,
            "total_tokens": total_tokens
        }

        raw_json = raw_json.strip()
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0].strip()
        elif not raw_json.startswith("{") and "{" in raw_json:
            start_idx = raw_json.find("{")
            end_idx = raw_json.rfind("}")
            if start_idx != -1 and end_idx != -1:
                raw_json = raw_json[start_idx:end_idx+1].strip()

        try:
            model_edl = json.loads(raw_json)
        except Exception as e:
            print(f"解析 Gemini JSON 失敗: {e}")
            print(f"原始回傳內容:\n{raw_json}")
            sys.exit(1)

    # 第三階段：比照字幕邏輯——時間鎖定與拍手防護
    print(f"==> 5. 執行文字剪輯時間鎖定 (Text-Based Locked Timestamps) 與拍手防護...")
    temp_wav = out_dir / f"temp_{base_name}.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(temp_wav)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audio, sr = sf.read(str(temp_wav))

    refined_edl = []
    for c in model_edl.get("final_edl", []):
        t_first, t_last = align_clip_with_whisper(whisper_units, c, total_dur)
        transcript = c.get("transcript", "") or c.get("content", "")

        tight_in, tight_out, clip_cps, in_m, out_m = refine_speech_bounds_locked(
            audio, sr, t_first, t_last, total_dur, transcript=transcript, pacing=args.pacing
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
            "take_selection_reason": c.get("take_selection_reason", ""),
            "visual_check": c.get("visual_check", "眼神直視鏡頭就緒，無眨眼閉眼"),
            "audio_check": c.get("audio_check", f"Whisper精準錨定 (In前置氣息={in_m:.2f}s, Out俐落收口={out_m:.2f}s)")
        })
        print(f"    Clip {c['clip_id']:2d}: In={tight_in:6.2f}s, Out={tight_out:6.2f}s ({dur:5.2f}s) | 首字: {t_first:.2f}s, 尾字: {t_last:.2f}s | {c['topic']}")

    temp_wav.unlink(missing_ok=True)
    total_out_dur = round(sum(c["duration"] for c in refined_edl), 2)
    avg_cps = round(sum(c["cps"] for c in refined_edl) / max(1, len(refined_edl)), 2)
    print(f"    初剪片段數: {len(refined_edl)} | 成片預計長度: {total_out_dur:.1f} 秒 (~{total_out_dur/60:.2f} 分鐘) | 全片平均語速: {avg_cps:.2f} 字/秒")

    # 輸出資料
    json_path = out_dir / f"{base_name}_{tag}_edl.json"
    json_path.write_text(json.dumps({
        "project_title": f"{base_name} AI 文字剪輯初剪 ({tag})",
        "pacing_style": "text_based_whisper_grounded",
        "inference_metrics": inference_metrics,
        "average_cps": avg_cps,
        "total_duration": total_out_dur,
        "final_edl": refined_edl
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"==> 6. 輸出結構化資料: {json_path.name}")

    csv_path = out_dir / f"{base_name}_{tag}_edl.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("Clip_ID,Topic,Source_In,Source_Out,Duration,CPS,In_Margin,Out_Margin,Transcript,Visual_Check,Audio_Check\n")
        for c in refined_edl:
            tr = c["transcript"].replace('"', '""')
            vc = c["visual_check"].replace('"', '""')
            ac = c["audio_check"].replace('"', '""')
            f.write(f'{c["clip_id"]},"{c["topic"]}",{c["source_in"]:.2f},{c["source_out"]:.2f},{c["duration"]:.2f},{c["cps"]:.2f},{c["in_margin"]:.2f},{c["out_margin"]:.2f},"{tr}","{vc}","{ac}"\n')
    print(f"==> 7. 輸出表格清單: {csv_path.name}")

    xml_path = out_dir / f"{base_name}_{tag}_edl.xml"
    generate_fcp7_xml(refined_edl, video_path, total_dur, xml_path, width, height, fps)
    print(f"==> 8. 輸出通用剪輯工程檔: {xml_path.name}")

    fcpxml_path = out_dir / f"{base_name}_{tag}_edl.fcpxml"
    generate_fcpxml(refined_edl, video_path, total_dur, total_out_dur, fcpxml_path, fps)
    print(f"==> 9. 輸出 Final Cut Pro X 工程檔: {fcpxml_path.name}")

    out_mp4 = out_dir / f"{base_name}_{tag}_rough_cut.mp4"
    print(f"==> 10. FFmpeg 渲染成片: {out_mp4.name} ...")
    render_cut_video(refined_edl, video_path, out_mp4, crf=args.crf)
    print(f"==> [完成] 最終成片已產出！檔案大小: {out_mp4.stat().st_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()

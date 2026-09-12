"""Whisper 轉錄、語意句子合併與文字對齊。"""

import difflib
import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import mlx_whisper
    HAS_MLX_WHISPER = True
except ImportError:
    HAS_MLX_WHISPER = False

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

HAS_WHISPER = HAS_MLX_WHISPER or HAS_FASTER_WHISPER


def normalize_text(text):
    return re.sub(r'[^\w一-鿿]', '', text).lower()


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

        # 4. 場記報幕代號或語系切換防黏合 (例如 BDA82 / CTA 報幕與正片日語/中文分離)
        is_curr_slate = bool(re.match(r'^[A-Za-z0-9\s\-_]+$', curr_text))
        is_s_japanese = bool(re.search(r'[぀-ヿ]', s_text))
        is_curr_japanese = bool(re.search(r'[぀-ヿ]', curr_text))
        if is_curr_slate and not bool(re.match(r'^[A-Za-z0-9\s\-_]+$', s_text)):
            should_split = True
        elif is_curr_japanese != is_s_japanese:
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
        logger.info("載入既有之 Whisper 聲學時間戳: %s", out_json.name)
        with open(out_json, "r", encoding="utf-8") as f:
            raw_results = json.load(f)
    elif HAS_WHISPER:
        temp_wav = out_json.parent / f"temp_{video_path.stem}_whisper.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", str(temp_wav)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if HAS_MLX_WHISPER:
            logger.info("本地調用 mlx-whisper (%s) [Apple Silicon Metal GPU 原生加速] 進行微觀字級時間戳轉錄...", model_name)
            mlx_repo = f"mlx-community/whisper-{model_name}-mlx"
            res = mlx_whisper.transcribe(str(temp_wav), path_or_hf_repo=mlx_repo, word_timestamps=True, language="zh")
            for i, s in enumerate(res.get("segments", [])):
                words_data = []
                for w in s.get("words", []):
                    words_data.append({
                        "word": w["word"].strip(),
                        "start": round(float(w["start"]), 2),
                        "end": round(float(w["end"]), 2)
                    })
                raw_results.append({
                    "id": i + 1,
                    "start": round(float(s["start"]), 2),
                    "end": round(float(s["end"]), 2),
                    "text": s["text"].strip(),
                    "words": words_data
                })
        else:
            logger.info("本地調用 faster-whisper (%s) 進行微觀字級時間戳轉錄...", model_name)
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
        logger.info("Whisper 轉錄完畢，共解析出 %d 個高精度時間戳片段。", len(raw_results))
    else:
        logger.info("[提示] 未安裝 mlx-whisper 或 faster-whisper，跳過本地字級轉錄。")
        return [], []

    # 執行語意句子合併
    if sentences_json.exists():
        with open(sentences_json, "r", encoding="utf-8") as f:
            sentences = json.load(f)
        logger.info("載入既有之語意句子劇本: %s (%d 句)", sentences_json.name, len(sentences))
    else:
        sentences = merge_whisper_segments_to_sentences(raw_results)
        with open(sentences_json, "w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)
        logger.info("語意合併完成：從 %d 個聲學碎片濃縮為 %d 個完整語意句子！", len(raw_results), len(sentences))

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


def _find_index(units, item):
    for i, u in enumerate(units):
        if u is item:
            return i
    return None


def _neighbor_bounds(whisper_units, first_item, last_item):
    """回傳 (prev_sentence_end, next_sentence_start)：相鄰句子的邊界，供聲學搜尋範圍上限使用。"""
    idx_first = _find_index(whisper_units, first_item)
    idx_last = _find_index(whisper_units, last_item)
    prev_end = None
    next_start = None
    if idx_first is not None and idx_first > 0:
        prev_end = whisper_units[idx_first - 1]["end"]
    if idx_last is not None and idx_last + 1 < len(whisper_units):
        next_start = whisper_units[idx_last + 1]["start"]
    return prev_end, next_start


def align_clip_with_whisper(whisper_units, clip_data, total_dur):
    """
    比照字幕方式：嚴格以 Whisper 物理時間為準，不猜測、不更動字尾！
    優先採用 start_sentence_id / end_sentence_id 或 start_segment_id / end_segment_id，
    次之採用文本子字串比對，再次之採用時間窗。

    回傳: (t_first, t_last, prev_sentence_end, next_sentence_start)
    prev_sentence_end / next_sentence_start 為相鄰 Whisper 句子的邊界（無相鄰句子時為 None），
    供 acoustic.refine_speech_bounds_locked 做結構性搜尋範圍上限。
    """
    if not whisper_units:
        return clip_data.get("source_in", 0), clip_data.get("source_out", total_dur), None, None

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
            prev_end, next_start = _neighbor_bounds(whisper_units, matched[0], matched[-1])
            transcript = clip_data.get("transcript", "") or clip_data.get("content", "")
            # 若提供了明確台詞文字，且句子內有詳細字級時間戳，精確對齊至台詞起訖字
            if transcript:
                matched_words = []
                for s in matched:
                    matched_words.extend(s.get("words", []))
                char_timeline = []
                for w in matched_words:
                    w_norm = normalize_text(w["word"])
                    if not w_norm:
                        continue
                    w_dur = (w["end"] - w["start"]) / len(w_norm)
                    for idx, ch in enumerate(w_norm):
                        ch_start = w["start"] + idx * w_dur
                        char_timeline.append((ch, round(ch_start, 3), round(ch_start + w_dur, 3)))

                tgt_norm = normalize_text(transcript)
                whisper_str = ''.join([item[0] for item in char_timeline])
                if tgt_norm and whisper_str and char_timeline:
                    matcher = difflib.SequenceMatcher(None, tgt_norm, whisper_str)
                    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
                    if blocks and sum(b.size for b in blocks) >= max(2, len(tgt_norm) * 0.4):
                        first_b = blocks[0]
                        last_b = blocks[-1]
                        t_first = char_timeline[first_b.b][1]
                        t_last = char_timeline[min(len(char_timeline) - 1, last_b.b + last_b.size - 1)][2]
                        return t_first, t_last, prev_end, next_start

            first_words = matched[0].get("words", [])
            last_words = matched[-1].get("words", [])
            t_first = first_words[0]["start"] if first_words else matched[0]["start"]
            t_last = last_words[-1]["end"] if last_words else matched[-1]["end"]
            return t_first, t_last, prev_end, next_start

    raw_in = clip_data.get("source_in", 0)
    raw_out = clip_data.get("source_out", total_dur)
    transcript = clip_data.get("transcript", "")

    # 2. 透過時間窗附近篩選
    cand_segs = [s for s in whisper_units if not (s["end"] < raw_in - 2.5 or s["start"] > raw_out + 2.5)]
    if not cand_segs:
        return raw_in, raw_out, None, None

    prev_end, next_start = _neighbor_bounds(whisper_units, cand_segs[0], cand_segs[-1])

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
            return t_first, t_last, prev_end, next_start

    # 若匹配落空，返回重疊 segment 範圍
    return cand_segs[0]["start"], cand_segs[-1]["end"], prev_end, next_start

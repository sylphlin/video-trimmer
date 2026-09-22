"""Whisper 轉錄、語意句子合併、時間戳工具與文字對齊。"""

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


def _compute_sentence_speaker(sentence):
    """計算句子內多數說話者與合法主講人狀態"""
    words = sentence.get('words', [])
    if words:
        target_cnt = sum(1 for w in words if w.get('is_target_speaker', True))
        sentence['is_target_speaker'] = (target_cnt >= len(words) / 2.0)
        spk_counts = {}
        for w in words:
            spk = w.get('speaker_id')
            if spk:
                spk_counts[spk] = spk_counts.get(spk, 0) + 1
        if spk_counts:
            sentence['speaker_id'] = max(spk_counts.items(), key=lambda x: x[1])[0]


def is_earlier_sentence_ng_retake(prev_text: str, next_text: str) -> bool:
    """
    判定前一個語意單元 (prev_text) 是否為後一個語意單元 (next_text) 的 NG 重講前綴或半截廢話。
    同時容忍句尾最後 1~2 個字因吃螺絲產生的同音錯字（例如「細微摔」vs「細微衰減」）。
    """
    p_norm = normalize_text(prev_text)
    n_norm = normalize_text(next_text)
    if len(p_norm) < 3 or len(n_norm) < 3:
        return False

    # 1. 開頭前綴完全相同（>= 3 字），且後句長度不短於前句的 75%（代表重新完整重講）
    if p_norm[:3] == n_norm[:3] and len(n_norm) >= int(len(p_norm) * 0.75):
        return True

    # 2. 前句扣除最後 1 個可能吃螺絲的錯字後（長度 >= 4），完整出現在後句中
    if len(p_norm) >= 4 and p_norm[:-1] in n_norm:
        return True

    # 3. 前句與後句開頭區段具有高序列相似度（>= 0.72）
    if len(p_norm) >= 5:
        head_window = n_norm[: len(p_norm) + 4]
        if difflib.SequenceMatcher(None, p_norm, head_window).ratio() >= 0.72:
            return True

    return False


def _split_sentence_on_paused_restarts(sentence_data: dict, min_pause_sec: float = 0.18) -> list[dict]:
    """
    僅在單字與單字之間存在「真實物理停頓 (gap >= min_pause_sec)」的前提下，
    若停頓後的詞語重啟了前面剛講過的開頭（>= 3 字），則於該物理停頓處切分為獨立 Sentence。
    嚴禁在無物理停頓的連音處切割，確保 100% 聲學安全。
    """
    words = sentence_data.get("words", [])
    if len(words) < 6:
        return [sentence_data]

    split_word_indices = []
    seg_start_idx = 0

    for j in range(1, len(words)):
        gap = float(words[j].get("start", 0.0)) - float(words[j - 1].get("end", 0.0))
        if gap < min_pause_sec:
            continue

        prev_text = "".join(w.get("word", "") for w in words[seg_start_idx:j]).strip()
        rem_text = "".join(w.get("word", "") for w in words[j : min(len(words), j + 18)]).strip()
        prev_norm = normalize_text(prev_text)
        rem_norm = normalize_text(rem_text)

        if len(prev_norm) >= 3 and len(rem_norm) >= 3:
            # 檢查停頓後的開頭 3 字是否重現了前一段的開頭或子句前綴
            if rem_norm[:3] in prev_norm:
                split_word_indices.append(j)
                seg_start_idx = j

    if not split_word_indices:
        return [sentence_data]

    sub_sentences = []
    boundaries = [0] + split_word_indices + [len(words)]
    for idx in range(len(boundaries) - 1):
        w_slice = words[boundaries[idx] : boundaries[idx + 1]]
        if not w_slice:
            continue
        sub_s = {
            "id": 0,
            "start": w_slice[0]["start"],
            "end": w_slice[-1]["end"],
            "text": "".join(w.get("word", "") for w in w_slice).strip(),
            "words": list(w_slice),
            "orig_segment_ids": list(sentence_data.get("orig_segment_ids", [])),
            "speaker_id": sentence_data.get("speaker_id", "SPEAKER_00"),
            "is_target_speaker": sentence_data.get("is_target_speaker", True),
        }
        _compute_sentence_speaker(sub_s)
        sub_sentences.append(sub_s)

    return sub_sentences


def merge_whisper_segments_to_sentences(segments, max_gap=0.45, max_sentence_dur=14.0):
    """
    將 Whisper 零碎的聲學 Segments 依據自然換氣停頓、聲紋主講人狀態與語法標點合併為完整語意句子。
    若相鄰片段呈現重講/重複前綴（Last Take Wins 候選），強制不黏合以保持獨立 Sentence ID。
    """
    if not segments:
        return []

    CLOSURE_PUNCT = ('。', '！', '？', '!', '?', '……', '...')
    CONJUNCTIONS = ('但是', '而且', '所以', '然而', '如果', '因為', '不過', '雖然', '或是', '或者')

    raw_sentences = []
    curr = None
    last_seg_text = ""

    for s in segments:
        s_text = s['text'].strip()
        s_start = s['start']
        s_end = s['end']
        s_words = s.get('words', [])
        s_is_target = s.get('is_target_speaker', True)
        s_spk = s.get('speaker_id', 'SPEAKER_00')

        if curr is None:
            curr = {
                'id': 1,
                'start': s_start,
                'end': s_end,
                'text': s_text,
                'words': list(s_words),
                'orig_segment_ids': [s['id']],
                'speaker_id': s_spk,
                'is_target_speaker': s_is_target
            }
            last_seg_text = s_text
            _compute_sentence_speaker(curr)
            continue

        gap = s_start - curr['end']
        curr_text = curr['text'].strip()
        dur = s_end - curr['start']

        # 檢查新片段是否為前一段的重講（重複前綴或高相似度）
        is_retake_restart = (
            is_earlier_sentence_ng_retake(curr_text, s_text)
            or is_earlier_sentence_ng_retake(last_seg_text, s_text)
        )

        should_split = False
        # 0. 若偵測到重講重啟，強制切分為獨立 Sentence ID
        if is_retake_restart:
            should_split = True
        # 1. 物理換氣/停頓明顯 (gap >= max_gap)
        elif gap >= max_gap:
            should_split = True
        # 2. 句子長度已很長且有適度微停頓
        elif dur >= max_sentence_dur and gap >= 0.20:
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

        # 5. 聲紋目標有效性跨越防黏合 (場外人員雜音與正片主講人強制斷開)
        if curr.get('is_target_speaker', True) != s_is_target:
            should_split = True

        # 6. 訪談說話者交替輪替 (Speaker Turn-taking: 主持人 vs 來賓強制分句)
        curr_spk = curr.get('speaker_id')
        if s_spk and curr_spk and curr_spk != s_spk:
            should_split = True

        # 7. 絕對長停頓物理斷句 (gap >= 1.0s: 防呆保護，主講人若有自我反省/重來，拆為獨立 Sentence)
        if gap >= 1.0:
            should_split = True

        # 連詞防斷保護 (若下個片段開頭是連詞，且非重講、未發生極長停頓或說話者切換，強制黏合)
        if should_split and not is_retake_restart and gap < 0.90:
            if curr.get('is_target_speaker', True) == s_is_target and curr.get('speaker_id') == s_spk:
                if any(s_text.startswith(c) for c in CONJUNCTIONS):
                    should_split = False

        if should_split:
            _compute_sentence_speaker(curr)
            raw_sentences.append(curr)
            curr = {
                'id': len(raw_sentences) + 1,
                'start': s_start,
                'end': s_end,
                'text': s_text,
                'words': list(s_words),
                'orig_segment_ids': [s['id']],
                'speaker_id': s_spk,
                'is_target_speaker': s_is_target
            }
            _compute_sentence_speaker(curr)
        else:
            curr['end'] = s_end
            curr['text'] = curr['text'] + ' ' + s_text if not curr['text'].endswith((' ', '，', '。')) else curr['text'] + s_text
            curr['words'].extend(s_words)
            curr['orig_segment_ids'].append(s['id'])
            _compute_sentence_speaker(curr)
        last_seg_text = s_text

    if curr is not None:
        _compute_sentence_speaker(curr)
        raw_sentences.append(curr)

    # 對每個合併後的句子檢查是否含有「帶物理停頓 (>=0.18s) 的句內重講」並安全拆開
    final_sentences = []
    for sent in raw_sentences:
        for sub_sent in _split_sentence_on_paused_restarts(sent):
            sub_sent["id"] = len(final_sentences) + 1
            final_sentences.append(sub_sent)

    return final_sentences


def transcribe_video_whisper(
    video_path,
    out_json_path,
    model_name="small"
):
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

    # 執行語意句子合併（每次由 raw_results 即時計算，確保最新斷句與重講拆分規則生效）
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
        "以下是由本地微觀語音模型（Whisper）對本片轉錄、並依據自然換氣停頓與重講邊界拆分之「句子清單」。",
        "每個 Sentence 均為一個候選表達單元，包含物理起訖秒數（start -> end）。",
        "【多模態發言人日誌審查與 Last Take Wins 鐵則】：",
        "請務必結合視訊畫面中主講人的嘴型、眼神方向與肢體動作：",
        "1. 僅挑選由畫面中央「目標主講人面對鏡頭正式發表」（target_host）的有效句子；",
        "2. 任何由場外小幫手/導播喊出的口令（如 Action、報幕代號 CDA82/CTA-S2、CDA84 等，主講人嘴巴閉著或在等待）屬於無效場外音，嚴禁選入 final_edl！",
        "3. 錄影空檔中主講人偏離鏡頭與工作人員之閒聊、自我檢討（如「這段不理想」），屬於 blooper/chatter，亦嚴禁選入 final_edl！",
        "4. 若相鄰或相近的多個 Sentence 講述相同或重複開頭的台詞（講者吃螺絲重錄），請務必在 `sentence_ids` 中徹底剔除前面的 NG 句，只保留最後一次完整流暢的 Sentence ID！",
        "請在輸出 final_edl 時，於 `sentence_ids` 明確列出保留的 Sentence ID 陣列（並填寫 `start_sentence_id` 與 `end_sentence_id`），",
        "並將 `source_in` 與 `source_out` 對齊起訖句子之時間：\n"
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
    In 點鎖定：嚴格跳過所有 is_target_speaker == False 的非目標人聲，直擊主講人真聲開口。

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
                    w_is_target = w.get("is_target_speaker", True)
                    w_dur = (w["end"] - w["start"]) / len(w_norm)
                    for idx, ch in enumerate(w_norm):
                        ch_start = w["start"] + idx * w_dur
                        char_timeline.append((ch, round(ch_start, 3), round(ch_start + w_dur, 3), w_is_target))

                tgt_norm = normalize_text(transcript)
                whisper_str = ''.join([item[0] for item in char_timeline])
                if tgt_norm and whisper_str and char_timeline:
                    matcher = difflib.SequenceMatcher(None, tgt_norm, whisper_str)
                    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
                    if blocks and sum(b.size for b in blocks) >= max(2, len(tgt_norm) * 0.4):
                        # 聲紋鎖定：尋找匹配區間中第一個屬於合法主講人的字元
                        t_first = None
                        for b in blocks:
                            for c_idx in range(b.b, b.b + b.size):
                                if char_timeline[c_idx][3]:  # is_target_speaker == True
                                    t_first = char_timeline[c_idx][1]
                                    break
                            if t_first is not None:
                                break
                        if t_first is None:
                            t_first = char_timeline[blocks[0].b][1]

                        last_b = blocks[-1]
                        t_last = char_timeline[min(len(char_timeline) - 1, last_b.b + last_b.size - 1)][2]
                        return t_first, t_last, prev_end, next_start

            first_words = matched[0].get("words", [])
            last_words = matched[-1].get("words", [])
            # 聲紋鎖定：若首句開頭有場外雜音單詞，跳至第一個合法主講人單詞
            target_words = [w for w in first_words if w.get("is_target_speaker", True)]
            if target_words:
                t_first = target_words[0]["start"]
            elif first_words:
                t_first = first_words[0]["start"]
            else:
                t_first = matched[0]["start"]

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
                w_is_target = w.get("is_target_speaker", True)
                w_dur = (w["end"] - w["start"]) / len(w_norm)
                for idx, ch in enumerate(w_norm):
                    ch_start = w["start"] + idx * w_dur
                    char_timeline.append((ch, round(ch_start, 3), round(ch_start + w_dur, 3), w_is_target))
        else:
            s_norm = normalize_text(s["text"])
            if s_norm:
                s_is_target = s.get("is_target_speaker", True)
                s_dur = (s["end"] - s["start"]) / len(s_norm)
                for idx, ch in enumerate(s_norm):
                    ch_start = s["start"] + idx * s_dur
                    char_timeline.append((ch, round(ch_start, 3), round(ch_start + s_dur, 3), s_is_target))

    tgt_norm = normalize_text(transcript)
    whisper_str = ''.join([item[0] for item in char_timeline])

    if tgt_norm and whisper_str and char_timeline:
        matcher = difflib.SequenceMatcher(None, tgt_norm, whisper_str)
        blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
        if blocks:
            # 聲紋鎖定：尋找第一個屬於合法主講人的字元
            t_first = None
            for b in blocks:
                for c_idx in range(b.b, b.b + b.size):
                    if char_timeline[c_idx][3]:  # is_target_speaker == True
                        t_first = char_timeline[c_idx][1]
                        break
                if t_first is not None:
                    break
            if t_first is None:
                t_first = char_timeline[blocks[0].b][1]

            last_b = blocks[-1]
            t_last = char_timeline[min(len(char_timeline) - 1, last_b.b + last_b.size - 1)][2]
            return t_first, t_last, prev_end, next_start

    # 若匹配落空，返回重疊 segment 範圍
    return cand_segs[0]["start"], cand_segs[-1]["end"], prev_end, next_start


def filter_ng_retake_sentences(sentences: list[dict], max_lookahead: int = 3, max_time_span: float = 25.0) -> list[dict]:
    """
    在候選句子序列中執行確定性的「Last Take Wins」過濾：
    若句子 S_i 與後續緊鄰的句子 S_j (j > i) 構成重講關係（S_i 為 NG 前綴或吃螺絲半截話），
    自動於自然句子邊界處剔除 S_i，僅保留最後一次完整句子。
    """
    if len(sentences) <= 1:
        return list(sentences)

    kept = []
    n = len(sentences)
    for i, s_curr in enumerate(sentences):
        is_ng = False
        limit = min(n, i + 1 + max_lookahead)
        for j in range(i + 1, limit):
            s_later = sentences[j]
            if float(s_later.get("start", 0.0)) - float(s_curr.get("end", 0.0)) > max_time_span:
                break
            if is_earlier_sentence_ng_retake(s_curr.get("text", ""), s_later.get("text", "")):
                logger.info(
                    "    [Last-Take-Wins 過濾] 剔除 NG 重複句 (Sentence %s: '%s') -> 保留正式句 (Sentence %s: '%s')",
                    s_curr.get("id"),
                    s_curr.get("text", ""),
                    s_later.get("id"),
                    s_later.get("text", ""),
                )
                is_ng = True
                break
        if not is_ng:
            kept.append(s_curr)
    return kept


def _split_sentence_words_by_internal_gap(sentence: dict, max_word_gap: float = 0.45) -> list[dict]:
    """
    若單一 Sentence 內部的單字與單字之間存在明顯看稿發呆停頓 (gap >= max_word_gap)，
    於該物理靜音區拆分為多個緊湊發音塊 (speech chunks)，使每個停頓都能被聲學引擎收緊。
    """
    words = [w for w in sentence.get("words", []) if w.get("is_target_speaker", True)]
    if not words:
        words = list(sentence.get("words", []))
    if not words:
        return [{
            "sentence_id": sentence.get("id", 0),
            "t_first": float(sentence.get("start", 0.0)),
            "t_last": float(sentence.get("end", 0.0)),
            "text": sentence.get("text", "").strip(),
        }]

    chunks = []
    cur_words = [words[0]]
    for w in words[1:]:
        gap = float(w.get("start", 0.0)) - float(cur_words[-1].get("end", 0.0))
        if gap >= max_word_gap:
            chunks.append({
                "sentence_id": sentence.get("id", 0),
                "t_first": float(cur_words[0]["start"]),
                "t_last": float(cur_words[-1]["end"]),
                "text": "".join(x.get("word", "") for x in cur_words).strip(),
            })
            cur_words = [w]
        else:
            cur_words.append(w)

    if cur_words:
        chunks.append({
            "sentence_id": sentence.get("id", 0),
            "t_first": float(cur_words[0]["start"]),
            "t_last": float(cur_words[-1]["end"]),
            "text": "".join(x.get("word", "") for x in cur_words).strip(),
        })

    return filter_ng_retake_sentences(chunks, max_lookahead=2, max_time_span=15.0)


def resolve_clip_sub_units(
    whisper_units: list[dict],
    clip_data: dict,
    total_dur: float,
    max_internal_gap: float = 0.40,
    max_word_gap: float = 0.45,
) -> list[dict]:
    """
    將單一 Clip 決策解析為一或多個「無句間長空白、無中間夾雜 NG 句」的緊湊語音子片段 (Sub-units)。
    1. 優先讀取 `sentence_ids` 陣列（明確排除中間跳過的 NG 句）；若無則讀取 `start_sentence_id`..`end_sentence_id`。
    2. 自動執行 `filter_ng_retake_sentences` 剔除區間內重講前的 NG 句。
    3. 只要相鄰句子之間跳過了 NG 句、或存在 >= max_internal_gap (0.40s) 的看稿停頓、
       或句內單字間存在 >= max_word_gap (0.45s) 的空白停頓，即於該物理靜音區拆開為獨立子片段，
       交由後續 `refine_speech_bounds_locked` 逐一收緊頭尾空白。
    """
    if not whisper_units:
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(whisper_units, clip_data, total_dur)
        return [{
            "t_first": t_first,
            "t_last": t_last,
            "prev_sentence_end": prev_end,
            "next_sentence_start": next_start,
            "transcript": clip_data.get("transcript", "") or clip_data.get("content", ""),
        }]

    matched = []
    raw_sent_ids = clip_data.get("sentence_ids")
    if isinstance(raw_sent_ids, list) and raw_sent_ids:
        valid_ids = {int(x) for x in raw_sent_ids if isinstance(x, (int, str)) and str(x).isdigit()}
        matched = [s for s in whisper_units if s.get("id") in valid_ids]

    if not matched:
        start_id = clip_data.get("start_sentence_id")
        if start_id is None:
            start_id = clip_data.get("start_segment_id")
        end_id = clip_data.get("end_sentence_id")
        if end_id is None:
            end_id = clip_data.get("end_segment_id")
        if start_id is not None and end_id is not None:
            matched = [s for s in whisper_units if start_id <= s.get("id", 0) <= end_id]

    if not matched:
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(whisper_units, clip_data, total_dur)
        return [{
            "t_first": t_first,
            "t_last": t_last,
            "prev_sentence_end": prev_end,
            "next_sentence_start": next_start,
            "transcript": clip_data.get("transcript", "") or clip_data.get("content", ""),
        }]

    # 過濾非目標主講人句子（如場外人員喊口令）
    target_matched = [s for s in matched if s.get("is_target_speaker", True)]
    if target_matched:
        matched = target_matched

    # 執行句子級 Last Take Wins 過濾，剔除區間內重講前的 NG 句
    matched = filter_ng_retake_sentences(matched)

    # 拆解為細粒度發音塊（消除句內 >= 0.45s 的空白停頓）
    fine_chunks = []
    for s in matched:
        fine_chunks.extend(_split_sentence_words_by_internal_gap(s, max_word_gap=max_word_gap))

    fine_chunks = filter_ng_retake_sentences(fine_chunks, max_lookahead=2, max_time_span=20.0)
    if not fine_chunks:
        t_first, t_last, prev_end, next_start = align_clip_with_whisper(whisper_units, clip_data, total_dur)
        return [{
            "t_first": t_first,
            "t_last": t_last,
            "prev_sentence_end": prev_end,
            "next_sentence_start": next_start,
            "transcript": clip_data.get("transcript", "") or clip_data.get("content", ""),
        }]

    # 將相鄰且間隔 < max_internal_gap (0.40s)、且未跳過句子的發音塊合併；若間隔 >= 0.40s 或跳過 NG 句則拆開收緊
    grouped_units = []
    cur_group = {
        "sentence_ids": [fine_chunks[0]["sentence_id"]],
        "t_first": fine_chunks[0]["t_first"],
        "t_last": fine_chunks[0]["t_last"],
        "texts": [fine_chunks[0]["text"]],
    }

    for ch in fine_chunks[1:]:
        gap = ch["t_first"] - cur_group["t_last"]
        prev_sid = cur_group["sentence_ids"][-1]
        cur_sid = ch["sentence_id"]
        skipped_sentence = (cur_sid > prev_sid + 1)

        if gap >= max_internal_gap or skipped_sentence:
            grouped_units.append(cur_group)
            cur_group = {
                "sentence_ids": [cur_sid],
                "t_first": ch["t_first"],
                "t_last": ch["t_last"],
                "texts": [ch["text"]],
            }
        else:
            if cur_sid not in cur_group["sentence_ids"]:
                cur_group["sentence_ids"].append(cur_sid)
            cur_group["t_last"] = ch["t_last"]
            cur_group["texts"].append(ch["text"])

    grouped_units.append(cur_group)

    # 為每個子片段計算相鄰聲學保護邊界 (prev_sentence_end, next_sentence_start)
    all_bounds = [(float(u.get("start", 0.0)), float(u.get("end", 0.0))) for u in whisper_units]
    sub_units = []
    for idx, g in enumerate(grouped_units):
        t_f = g["t_first"]
        t_l = g["t_last"]

        # 尋找緊鄰於 t_f 之前的聲音結束點
        prev_ends = [b_end for (_, b_end) in all_bounds if b_end <= t_f + 0.02]
        if idx > 0:
            prev_ends.append(grouped_units[idx - 1]["t_last"])
        prev_end = max(prev_ends) if prev_ends else None

        # 尋找緊鄰於 t_l 之後的聲音開始點
        next_starts = [b_start for (b_start, _) in all_bounds if b_start >= t_l - 0.02]
        if idx + 1 < len(grouped_units):
            next_starts.append(grouped_units[idx + 1]["t_first"])
        next_start = min(next_starts) if next_starts else None

        sub_text = " ".join(t for t in g["texts"] if t).strip()
        sub_units.append({
            "t_first": t_f,
            "t_last": t_l,
            "prev_sentence_end": prev_end,
            "next_sentence_start": next_start,
            "transcript": sub_text or (clip_data.get("transcript", "") or clip_data.get("content", "")),
        })

    return sub_units


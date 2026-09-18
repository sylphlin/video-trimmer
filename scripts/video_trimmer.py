#!/usr/bin/env python3
"""
video-trimmer (AI-Powered Smart Video Trimmer)
---------------------------------------------
An intelligent, end-to-end video rough-cut & trimming engine powered by:
- Whisper word-level acoustic ground-truth (Apple Silicon mlx-whisper & faster-whisper)
- Gemini 3.8 Flash multimodal video understanding & take selection
- Acoustic Onset Snapping (Smart Gap Shortening: eliminating dead air & pre-speech pauses)
- Adaptive noise floor tracking & equal-power audio micro-crossfades
- Text-based locked timestamp precision
- Export to Premiere Pro (XML), DaVinci Resolve (XML), Final Cut Pro (FCPXML), and direct MP4.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import soundfile as sf
    import numpy as np  # noqa: F401  (used transitively by acoustic.py; kept for early dependency check)
    from google import genai
    from google.genai import types
except ImportError as e:
    # 此處早於 main() 的 logging 設定即需中止，且是模組載入期的硬性依賴檢查
    # （而非業務邏輯例外），故沿用 sys.exit()；仍改用 logging 而非 print()。
    logging.basicConfig(level=logging.ERROR, format="%(message)s", stream=sys.stderr)
    logging.error("缺少必要套件: %s", e)
    logging.error("請先執行: pip install -r requirements.txt")
    sys.exit(1)

import os
from .acoustic import refine_speech_bounds_locked
from .constants import ALLOWED_INPUT_EXTENSIONS
from .exceptions import InvalidInputError, VideoTrimmerError
from .exporters import generate_edl_csv, generate_fcp7_xml, generate_fcpxml
from .gcs_utils import delete_gcs_blob, is_gdrive_source, download_gdrive_file_with_cache
from .gemini_client import (
    build_prompt,
    get_gemini_client,
    load_env_file,
    run_gemini_inference,
    stage_video_to_gcs,
)
from .render import probe_video, render_cut_video
from .transcribe import align_clip_with_whisper, transcribe_video_whisper

logger = logging.getLogger(__name__)


def _setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)


def _validate_input_extension(video_path):
    if video_path.suffix.lower() not in ALLOWED_INPUT_EXTENSIONS:
        allowed = "、".join(ALLOWED_INPUT_EXTENSIONS)
        raise InvalidInputError(f"不支援的影片格式: {video_path.suffix}（允許的副檔名: {allowed}）")


def _append_usage_log(out_dir, video_path, model, usage, mode, duration):
    """獨立於 logging 系統之外，永遠記錄每次 Gemini 呼叫的 token 用量，供後續人工彙總成本。"""
    usage_log_path = out_dir / "usage_log.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "video": video_path.name,
        "model": model,
        "mode": mode,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "candidates_tokens": usage.get("candidates_tokens", 0),
        "thoughts_tokens": usage.get("thoughts_tokens", 0),
        "tool_use_tokens": usage.get("tool_use_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "duration_seconds": round(duration, 2),
    }
    with open(usage_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _run(args):
    if is_gdrive_source(args.input):
        out_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            video_path = download_gdrive_file_with_cache(
                args.input,
                target_dir=out_dir / "gdrive_inputs",
                project_id=args.project,
            )
        except Exception as e:
            raise InvalidInputError(f"從 Google Drive 讀取影片失敗: {e}") from e
    else:
        video_path = Path(args.input).resolve()
        if not video_path.exists():
            raise InvalidInputError(f"找不到影片檔案 {video_path}")
        out_dir = Path(args.output_dir).resolve() if args.output_dir else video_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

    _validate_input_extension(video_path)
    base_name = video_path.stem

    # 決定輸出檔案標籤
    tag = args.suffix if args.suffix else ("agentic" if args.agentic else "static")

    logger.info("==> 1. 檢測影片資訊: %s", video_path.name)
    total_dur, width, height, fps = probe_video(video_path)
    logger.info("    時長: %.1f 秒 (~%.1f 分鐘) | 解析度: %dx%d | 幀率: %.3f fps",
                total_dur, total_dur / 60, width, height, fps)

    # 第一階段：Whisper 微觀聲學時間戳轉錄 (Word-level Ground Truth) 與語意句子合併
    whisper_segs = []
    whisper_sentences = []

    if not args.skip_whisper:
        whisper_json = out_dir / f"{base_name}_whisper_raw.json"
        logger.info("==> 2. 執行 Whisper 微觀字級聲學時間戳轉錄與語意句子合併...")
        whisper_segs, whisper_sentences = transcribe_video_whisper(
            video_path,
            whisper_json,
            model_name="small",
        )

    whisper_units = whisper_sentences if whisper_sentences else whisper_segs

    # 第二階段：Gemini 宏觀多模態視訊理解與選鏡決策
    inference_metrics = {}
    if args.cached_json:
        cached_file = Path(args.cached_json).resolve()
        if not cached_file.exists():
            raise InvalidInputError(f"找不到快取 JSON 檔案: {cached_file}")
        logger.info("==> 3. 跳過 Gemini 模型分析，直接載入快取初剪決策: %s", cached_file.name)
        with open(cached_file, "r", encoding="utf-8") as f:
            model_edl = json.load(f)
    else:
        load_env_file()
        client = get_gemini_client(project_id=args.project, location=args.region)

        prompt_candidates = [
            Path(__file__).parent / "prompts" / "video_cut_prompt.md",
            Path(__file__).parent.parent / "prompts" / "video_cut_prompt.md",
            Path.cwd() / "prompts" / "video_cut_prompt.md",
            Path(__file__).parent / "video-trimmer" / "prompts" / "video_cut_prompt.md",
            Path(__file__).parent / "pansci-ai-rough-cut" / "prompts" / "video_cut_prompt.md",
        ]
        script_path = Path(args.script).resolve() if args.script else None
        prompt = build_prompt(prompt_candidates, script_path=script_path, whisper_units=whisper_units)

        resolved_bucket = (
            args.bucket
            or os.environ.get("VIDEO_TRIMMER_BUCKET")
            or os.environ.get("VIDEO_STORAGE_BUCKET")
            or os.environ.get("MEETING_STORAGE_BUCKET")
            or os.environ.get("GCS_BUCKET")
            or ""
        ).removeprefix("gs://") or None

        logger.info("==> 3. 準備視訊至 Cloud Storage 暫存...")
        gcs_uri, mime_type, is_ephemeral = stage_video_to_gcs(video_path, resolved_bucket)

        mode_str = "🤖 Agentic Video Understanding (動態影格探索模式)" if args.agentic else "📺 Static Multimodal (靜態抽幀)"
        logger.info("==> 4. 調用 %s [模式: %s] 結合 Whisper 劇本進行文字剪輯與選鏡決策...", args.model, mode_str)

        try:
            raw_json, usage, duration = run_gemini_inference(client, types, args.model, gcs_uri, mime_type, prompt, args.agentic)
        finally:
            if is_ephemeral and not args.keep_gcs_upload:
                delete_gcs_blob(gcs_uri)

        logger.info("    模型推論完成 (耗時 %.1f 秒)！", duration)
        logger.info("    📊 Token 統計: 輸入=%s | 輸出=%s | 思維=%s | 工具探索=%s | 總計=%s",
                    f"{usage['prompt_tokens']:,}", f"{usage['candidates_tokens']:,}",
                    f"{usage['thoughts_tokens']:,}", f"{usage['tool_use_tokens']:,}", f"{usage['total_tokens']:,}")

        mode = "agentic" if args.agentic else "static"
        inference_metrics = {
            "mode": mode,
            "duration_seconds": round(duration, 2),
            "prompt_tokens": usage["prompt_tokens"],
            "candidates_tokens": usage["candidates_tokens"],
            "thoughts_tokens": usage["thoughts_tokens"],
            "tool_use_tokens": usage["tool_use_tokens"],
            "total_tokens": usage["total_tokens"]
        }
        _append_usage_log(out_dir, video_path, args.model, usage, mode, duration)

        raw_json = raw_json.strip()
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0].strip()
        elif not raw_json.startswith("{") and "{" in raw_json:
            start_idx = raw_json.find("{")
            end_idx = raw_json.rfind("}")
            if start_idx != -1 and end_idx != -1:
                raw_json = raw_json[start_idx:end_idx + 1].strip()

        try:
            model_edl = json.loads(raw_json)
        except Exception as e:
            raise VideoTrimmerError(f"解析 Gemini JSON 失敗: {e}\n原始回傳內容:\n{raw_json}") from e

    # 第三階段：比照字幕邏輯——文字剪輯時間鎖定與 Whisper 句子邊界防護
    logger.info("==> 5. 執行文字剪輯時間鎖定 (Text-Based Locked Timestamps)...")
    temp_wav = out_dir / f"temp_{base_name}.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(temp_wav)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audio, sr = sf.read(str(temp_wav))

    refined_edl = []
    for c in model_edl.get("final_edl", []):
        t_first, t_last, prev_sentence_end, next_sentence_start = align_clip_with_whisper(whisper_units, c, total_dur)
        transcript = c.get("transcript", "") or c.get("content", "")

        tight_in, tight_out, clip_cps, in_m, out_m = refine_speech_bounds_locked(
            audio, sr, t_first, t_last, total_dur, transcript=transcript, pacing=args.pacing,
            prev_sentence_end=prev_sentence_end, next_sentence_start=next_sentence_start
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
        logger.info("    Clip %2d: In=%6.2fs, Out=%6.2fs (%5.2fs) | 首字: %.2fs, 尾字: %.2fs | %s",
                    c["clip_id"], tight_in, tight_out, dur, t_first, t_last, c["topic"])

    temp_wav.unlink(missing_ok=True)
    total_out_dur = round(sum(c["duration"] for c in refined_edl), 2)
    avg_cps = round(sum(c["cps"] for c in refined_edl) / max(1, len(refined_edl)), 2)
    logger.info("    初剪片段數: %d | 成片預計長度: %.1f 秒 (~%.2f 分鐘) | 全片平均語速: %.2f 字/秒",
                len(refined_edl), total_out_dur, total_out_dur / 60, avg_cps)

    # 輸出資料
    json_path = out_dir / f"{base_name}_{tag}_edl.json"
    json_path.write_text(json.dumps({
        "project_title": f"{base_name} AI Video Trimmer ({tag})",
        "pacing_style": "text_based_whisper_grounded",
        "inference_metrics": inference_metrics,
        "average_cps": avg_cps,
        "total_duration": total_out_dur,
        "final_edl": refined_edl
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("==> 6. 輸出結構化資料: %s", json_path.name)

    csv_path = out_dir / f"{base_name}_{tag}_edl.csv"
    generate_edl_csv(refined_edl, csv_path)
    logger.info("==> 7. 輸出表格清單: %s", csv_path.name)

    xml_path = out_dir / f"{base_name}_{tag}_edl.xml"
    generate_fcp7_xml(refined_edl, video_path, total_dur, xml_path, width, height, fps)
    logger.info("==> 8. 輸出通用剪輯工程檔: %s", xml_path.name)

    fcpxml_path = out_dir / f"{base_name}_{tag}_edl.fcpxml"
    generate_fcpxml(refined_edl, video_path, total_dur, total_out_dur, fcpxml_path, fps)
    logger.info("==> 9. 輸出 Final Cut Pro X 工程檔: %s", fcpxml_path.name)

    out_mp4 = out_dir / f"{base_name}_{tag}_trimmed.mp4"
    logger.info("==> 10. FFmpeg 渲染成片: %s ...", out_mp4.name)
    render_cut_video(refined_edl, video_path, out_mp4, crf=args.crf)
    logger.info("==> [完成] 最終成片已產出！檔案大小: %.1f MB", out_mp4.stat().st_size / (1024 * 1024))


def main():
    load_env_file()
    default_model = os.environ.get("MODEL_NAME") or os.environ.get("TRIMMER_MODEL") or "gemini-3.8-flash"

    parser = argparse.ArgumentParser(description="video-trimmer: AI-Powered Smart Video Trimmer (Whisper Ground-Truth + Vertex AI Gemini)")
    parser.add_argument("--input", "-i", required=True, help="輸入影片檔案路徑 (MP4/MOV) 或 Google Drive 分享連結 (https://drive.google.com/... / gdrive://...)")
    parser.add_argument("--output-dir", "-o", default=None, help="輸出資料夾 (預設為影片所在目錄)")
    parser.add_argument("--model", "-m", default=default_model, help=f"使用的 Gemini 模型名稱 (預設: {default_model})")
    parser.add_argument("--project", default=None, help="Google Cloud 專案 ID (預設讀取 GOOGLE_CLOUD_PROJECT/GCP_PROJECT 或 ADC)")
    parser.add_argument("--region", default=None, help="Vertex AI 區域/位置 (預設讀取 GOOGLE_CLOUD_LOCATION/GCP_REGION 或 'global')")
    parser.add_argument("--bucket", default=None, help="用於暫存視訊的 GCS Bucket (預設讀取 VIDEO_TRIMMER_BUCKET)")
    parser.add_argument("--keep-gcs-upload", action="store_true", help="保留上傳至 GCS 的暫存視訊，不於推論後自動清理")
    parser.add_argument("--crf", type=int, default=18, help="FFmpeg H.264 畫質參數 (預設 18)")
    parser.add_argument("--pacing", "-p", choices=["auto", "dynamic", "compact", "breathing"], default="auto",
                        help="剪輯節奏風格: 'auto'/'dynamic' (文字剪輯微氣息鎖定, 推薦預設)")
    parser.add_argument("--agentic", action="store_true", help="啟用 Gemini Agentic Video Understanding 動態探索模式")
    parser.add_argument("--suffix", default=None, help="自訂輸出檔案名稱標籤後綴 (預設為 agentic 或 static)")
    parser.add_argument("--script", "-s", default=None, help="可選的分鏡講稿或文本檔案路徑 (TXT/MD)")
    parser.add_argument("--cached-json", default=None,
                        help="指定既有之初剪決策 JSON 檔案路徑，跳過 Gemini 上傳與雲端分析")
    parser.add_argument("--skip-whisper", action="store_true", help="跳過本地 Whisper 轉錄，僅使用純能量檢測")
    parser.add_argument("--verbose", action="store_true", help="輸出詳細除錯訊息 (DEBUG level)")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    try:
        _run(args)
    except VideoTrimmerError as e:
        logger.error("錯誤: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Video Trimmer (`video-trimmer`)

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

[English (en)](README.md) | [繁體中文 (zh-TW)](README.zh-TW.md) | [简体中文 (zh-CN)](README.zh-CN.md) | [日本語 (ja)](README.ja.md) | [한국어 (ko)](README.ko.md)

---

## 專案總覽 (Overview)

**Video Trimmer** 是專為單機位口播、教學影片與演講錄影設計的 AI 自動粗剪與去蕪存菁引擎。系統結合 **Google Vertex AI Gemini 3.8 Flash** 原生多模態影片理解、**Whisper 毫秒級逐字聲學時間戳**（Apple Silicon Metal `mlx-whisper` 加速）以及 **聲學起音鎖定（Acoustic Onset Snapping）**。每次執行會自動剔除 NG 重錄、吃螺絲與無效停頓，並匯出多格式專業剪輯時間軸（`.xml`, `.fcpxml`, `.edl`, `.csv`）與完成粗剪的 MP4 影片。

---

## 核心技術特點

1. **多模態原生影片理解 (`gemini-3.8-flash`)**：
   - 透過 Cloud Storage（`gs://${BUCKET}/raw/`）將原始影片送入 Vertex AI 分析。
   - 同步評估講者眼神、表情、口誤與重錄段落。
2. **Last-Take-Wins（保留最後成功重錄）**：
   - 自動識別同一段台詞的多次試講或重錄，精準保留最後一次完整成功的版本。
3. **多模態主講者分離**：
   - 結合畫面嘴型同步與麥克風收音距離，區分鏡頭前主講者與場外工作人員口令（如 `Action`、`Cut`）。
4. **聲學起音鎖定 (`tighten_clip_to_speech`)**：
   - 掃描本地音訊波形，將剪輯入點鎖定於聲帶發聲前 80 ms，消除開口前的冗長空白且不截斷字首音節。
5. **自適應底噪與字尾保留**：
   - 依據講者語速（CPS）與環境底噪動態計算緩衝邊界，完整保留鼻音結尾與自然思考停頓。
6. **15 ms 等功率音訊微淡入淡出**：
   - 於每個剪輯切點自動注入 15 ms 等功率淡入淡出（`afade=t=in:d=0.015:curve=iqsin` 與 `afade=t=out:d=0.015:curve=oqsin`），消除跳接爆音（Audio Pop）。
7. **拍攝腳本對齊 (`--script`)**：
   - 支援傳入拍攝腳本（`shooting_script.md`），引導模型逐段比對台詞完整性。
8. **多平台 NLE 時間軸匯出**：
   - 支援匯出 **Final Cut Pro 7 XML**（`.xml`，適用於 Adobe Premiere Pro 與 DaVinci Resolve）、**Apple Final Cut Pro FCPXML**（`.fcpxml`）、**CMX 3600 EDL**（`.edl`）與 **CSV** 剪輯表。

---

## 安裝與 Google Cloud 環境設定

### 1. 安裝 FFmpeg

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg
```

### 2. 安裝為 Antigravity Plugin 或本地 CLI

```bash
# 全域 Antigravity Plugin（建議）
git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/plugins/video-trimmer

# 安裝 Python 相依套件與 Apple Silicon Metal 加速
pip install -r requirements.txt
pip install mlx-whisper
```

### 3. 一鍵配置 Google Cloud 資源 (`./setup.sh`)

```bash
gcloud auth application-default login
chmod +x setup.sh
./setup.sh --project YOUR_GCP_PROJECT_ID --region us-central1
```

---

## 命令列使用說明 (CLI Usage)

```bash
# 基本影片粗剪
python3 video_trimmer.py -i "raw_footage.mp4"

# 搭配拍攝腳本對齊
python3 video_trimmer.py -i "raw_footage.mp4" --script "shooting_script.md"

# 啟用 Agentic 影片理解模式
python3 video_trimmer.py -i "raw_footage.mp4" --agentic

# 採用緊湊節奏模式（適用於快節奏教學影片）
python3 video_trimmer.py -i "sample_take.mp4" --pacing compact --suffix "fast"

# 直接讀取 Google Drive 分享連結進行粗剪
python3 video_trimmer.py -i "https://drive.google.com/file/d/FILE_ID/view?usp=sharing" --agentic -o output/
```

---

## 匯出檔案與 NLE 匯入步驟

針對輸入檔案 `raw_footage.mp4`，系統會產出：
1. **`raw_footage_<tag>_trimmed.mp4`**：含 15 ms 音訊微淡入淡出的粗剪影片。
2. **`raw_footage_<tag>_edl.xml`**：供 **DaVinci Resolve** 與 **Adobe Premiere Pro** 匯入的 FCP7 XML 時間軸。
3. **`raw_footage_<tag>_edl.fcpxml`**：供 **Final Cut Pro** 匯入的 FCPXML 時間軸。
4. **`raw_footage_<tag>_edl.json` / `.csv`**：結構化剪輯決策紀錄與表格。

---

## Google Drive 分享連結與 GCS 雙層生命週期規則

| GCS 路徑前綴 (`matchesPrefix`) | 儲存內容 | 保留天數 (`age`) | 規則說明 |
| :--- | :--- | :--- | :--- |
| **`raw/`** | 暫存原始影片 (`raw/<filename>.mp4`) | **2 天 (`age: 2`)** | 推論完成後於 `finally` 區塊立即刪除，並以 2 天自動刪除規則作為安全備援。 |
| **`output/`**、**`deliverables/`**、**`trimmed/`** | 粗剪影片、XML/FCPXML 時間軸與 EDL 報告 | **15 天 (`age: 15`)** | 保留 15 天供團隊審閱與下載，期滿自動清理。 |

---

## 單元測試 (Unit Testing)

```bash
python3 -m unittest discover -s tests -v
```

---

## 授權條款 (License)

本專案採用 [MIT License](LICENSE) 授權。

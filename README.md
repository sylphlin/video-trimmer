# 泛科學 AI 自動初剪系統 (PanSci AI Auto Rough Cut)

> 基於 **Gemini 3.8 Flash 多模態長視訊理解** 與 **緊湊式減法剪輯理論**，為單機口播、知識科普演講打造的端到端自動初剪開源工具。
> 支援「重錄取最後一次（Last Take Wins）」、「眼神就緒防眨眼」、「消除死寂停頓」，一鍵輸出主流剪輯軟體工程檔（Premiere / DaVinci / Final Cut Pro）與成片。

---

## 目錄結構 (Directory Structure)

本專案為完全獨立的工具包，可整體拷貝或作為獨立 Git Repository 搬遷至任何環境使用：

```text
pansci-ai-rough-cut/
├── README.md                 # 專案完整說明與指引
├── auto_rough_cut.py         # 核心 CLI 一鍵自動初剪程式
├── requirements.txt          # Python 依賴清單
├── .gitignore                # Git 忽略設定
├── prompts/
│   └── video_cut_prompt.md   # 核心端到端多模態初剪提示詞 (Markdown 格式)
└── examples/                 # 初剪結果範例 (EDL / XML / FCPXML / JSON / CSV)
    ├── take1_edl.json / .xml / .fcpxml / .csv
    └── take2_edl.json / .xml / .fcpxml / .csv
```

---

## 核心亮點與特色

1. **端到端多模態直接理解 (Native Multimodal Video Understanding)**：
   * 不依賴傳統語音轉文字（ASR）＋時間碼對照的脆弱流程。
   * 將原始視訊直接上傳至 Gemini Files API，由大模型同步理解「畫面中講者的眼神／表情／肢體動作」與「語音情緒」。
2. **語意重複「取最後一次」（Last Take Wins）**：
   * 講者吃螺絲、忘詞卡詞、同主題多次錄製時，系統自動識別重複情境，永遠只保留最後一次完整流暢的成功 Take。
3. **緊湊式減法剪輯（Compact Subtraction Pacing）**：
   * 採用現代 YouTube / 專業口播新聞節奏：一句話尾音結束後 **+0.08 秒瞬間切出**，下一段開口前僅留 **0.09 秒微氣息緩衝**。
   * 相鄰片段銜接停頓嚴格控制在 **0.15 ~ 0.20 秒**，徹底杜絕片段之間的發呆定格感。
4. **視覺就緒與防閉眼眨眼（Visual Readiness & Blink Avoidance）**：
   * 切入第一格畫面嚴格檢視眼神正視鏡頭、表情進入狀態，避開眨眼與嘴型半開。
5. **提示詞全面採用 Markdown (`prompts/video_cut_prompt.md`)**：
   * 方便人類閱讀、排版維護與版本控管，並利用 Markdown 結構引導 Gemini 輸出標準 JSON。
6. **提示音已完全非必要**：
   * 傳統拍攝使用的「嗶」聲打板在多模態大模型下已完全不需使用，大幅降低現場拍攝門檻。

---

## 快速開始 (Quick Start)

### 1. 安裝系統工具 (FFmpeg)

本工具需使用系統層級的 `ffmpeg` 與 `ffprobe` 進行無損剪輯與波形分析：

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg
```

### 2. 安裝 Python 依賴

```bash
pip install -r requirements.txt
```

### 3. 設定 Gemini API Key

請至 [Google AI Studio](https://aistudio.google.com/) 取得 API Key：

```bash
export GEMINI_API_KEY="你的_GEMINI_API_KEY"

# 或直接寫入 ~/.gemini/.env
echo 'GEMINI_API_KEY="你的_GEMINI_API_KEY"' >> ~/.gemini/.env
```

---

## 使用方式 (Usage)

只需一行指令，即可對任何拍攝原片執行自動初剪：

```bash
python auto_rough_cut.py --input "/path/to/your_video.mp4"
```

### 剪輯節奏風格支援 (`--pacing`)

系統支援「自適應語速動態浮動（Dynamic Floating Pacing，預設推薦）」以及手動指定風格：

* **`auto` / `dynamic`（連續型動態浮動呼吸，預設推薦）**：
  * **原理**：系統自動分析每段台詞的語速 **CPS（每秒字數/音節數）**，透過連續插值公式動態計算最適留白：
    * 遇快口播（CPS ≥ 5.5，如科技快報、短影音）：自動收緊至 In=0.06s / Out=0.08s 俐落切出。
    * 遇慢說書（CPS ≤ 4.2，如工頭堅歷史漫談）：自動放寬至 In=0.35s / Out=0.45s 飽滿呼吸感。
  * **純靜默邊界防護（Silence-Only Boundary）**：呼吸留白只取自主講人開口前的純淨環境底噪，逐幀檢驗能量；若回溯過程遇到任何突發雜訊（拍手打板、相機提示嗶聲、導演喊話『來！』），刀口立即止步於該雜訊之後的靜默區，**100% 杜絕場外音與拍手**！
* **`compact`（緊湊減法模式）**：
  * 適用：泛科學新聞、快節奏科普短影音、高密度口播。
  * 特色：落音即切（+0.08s）、開口微氣息（-0.06s），段落間隔約 **0.15 ~ 0.25 秒**，一氣呵成。
* **`breathing`（固定呼吸空間模式）**：
  * 適用：文化歷史、慢說書、深度人物訪談、紀錄片。
  * 特色：固定保留開口前 0.35s 氣息與眼神定格、句尾留白 0.45s 沉澱餘韻，段落間隔約 **0.70 ~ 0.90 秒**。

### 常用指令範例

```bash
# 1. 自適應動態呼吸模式 (預設靜態抽幀 + Whisper 語意句子合併)
python auto_rough_cut.py --input "take 1_1080p.mp4"

# 2. 🤖 啟用 Agentic Video Understanding 動態探索模式 (推薦長片與雜訊複雜環境)
python auto_rough_cut.py --input "take 1_1080p.mp4" --agentic

# 3. 指定輸出標籤與緊湊減法模式
python auto_rough_cut.py --input "take 1_1080p.mp4" --agentic --suffix "agentic_test" --pacing compact
```

### 完整參數說明

```bash
python auto_rough_cut.py \
  --input "video.mp4" \              # 輸入影片路徑 (必填)
  --output-dir "./output" \          # 輸出資料夾 (預設為影片所在同目錄)
  --agentic \                        # 啟用 Google Interactions API 原生 Agentic 視訊動態探索
  --suffix "my_cut" \                # 自訂輸出檔案標籤 (如 _my_cut_edl.json)
  --pacing auto \                    # 節奏風格: 'auto' (預設動態), 'compact', 'breathing'
  --model "gemini-3.8-flash" \       # 指定模型 (預設 gemini-3.8-flash)
  --script "script.txt" \            # 附帶參考分鏡講稿 (可選)
  --skip-whisper \                   # 跳過 Whisper 聲學對齊 (僅測試時使用)
  --crf 18                           # 渲染畫質參數 (CRF 18 為視覺無損)
```

### 產出檔案

執行後將在輸出目錄自動產生核心檔案：
1. **`<檔名>_<tag>_rough_cut.mp4`**：緊湊連續版成片（音訊邊緣自動套用 15ms 抗爆音微淡化）。
2. **`<檔名>_<tag>_edl.xml`**：通用 FCP 7 XML 剪輯工程檔（供 Premiere Pro、DaVinci Resolve 使用）。
3. **`<檔名>_<tag>_edl.fcpxml`**：FCPXML 格式（供 Final Cut Pro X 使用）。
4. **`<檔名>_<tag>_edl.json`**：包含每段講述口白、視覺就緒審查、Token 統計與精準時間戳的結構化資料。
5. **`<檔名>_<tag>_edl.csv`**：供 Excel / Numbers 檢視的剪輯清單。
6. **`<檔名>_whisper_sentences.json`**：Whisper 微觀字級對齊後的語意句子完整劇本。

---

## 後期剪輯軟體匯入指引

* **DaVinci Resolve**：
  1. 打開專案，在媒體池點選右鍵 `匯入` -> `時間軸...` (或按 `Ctrl+Shift+I` / `Cmd+Shift+I`)。
  2. 選擇 `<檔名>_edl.xml`，即可直接帶出所有切割段落與原片連結。
* **Adobe Premiere Pro**：
  1. `檔案` -> `匯入` (Import)，選擇 `<檔名>_edl.xml`。
  2. 匯入後會在專案面板生成一個 Sequence，雙擊即可在時間軸展開剪輯成果。
* **Final Cut Pro X**：
  1. `File` -> `Import` -> `XML...`。
  2. 選擇 `<檔名>_edl.fcpxml` 即可直接生成剪輯事件。

---

## 授權條款

MIT License

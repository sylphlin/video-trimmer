# Video Trimmer (`video-trimmer`)

> **AI 驅動之智慧影片粗剪與修剪引擎**
> 
> 專為口播說話影片（Talking-head videos）、影音 Podcast、教學導讀與公開演講打造之端到端智慧影片粗剪工具。
> 結合 **Gemini 3.8 Flash 多模態原生視訊理解**、**Whisper 詞級聲學時間戳對齊**（支援 Apple Silicon Metal GPU 硬體加速 `mlx-whisper`）與 **聲學起振自動貼齊（Acoustic Onset Snapping / Smart Gap Shortening）**。

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

[English (en)](README.md) | [繁體中文 (zh-TW)](README.zh-TW.md) | [简体中文 (zh-CN)](README.zh-CN.md) | [日本語 (ja)](README.ja.md) | [한국어 (ko)](README.ko.md)

---

## 核心特色與技術創新

1. **原生多模態視訊理解（Gemini 3.8 Flash）**：
   - 不依賴單純的語音文字轉錄（ASR）比對。
   - 透過 Gemini Files API / Interactions API 直傳原始視訊，同步評估**講者眼神注視、臉部表情、吃螺絲與重錄片段**。
2. **智慧「最終錄製優先（Last Take Wins）」演算法**：
   - 自動辨識講者口誤、忘詞、語氣練習或同段落重複錄製，精準保留最後一次流暢成功的錄製（Take）。
3. **多模態動態講者分離（Multimodal Active Speaker Diarization）**：
   - 同步解析鏡頭視覺線索（眼神、嘴型同步、肢體動作）與音訊聲學特徵（胸前領夾麥克風 vs 遠處空間殘響）。
   - 準確區隔畫面主角講者與場外工作人員喊話（例如「Action」、「CTA-S2」、「CDA84」）或鏡頭外的幕後花絮閒聊，無需依賴繁重的本地語者分離模型即可達成角色分離。
4. **聲學起振自動貼齊（Acoustic Onset Snapping / Smart Gap Shortening）**：
   - Whisper 轉錄時間戳往往比說話者真正發聲早 0.3 秒至 0.8 秒。`video-trimmer` 動態掃描波形能量，將剪切起點精確貼齊在**聲帶起振前 80ms**，消除說話前尷尬的多餘死寂。
5. **自適應環境底噪與尾音追蹤**：
   - 動態分析局部空間底噪，保護輕柔鼻音尾音（例如日語「〜ございます」、發音弱化字尾）不被截斷。
6. **15ms 音訊等功率微淡入淡出（Audio Equal-Power Micro-Crossfade）**：
   - FFmpeg 算圖輸出時，在每個剪輯接點自動套用 15ms 微淡入淡出（`afade`），徹底消除數位爆音（Click/Pop）與底噪階梯斷差。
7. **腳本引導對齊（`--script`）**：
   - 可選配傳入拍攝腳本或逐字稿，引導分段對齊，避免遺漏關鍵腳本內容。
8. **一鍵匯出主流 NLE 剪輯專案**：
   - 生成業界標準 **FCP 7 XML**（相容 Adobe Premiere Pro 與 DaVinci Resolve）與 **FCPXML**（Final Cut Pro X），並可直接算圖輸出高品質 **MP4**。

---

## 專案結構

本專案完全符合 [Agent Plugins 1.0 Specification](https://agent-plugins.org/) 與 [Agent Skills Specification](https://agentskills.io/specification)：

```text
video-trimmer/
├── plugin.json                 # Agent Plugins 1.0 規範清單
├── rules/
│   └── AGENTS.md               # 外部 AI Client 執行期守則（嚴格唯讀與 Fail-Fast 防護）
├── skills/
│   └── video-trimmer/
│       └── SKILL.md            # Agent Skill 規格文件與執行指引
├── AGENTS.md                   # 專案長期維護與開發守則（ASD-STE100 英文標準）
├── README.md                   # 公開說明文件（英文）
├── README.zh-TW.md             # 公開說明文件（繁體中文）
├── README.zh-CN.md             # 公開說明文件（簡體中文）
├── README.ja.md                # 公開說明文件（日文）
├── README.ko.md                # 公開說明文件（韓文）
├── LICENSE                     # MIT 開源授權
├── .env.example                # Vertex AI 與 GCS 環境變數範本
├── setup.sh                    # 100% Native gcloud GCP 資源配置腳本（零 Terraform 依賴）
├── pyproject.toml              # PEP 621 Python 打包與 CLI 命令設定
├── requirements.txt            # Python 執行環境依賴
├── video_trimmer.py            # 主 CLI 命令轉發器
├── auto_rough_cut.py           # 向後相容包裝器
├── scripts/                    # 核心引擎模組
│   ├── __init__.py
│   ├── video_trimmer.py        # CLI 參數解析與主要管線編排
│   ├── constants.py            # 集中管理之命名常數
│   ├── exceptions.py           # 自定義例外階層架構
│   ├── acoustic.py             # CPS、動態保護邊界、聲學能量偵測
│   ├── transcribe.py           # Whisper 語音轉錄、語意句合併、剪輯區間對齊
│   ├── gemini_client.py        # Vertex AI (ADC) 用戶端與多模態推論
│   ├── gcs_utils.py            # GCS 暫存上傳與臨時檔案清除
│   ├── exporters.py            # FCP7 XML / FCPXML / CSV 輸出器
│   └── render.py               # ffprobe 規格檢驗與 ffmpeg 渲染輸出
├── tests/                      # 離線單元測試套件
├── prompts/
│   └── video_cut_prompt.md     # 多模態剪輯 Prompt 規格
└── examples/                   # 範例專案輸出（EDL、XML、FCPXML、JSON）
```

---

## 快速開始

### 1. 系統環境需求

請確保系統已安裝 [FFmpeg](https://ffmpeg.org/) 並已加入 `PATH`：

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg
```

### 2. 安裝與佈署

#### 方法 A：Google Antigravity 與 Agent Plugins 1.0 安裝（AI Agent 推薦方式）

可直接安裝至 Google Antigravity 或任何支援 [Agent Plugins 1.0](https://agent-plugins.org/) 規範之 AI Client：

1. **安裝為 Agent Plugin（推薦：自動載入 `plugin.json` 與 `rules/AGENTS.md` 唯讀保護）**：
   - **全域插件（Global Plugin）**（所有專案與工作區通用）：
     ```bash
     git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/plugins/video-trimmer
     ```
   - **工作區插件（Workspace Plugin）**（僅限當前工作區）：
     ```bash
     git clone https://github.com/sylphlin/video-trimmer.git .agents/plugins/video-trimmer
     ```

#### 方法 B：獨立 Python CLI 安裝

複製倉庫並於本機安裝依賴：

```bash
git clone https://github.com/sylphlin/video-trimmer.git
cd video-trimmer

# 安裝核心依賴
pip install -r requirements.txt

# （推薦 macOS Apple Silicon 用戶）安裝 mlx-whisper 以取得 Metal GPU 硬體加速
pip install mlx-whisper

# 或以可編輯模式安裝 CLI 工具
pip install -e .
```

### 3. Google Cloud 原生資源配置（100% Native gcloud，零 Terraform 依賴）

本工具專用 **Google Cloud Vertex AI** 搭配 **應用程式預設憑證（ADC）** 以及 **Google Cloud Storage (GCS)** 進行多模態視訊暫存。

所有雲端資源（GCS 儲存貯體、CORS 設定、2 天暫存檔案自動清除生命週期、專屬 Service Account 與最小權限 IAM 角色）皆直接透過原生 `gcloud` 指令配置——**無任何外部 Terraform 依賴，100% Cloud Shell Ready**。

#### 選項 A：一鍵自動化配置（推薦）

直接執行隨附之自動化配置腳本：

```bash
# 賦予執行權限（僅需執行一次）
chmod +x setup.sh

# 自動化配置（自動讀取現有 gcloud 專案並配置 .env）：
./setup.sh

# 或明確指定 GCP 專案與區域：
./setup.sh --project 你的專案ID --region us-central1

# 乾跑預覽（不修改任何雲端資源）：
./setup.sh --dry-run
```

該腳本將自動完成：
1. 建立並驗證 GCS 貯體 `gs://video-preprocessing-${PROJECT_ID}`，啟用 `--uniform-bucket-level-access` 與 `--public-access-prevention`。
2. 配置 24 小時快取之 CORS 規則（支援 `GET` 與 `HEAD`），以利 Signed URL 影片串流。
3. 為暫存目錄 `raw/` 配置 **2 天自動清除生命週期規則**，避免雲端儲存空間累積費用。
4. 建立專屬 Service Account `video-trimmer-sa` 並授予最小權限（`roles/storage.objectUser`、`roles/aiplatform.user`、`roles/logging.logWriter`）。
5. 自動寫入本機 `.env` 設定檔。

#### 選項 B：手動原生 gcloud 配置

若偏好在終端機中手動執行原生指令：

```bash
export PROJECT_ID="你的GCP專案ID"
export REGION="us-central1"
export BUCKET_NAME="video-preprocessing-${PROJECT_ID}"
export SA="video-trimmer-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# 1. 建立 GCS 暫存儲存貯體
gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access \
    --public-access-prevention

# 2. 配置暫存影片 2 天自動清除規則
cat << 'EOF' > /tmp/lifecycle.json
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {
        "age": 2,
        "matchesPrefix": ["raw/"]
      }
    }
  ]
}
EOF
gcloud storage buckets update "gs://${BUCKET_NAME}" --lifecycle-file=/tmp/lifecycle.json

# 3. 建立專屬 Service Account 並綁定最小權限 IAM 角色
gcloud iam service-accounts create video-trimmer-sa \
    --display-name="Video Trimmer Service Account" \
    --project="${PROJECT_ID}"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA}" \
    --role="roles/aiplatform.user"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
    --member="serviceAccount:${SA}" \
    --role="roles/storage.objectUser"

# 4. 於本機進行 ADC 憑證登入驗證
gcloud auth application-default login

# 5. 配置本機 .env
cp .env.example .env
# 於 .env 中填寫 GOOGLE_CLOUD_PROJECT、VIDEO_TRIMMER_BUCKET 等參數
```

---

## 使用方法

對原始拍攝視訊執行粗剪管線：

```bash
# 使用已安裝之 CLI 命令：
video-trimmer -i "/路徑/至/raw_footage.mp4"

# 或直接使用 Python 執行：
python video_trimmer.py -i "/路徑/至/raw_footage.mp4"
```

### 進階範例

```bash
# 1. 提供拍攝腳本以進行章節引導對齊
video-trimmer -i "take 1.mp4" --script "script.md"

# 2. 啟用 Agentic 視訊理解（動態多輪影格探索）
video-trimmer -i "interview.mp4" --agentic

# 3. 採用緊湊節奏模式（適合高節奏知識型 YouTube 說書或科技評測）
video-trimmer -i "news.mp4" --pacing compact --suffix "fast"

# 4. 使用已快取的 EDL JSON 重新算圖剪接（本機秒級處理，無須重新呼叫雲端 API）
video-trimmer -i "take 1.mp4" --cached-json "take 1_agentic_edl.json" --suffix "fine_tuned"
```

---

## CLI 命令列參數參考

| 參數 | 縮寫 | 預設值 | 說明 |
| :--- | :---: | :---: | :--- |
| `--input` | `-i` | *(必要)* | 輸入之原始視訊檔案路徑（`.mp4`、`.mov`）。 |
| `--output-dir` | `-o` | 視訊同目錄 | 儲存所有產出檔案之目錄路徑。 |
| `--model` | `-m` | `gemini-3.8-flash` | Gemini 模型識別碼（預設自 `$MODEL_NAME` 讀取）。 |
| `--project` | | `None` | Google Cloud 專案 ID（預設自 `$GOOGLE_CLOUD_PROJECT` 或 ADC 讀取）。 |
| `--region` | | `None` | Vertex AI 地理位置（預設自 `$GOOGLE_CLOUD_LOCATION` 或 `global` 讀取）。 |
| `--bucket` | | `None` | 用於視訊暫存之 GCS 貯體名稱（預設自 `$VIDEO_TRIMMER_BUCKET` 讀取）。 |
| `--keep-gcs-upload` | | `False` | 保留 GCS 上的暫存視訊，不於推論完成後立即刪除。 |
| `--script` | `-s` | `None` | 拍攝腳本或逐字稿文字檔路徑（`.md` / `.txt`）。 |
| `--agentic` | | `False` | 啟用動態影格檢索的多輪 Agentic 視訊理解模式。 |
| `--pacing` | `-p` | `auto` | 語音節奏保護策略：`auto`（動態 CPS）、`compact`（緊湊）、`breathing`（放寬呼吸停頓）。 |
| `--cached-json`| | `None` | 傳入現有 EDL JSON 檔案以略過雲端 Gemini 推論，直接進行本機剪輯渲染。 |
| `--suffix` | | `None` | 為產出之檔案名稱加入自訂後綴標籤。 |
| `--crf` | | `18` | FFmpeg H.264 算圖之 CRF 畫質參數（18 為視覺無損）。 |
| `--skip-whisper`| | `False` | 略過本機 Whisper 轉錄（改用語音能量 Fallback 模式）。 |
| `--verbose` | | `False` | 輸出詳細 DEBUG 層級日誌（預設為 INFO）。 |

---

## 產出檔案說明

針對輸入視訊 `take 1.mp4`，`video-trimmer` 將產生以下檔案：

1. **`take 1_<tag>_trimmed.mp4`**：已自動完成粗剪組合、包含等功率音訊微淡入淡出的最終渲染影片。
2. **`take 1_<tag>_edl.xml`**：標準 FCP 7 XML 剪輯時間軸，相容於 **Adobe Premiere Pro** 與 **DaVinci Resolve**。
3. **`take 1_<tag>_edl.fcpxml`**：專用於 **Final Cut Pro X** 之 Apple FCPXML 剪輯時間軸。
4. **`take 1_<tag>_edl.json`**：包含保留語句、講者速度 CPS、時間戳之結構化剪輯決策 JSON。
5. **`take 1_<tag>_edl.csv`**：可用於試算表檢視之剪輯片段對照表，含視覺與聲學驗證備註。
6. **`take 1_whisper_sentences.json`**：包含詞級時間戳之完整語音轉錄分析紀錄。
7. **`usage_log.jsonl`**（位於輸出目錄）：每次呼叫 Gemini API 之詳細紀錄，追蹤 Token 消耗量與呼叫耗時，利於用量統計。

---

## 匯入非線性剪輯軟體（NLE）

- **DaVinci Resolve**：
  1. 於媒體集區（Media Pool）點選右鍵 ➜ `時間線` ➜ `匯入` ➜ `AAF / EDL / XML...`（快捷鍵 `Ctrl+Shift+I` / `Cmd+Shift+I`）。
  2. 選取生成的 `_edl.xml`，即可瞬間建立連結回原始視訊的精準粗剪時間線。
- **Adobe Premiere Pro**：
  1. 點選選單 `檔案` ➜ `匯入...`（快捷鍵 `Cmd+I` / `Ctrl+I`）。
  2. 選取 `_edl.xml`，在專案面板中按兩下產生的序列即可直接編輯。
- **Final Cut Pro X**：
  1. 點選選單 `檔案` ➜ `匯入` ➜ `XML...`。
  2. 選取產生的 `_edl.fcpxml` 即可直接載入。

---

---

## ☁️ Google Drive 雲端硬碟直通與 GCS Lifecycle 自動清理規則 (ADC 零金鑰直連)

在實務製作流程中，攝影師常將單機 NG 毛片直接上傳至 **Google Drive（個人雲端硬碟或團隊共用雲端硬碟 Shared Drives）**。`video-trimmer` 支援透過 `gcloud` ADC（`drive.readonly` 權限）直接讀取 Google Drive 分享連結，並搭配 GCS 雙層智慧快取與自動清理：

### 1. 一鍵啟用雲端環境與 Google Drive 權限 (`./setup.sh`)
```bash
# 步驟 1：登入 ADC 並授予 Google Drive 唯讀權限
gcloud auth application-default login

# 步驟 2：一鍵啟用 Vertex AI / GCS / Drive API、建立儲存桶並掛載雙層 Lifecycle 規則
./setup.sh --project YOUR_GCP_PROJECT_ID
```

### 2. 📌 Google Drive 支援情境與實戰範例

| 支援情境 | 輸入參數格式 | 智慧快取與自動處理行為 |
| :--- | :--- | :--- |
| **情境 A：Google Drive 毛片直接粗剪**<br/>*(免手動從瀏覽器下載)* | `-i "https://drive.google.com/file/d/<FILE_ID>/view"`<br/>或 `-i "gdrive://<FILE_ID>"` | 透過 Drive API v3 驗證遠端 `md5Checksum`，自動快取至 `<output_dir>/gdrive_inputs/` 供本地 Whisper 詞級轉錄與 FFmpeg 渲染，並自動轉存至 GCS `raw/`（若遠端 `sha256` / `gdrive_md5` 已相符則秒級跳過上傳）。 |
| **情境 B：搭配雲端或本地講稿進行選鏡**<br/>*(Last-Take-Wins 對稿粗剪)* | `-i "<Google Drive 影片連結>"`<br/>`-s script.md --agentic` | 自動比對拍攝講稿與多次重錄 (Retakes)，保留最後一次完美 Take 並切除廢話與停頓，匯出 `.mp4`、`.xml` (Premiere/Resolve) 與 `.fcpxml` (Final Cut Pro)。 |
| **情境 C：GCS 既有雲端物件直連推論** | `-i "gs://video-preprocessing-proj/raw/take1.mp4"` | 若影片已位於 GCS，Vertex AI 直接讀取該 `gs://` URI，零重複上傳。 |

#### 💻 CLI 實戰指令範例：
```bash
# 【情境 A】直接貼上 Google Drive 毛片連結執行 Agentic Video 智慧粗剪並輸出 NLE 時間線：
python3 video_trimmer.py \
  -i "https://drive.google.com/file/d/1RawTakeVideoIdxxxxxx/view?usp=sharing" \
  --agentic -o output/

# 【情境 B】Google Drive 毛片連結 + 拍攝講稿對齊 + 緊湊節奏模式：
python3 video_trimmer.py \
  -i "https://drive.google.com/file/d/1RawTakeVideoIdxxxxxx/view?usp=sharing" \
  -s shooting_script.md --pacing compact --agentic -o output/
```

#### 💬 Antigravity Agent 自然語言對話範例：
> 「幫我把這支放在 Google Drive 上的訪談毛片自動剪掉 NG 重錄、口誤與超過 0.5 秒的空白停頓，輸出 Premiere XML 與粗剪成片：`https://drive.google.com/file/d/1RawTakeVideoIdxxxxxx/view?usp=sharing`」

### 3. 🗑️ GCS Bucket Lifecycle 雙層自動清理規則 (`raw/` 2天 / 產出物 15天)

| GCS 路徑前綴 (`matchesPrefix`) | 儲存檔案類型 | 保留期限 (`age`) | 規則說明 |
| :--- | :--- | :--- | :--- |
| **`raw/`** | 供 Vertex AI 推論暫存之原始視訊 (`raw/<filename>.mp4`) | **2 天 (`age: 2`)** | 保留 2 天讓同專案重複微調參數（如 `--pacing`）時秒級命中 `sha256` / `gdrive_md5` 快取免重傳，2 天後由 GCS 自動刪除。 |
| **`output/`、`deliverables/`、`trimmed/`** | 雲端備份之粗剪成片、XML/FCPXML 時間線與 JSON/CSV 報告 | **15 天 (`age: 15`)** | 產出物保留 15 天供團隊下載與審閱，15 天後自動清理。 |

---

## 開發與測試

執行離線單元測試（使用合成音訊與固定測試夾具，無須連接真實視訊或 API）：

```bash
pip install -e ".[dev]"
pytest tests/
# 或
python3 -m unittest discover tests
```

---

## 授權條款

[MIT License](LICENSE) © 2026 sylphlin

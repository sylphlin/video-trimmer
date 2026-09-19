# Video Trimmer (`video-trimmer`)

> **AI搭載型スマート動画ラフカット＆トリミングエンジン**
> 
> トーキングヘッド動画、ビデオポッドキャスト、チュートリアル解説、スピーチ向けに設計されたエンドツーエンドのインテリジェント動画ラフカットツール。
> **Gemini 3.8 Flash マルチモーダルネイティブ動画理解**、**Whisper 単語単位音響タイムスタンプ同期**（Apple Silicon Metal GPU アクセラレーション `mlx-whisper` 対応）、および **音響オンセット自動スナップ（Acoustic Onset Snapping / Smart Gap Shortening）** を統合。

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

[English (en)](README.md) | [繁體中文 (zh-TW)](README.zh-TW.md) | [简体中文 (zh-CN)](README.zh-CN.md) | [日本語 (ja)](README.ja.md) | [한국어 (ko)](README.ko.md)

---

## 主な特長と技術革新

1. **マルチモーダルネイティブ動画理解（Gemini 3.8 Flash）**：
   - 単純な音声文字起こし（ASR）テキスト一致だけに依存しません。
   - Gemini Files API / Interactions API を介して素材動画を直接アップロードし、**話者の視線、表情、言い淀み（噛み）、リテイク** を同時に総合判定します。
2. **インテリジェント「最終テイク優先（Last Take Wins）」アルゴリズム**：
   - 言い間違い、セリフの練習、同一セクションの複数回リテイクを自動検出し、最終的なベストテイクのみを正確に抽出・維持します。
3. **マルチモーダル動的話者分離（Multimodal Active Speaker Diarization）**：
   - 映像上の視覚手がかり（視線、口の動きの同期、身振り手振り）と音響特徴（ピンマイク近接音 vs 遠くの部屋の反響）を連動して解析します。
   - カメラ前のメイン話者と、オフカメラのディレクター・撮影スタッフによるキュー出し（例:「Action」「CTA-S2」「CDA84」）やテイク間の雑談を的確に分離。重厚なローカル話者分離モデルを不要にします。
4. **音響オンセット自動スナップ（Acoustic Onset Snapping / Smart Gap Shortening）**：
   - Whisper の文字起こしタイムスタンプは、話者が実際に発声を開始する 0.3 秒〜0.8 秒前にトリガーされる傾向があります。`video-trimmer` は波形エネルギーを動的に解析し、**声帯振動開始の 80ms 前** にカット始点を正確にスナップさせ、発話前の不自然な沈黙を排除します。
5. **適応型環境ノイズフロア＆語尾減衰トラッキング**：
   - 周囲の環境ノイズを動的に分析し、繊細な鼻音や弱化する語尾（例:「〜ございます」や語尾のフェード）の欠落を防ぎます。
6. **15ms 音声等パワーマイクロクロスフェード（Audio Equal-Power Micro-Crossfade）**：
   - FFmpeg レンダリング時、すべてのカット境界に 15ms のイコールパワーマイクロフェードを自動適用し、ジャンプカット時のデジタルポップノイズや背景ノイズの段差を完全解消します。
7. **撮影台本ガイダンス（`--script`）**：
   - 撮影台本や原稿テキスト（`.md` / `.txt`）をオプション指定可能。章立てや構成に沿った的確なカット選定を支援します。
8. **主要 NLE プロジェクトへのワンクリック書き出し**：
   - 業界標準の **FCP 7 XML**（Adobe Premiere Pro および DaVinci Resolve 対応）と **FCPXML**（Final Cut Pro X 対応）を出力し、高品質 **MP4** の直接レンダリングにも対応します。

---

## ディレクトリ構造

本プロジェクトは [Agent Plugins 1.0 Specification](https://agent-plugins.org/) および [Agent Skills Specification](https://agentskills.io/specification) に準拠しています：

```text
video-trimmer/
├── plugin.json                 # Agent Plugins 1.0 仕様マニフェスト
├── rules/
│   └── AGENTS.md               # 外部 AI クライアント実行時インバリアント（厳格な読み取り専用・フェイルファスト）
├── skills/
│   └── video-trimmer/
│       └── SKILL.md            # Agent Skill 仕様定義マニュアル
├── AGENTS.md                   # プロジェクト保守・開発運用規範（ASD-STE100 英語基準）
├── README.md                   # 公開ドキュメント（英語）
├── README.zh-TW.md             # 公開ドキュメント（繁体字中国語）
├── README.zh-CN.md             # 公開ドキュメント（簡体字中国語）
├── README.ja.md                # 公開ドキュメント（日本語）
├── README.ko.md                # 公開ドキュメント（韓国語）
├── LICENSE                     # MIT ライセンス
├── .env.example                # Vertex AI & GCS 環境変数テンプレート
├── setup.sh                    # 100% ネイティブ gcloud GCP リソースプロビジョニングスクリプト（Terraform 不要）
├── pyproject.toml              # PEP 621 Python パッケージ＆ CLI コマンド定義
├── requirements.txt            # Python ランタイム依存関係
├── video_trimmer.py            # メイン CLI コマンドフォワーダー
├── auto_rough_cut.py           # 後方互換ラッパー
├── scripts/                    # コアエンジンモジュール
│   ├── __init__.py
│   ├── video_trimmer.py        # CLI 引数解析＆パイプラインオーケストレーション
│   ├── constants.py            # 集中管理の名前付き定数
│   ├── exceptions.py           # カスタム例外階層
│   ├── acoustic.py             # CPS、動的マージン、音響エネルギー検出
│   ├── transcribe.py           # Whisper 音声認識、文結合、クリップ同期
│   ├── gemini_client.py        # Vertex AI (ADC) クライアント＆マルチモーダル推論
│   ├── gcs_utils.py            # GCS 一時アップロード＆クリーンアップ
│   ├── exporters.py            # FCP7 XML / FCPXML / CSV エクスポーター
│   └── render.py               # ffprobe 検証＆ ffmpeg レンダリング
├── tests/                      # オフライン単体テスト
├── prompts/
│   └── video_cut_prompt.md     # マルチモーダルカット選定プロンプト仕様
└── examples/                   # サンプル出力ファイル群（EDL、XML、FCPXML、JSON）
```

---

## クイックスタート

### 1. システム要件

システムに [FFmpeg](https://ffmpeg.org/) がインストールされ、`PATH` に登録されていることを確認してください：

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg
```

### 2. インストールとデプロイ

#### 方法 A: Google Antigravity & Agent Plugins 1.0 によるインストール（AI エージェント推奨）

Google Antigravity または [Agent Plugins 1.0](https://agent-plugins.org/) 準拠の AI クライアントに直接インストールします：

1. **Agent Plugin としてインストール（推奨: `plugin.json` と `rules/AGENTS.md` の保護規範を自動読込）**：
   - **グローバルプラグイン（Global Plugin）**（すべてのプロジェクトで利用可能、推奨）：
     ```bash
     git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/plugins/video-trimmer
     ```
   - **ワークスペースプラグイン（Workspace Plugin）**（現在のワークスペース限定）：
     ```bash
     git clone https://github.com/sylphlin/video-trimmer.git .agents/plugins/video-trimmer
     ```

#### 方法 B: スタンドアロン Python CLI インストール

リポジトリをクローンして依存関係をローカルにインストールします：

```bash
git clone https://github.com/sylphlin/video-trimmer.git
cd video-trimmer

# コア依存関係のインストール
pip install -r requirements.txt

# （macOS Apple Silicon 推奨）Metal GPU 加速用 mlx-whisper のインストール
pip install mlx-whisper

# 編集可能モードで CLI ツールとしてインストール
pip install -e .
```

### 3. Google Cloud ネイティブ環境設定（100% Native gcloud、Terraform 完全不要）

本ツールは動画の一時配置ストレージとして **Google Cloud Storage (GCS)** を使用し、推論処理に **Google Cloud Vertex AI**（**Application Default Credentials: ADC**）を排他的に使用します。

必要なクラウドリソース（GCS バケット、CORS 設定、2 日間一時ファイル自動削除ルール、専用サービスアカウント、最小特権 IAM ロール）はすべてネイティブな `gcloud` コマンドで構成されます（**Terraform 不要、Cloud Shell 対応**）。

#### オプション A: ワンクリック自動プロビジョニング（推奨）

同梱されている自動環境構築スクリプトを実行します：

```bash
# 実行権限の付与（初回のみ）
chmod +x setup.sh

# 自動プロビジョニング（既存の gcloud 設定を読み込み、.env を自動生成）：
./setup.sh

# または GCP プロジェクトとリージョンを明示的に指定：
./setup.sh --project YOUR_PROJECT_ID --region us-central1

# ドライラン確認（リソースを変更せずコマンドのみ確認）：
./setup.sh --dry-run
```

本スクリプトは以下を自動実行します：
1. GCS バケット `gs://video-preprocessing-${PROJECT_ID}` の作成・検証（`--uniform-bucket-level-access` および `--public-access-prevention` 適用）。
2. Signed URL による動画再生用の CORS 設定（24 時間キャッシュ、`GET` / `HEAD` 許可）。
3. プレフィックス `raw/` に対する **2 日間自動削除ライフサイクルルール** の設定（ストレージ費用の累積を防止）。
4. 専用サービスアカウント `video-trimmer-sa` の作成と最小特権の付与（`roles/storage.objectUser`、`roles/aiplatform.user`、`roles/logging.logWriter`）。
5. ローカル設定ファイル `.env` の自動更新。

#### オプション B: 手動によるネイティブ gcloud コマンド設定

シェル上で手動設定を行う場合：

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export BUCKET_NAME="video-preprocessing-${PROJECT_ID}"
export SA="video-trimmer-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# 1. GCS 一時バケットの作成
gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access \
    --public-access-prevention

# 2. 2 日間一時ファイル自動クリーンアップルールの構成
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

# 3. 専用サービスアカウントの作成と最小特権 IAM ロールの付与
gcloud iam service-accounts create video-trimmer-sa \
    --display-name="Video Trimmer Service Account" \
    --project="${PROJECT_ID}"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA}" \
    --role="roles/aiplatform.user"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
    --member="serviceAccount:${SA}" \
    --role="roles/storage.objectUser"

# 4. ローカル ADC 認証ログイン
gcloud auth application-default login

# 5. ローカル .env の設定
cp .env.example .env
# .env 内で GOOGLE_CLOUD_PROJECT や VIDEO_TRIMMER_BUCKET を設定
```

---

## 使用方法

素材動画に対してトリミングパイプラインを実行します：

```bash
# インストールした CLI コマンドで実行：
video-trimmer -i "/path/to/raw_footage.mp4"

# または Python で直接実行：
python video_trimmer.py -i "/path/to/raw_footage.mp4"
```

### 応用実行例

```bash
# 1. 撮影台本を指定して構成同期
video-trimmer -i "take 1.mp4" --script "script.md"

# 2. Agentic 動画理解（動的マルチターンフレーム探索）を有効化
video-trimmer -i "interview.mp4" --agentic

# 3. テンポの速い YouTube 解説動画向けのコンパクトペース設定
video-trimmer -i "news.mp4" --pacing compact --suffix "fast"

# 4. キャッシュ済み EDL JSON を用いた即時再レンダリング（ローカルで数秒処理）
video-trimmer -i "take 1.mp4" --cached-json "take 1_agentic_edl.json" --suffix "fine_tuned"
```

---

## CLI オプションリファレンス

| オプション | 短縮形 | デフォルト値 | 説明 |
| :--- | :---: | :---: | :--- |
| `--input` | `-i` | *(必須)* | 入力素材動画ファイルのパス（`.mp4`、`.mov`）。 |
| `--output-dir` | `-o` | 動画と同じディレクトリ | すべての出力ファイルを保存するディレクトリ。 |
| `--model` | `-m` | `gemini-3.8-flash` | Gemini モデル識別子（デフォルトは `$MODEL_NAME`）。 |
| `--project` | | `None` | Google Cloud プロジェクト ID（デフォルトは `$GOOGLE_CLOUD_PROJECT` または ADC）。 |
| `--region` | | `None` | Vertex AI リージョン（デフォルトは `$GOOGLE_CLOUD_LOCATION` または `global`）。 |
| `--bucket` | | `None` | 動画一時配置用 GCS バケット名（デフォルトは `$VIDEO_TRIMMER_BUCKET`）。 |
| `--keep-gcs-upload` | | `False` | 推論完了後も GCS の一時動画ファイルを削除せずに維持。 |
| `--script` | `-s` | `None` | 撮影台本・原稿テキストファイルへのパス（`.md` / `.txt`）。 |
| `--agentic` | | `False` | 動的フレーム探索を行う Agentic 動画理解モードを有効化。 |
| `--pacing` | `-p` | `auto` | ペース配分戦略: `auto`（動的 CPS）、`compact`（緊密）、`breathing`（ブレス尊重）。 |
| `--cached-json`| | `None` | 既存の EDL JSON を指定して Gemini 推論をバイパスし、ローカルで再カット出力。 |
| `--suffix` | | `None` | 生成される出力ファイル名に付与するカスタムサフィックスタグ。 |
| `--crf` | | `18` | FFmpeg H.264 レンダリング時の CRF 品質パラメータ（18 = 視覚的ロスレス）。 |
| `--skip-whisper`| | `False` | ローカル Whisper 認識をスキップ（音響エネルギーフォールバックを使用）。 |
| `--verbose` | | `False` | 詳細な DEBUG レベルのログを出力（デフォルトは INFO）。 |

---

## 生成される出力ファイル

入力ファイル `take 1.mp4` に対し、`video-trimmer` は以下のファイルを生成します：

1. **`take 1_<tag>_trimmed.mp4`**：マイクロクロスフェードが適用された最終結合レンダリング動画。
2. **`take 1_<tag>_edl.xml`**：**Adobe Premiere Pro** および **DaVinci Resolve** に対応する標準 FCP 7 XML タイムライン。
3. **`take 1_<tag>_edl.fcpxml`**：**Final Cut Pro X** 専用 Apple FCPXML タイムライン。
4. **`take 1_<tag>_edl.json`**：採用発話文、話速 CPS、タイムスタンプを含む構造化判定 JSON。
5. **`take 1_<tag>_edl.csv`**：視覚・音響検証メモを含む表計算用カットリスト。
6. **`take 1_whisper_sentences.json`**：単語単位タイムスタンプを含む完全な文字起こし文データ。
7. **`usage_log.jsonl`**（出力ディレクトリ内）：Gemini API 呼び出しごとのトークン消費量と所要時間の記録。

---

## NLE（動画編集ソフト）への取り込み

- **DaVinci Resolve**：
  1. メディアプールで右クリック ➜ `タイムライン` ➜ `読み込み` ➜ `AAF / EDL / XML...`（ショートカット: `Ctrl+Shift+I` / `Cmd+Shift+I`）。
  2. 生成された `_edl.xml` を選択すると、元動画にリンクしたカット済みタイムラインが瞬時に構築されます。
- **Adobe Premiere Pro**：
  1. `ファイル` ➜ `読み込み...`（ショートカット: `Cmd+I` / `Ctrl+I`）。
  2. `_edl.xml` を選択し、プロジェクトパネルに現れたシーケンスをダブルクリックして開きます。
- **Final Cut Pro X**：
  1. `ファイル` ➜ `読み込み` ➜ `XML...`。
  2. 生成された `_edl.fcpxml` を選択して読み込みます。

---

## 開発とテスト

オフライン単体テストの実行（合成音声とテストフィクスチャを使用し、外部動画や API 呼び出しなしで完結）：

```bash
pip install -e ".[dev]"
pytest tests/
# または
python3 -m unittest discover tests
```

---

## ライセンス

[MIT License](LICENSE) © 2026 sylphlin


---

## ☁️ Google Drive 直結シナリオと GCS ライフサイクル自動削除ルール (ADC 認証)

`video-trimmer` は `gcloud` ADC（`drive.readonly` スコープ）を使用して Google Drive の共有リンク（`https://drive.google.com/file/d/.../view`）を直接入力として受け付け、MD5/SHA-256 キャッシュにより高速にステージングします。

```bash
# Step 1: Google Drive 読み取り権限を含めて ADC ログイン
gcloud auth application-default login
./setup.sh --project YOUR_GCP_PROJECT_ID

# Step 2: Google Drive 上の素材リンクを直接指定して Agentic 粗編集を実行
python3 video_trimmer.py \
  -i "https://drive.google.com/file/d/1RawTakeVideoIdxxxxxx/view?usp=sharing" \
  --agentic -o output/
```

- **GCS 2段階ライフサイクル**：`raw/`（一時ステージング動画）は **2日間 (`age: 2`)** 保持後に自動削除、`output/`・`deliverables/`・`trimmed/`（成果物）は **15日間 (`age: 15`)** 保持後に自動削除されます。

---

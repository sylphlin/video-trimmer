# Video Trimmer (`video-trimmer`)

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

[English (en)](README.md) | [繁體中文 (zh-TW)](README.zh-TW.md) | [简体中文 (zh-CN)](README.zh-CN.md) | [日本語 (ja)](README.ja.md) | [한국어 (ko)](README.ko.md)

---

## 概要 (Overview)

**Video Trimmer** は、トーク動画、チュートリアル、プレゼンテーション録画向けの AI 自動ラフカット＆トリミングエンジンです。**Google Vertex AI Gemini 3.8 Flash** のマルチモーダル動画推論、**Whisper 単語レベル音響タイムスタンプ**（`mlx-whisper`）、および **音響オンセット・スナッピング** を組み合わせ、NG テイクや言い淀み、無音区間を自動除去し、NLE タイムライン（`.xml`, `.fcpxml`, `.edl`, `.csv`）と MP4 動画を出力します。

---

## 主な機能

1. **マルチモーダル動画推論 (`gemini-3.8-flash`)**：話者の視線、表情、言い直し、リテイクを同時に評価します。
2. **Last-Take-Wins（最終成功テイクの採用）**：同一セクションの複数リテイクを検出し、最後の成功テイクのみを残します。
3. **音響オンセット・スナッピング (`tighten_clip_to_speech`)**：声帯振動の 80 ms 前にカット点を配置し、語頭の音素を欠損させずに無音を除去します。
4. **15 ms 等パワー音声マイクロクロスフェード**：すべてのカット境界に 15 ms のマイクロフェード（`afade=t=in:d=0.015:curve=iqsin` / `afade=t=out:d=0.015:curve=oqsin`）を適用し、ポップノイズを防止します。
5. **マルチ NLE タイムライン出力**：**FCP7 XML**（Premiere Pro / DaVinci Resolve）、**FCPXML**（Final Cut Pro）、**CMX 3600 EDL**、および **CSV** を出力します。

---

## セットアップと Google Cloud 構成

```bash
# 1. FFmpeg と Python パッケージのインストール
brew install ffmpeg
git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/plugins/video-trimmer
pip install -r requirements.txt
pip install mlx-whisper

# 2. ADC 認証と setup.sh の実行
gcloud auth application-default login
chmod +x setup.sh
./setup.sh --project YOUR_GCP_PROJECT_ID --region us-central1
```

---

## コマンドライン使用法 (CLI Usage)

```bash
# 標準ラフカット実行（Agentic 動画理解モード有効）
python3 video_trimmer.py -i "raw_footage.mp4" --agentic

# 台本（スクリプト）を用いたアライメント
python3 video_trimmer.py -i "raw_footage.mp4" --script "shooting_script.md" --agentic

# 静的マルチモーダルモード（static モード使用時のみ --agentic を省略）
python3 video_trimmer.py -i "raw_footage.mp4"

# Google Drive 共有リンクからの直接ラフカット
python3 video_trimmer.py -i "https://drive.google.com/file/d/FILE_ID/view?usp=sharing" --agentic -o output/
```

---

## Google Drive 連携と GCS 2 階層ライフサイクルポリシー

| GCS パス接頭辞 (`matchesPrefix`) | 保存対象 | 保持期間 (`age`) | 目的 |
| :--- | :--- | :--- | :--- |
| **`raw/`** | 一時ステージング動画 (`raw/<filename>.mp4`) | **2 日間 (`age: 2`)** | 推論完了後に即時削除され、バックアップとして 2 日後に自動削除されます。 |
| **`output/`**、**`deliverables/`**、**`trimmed/`** | トリミング済み動画、XML/FCPXML タイムライン | **15 日間 (`age: 15`)** | チーム確認用に 15 日間保持した後、自動削除します。 |

---

## ライセンス (License)

本プロジェクトは [MIT License](LICENSE) の下で提供されています。

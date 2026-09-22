# Video Trimmer (`video-trimmer`)

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

[English (en)](README.md) | [繁體中文 (zh-TW)](README.zh-TW.md) | [简体中文 (zh-CN)](README.zh-CN.md) | [日本語 (ja)](README.ja.md) | [한국어 (ko)](README.ko.md)

---

## 项目概览 (Overview)

**Video Trimmer** 是面向口播视频、教程与演讲录像的 AI 自动粗剪引擎。系统结合 **Google Vertex AI Gemini 3.8 Flash** 多模态视频推理、**Whisper 逐字声学时间戳**（`mlx-whisper`）与 **声学起音锁定（Acoustic Onset Snapping）**。每次运行会自动剔除 NG 重录、口误与静音停顿，并导出专业 NLE 时间线（`.xml`, `.fcpxml`, `.edl`, `.csv`）及渲染完成的 MP4 视频。

---

## 核心技术能力

1. **多模态原生视频推理 (`gemini-3.8-flash`)**：同步评估讲者眼神、面部表情、卡顿与重录段落。
2. **Last-Take-Wins（保留最后成功重录）**：自动识别同一段落的多次尝试，仅保留最后一次完整成功的镜头。
3. **多模态主讲人分离**：结合画面口型与麦克风距离，区分出镜主讲人与场外导演口令。
4. **声学起音锁定 (`tighten_clip_to_speech`)**：将剪辑入点锁定在声带振动前 80 ms，消除冗长前导空白且不截断字首音素。
5. **15 ms 等功率音频微交叉淡化**：在每个剪辑边界注入 15 ms 等功率淡入淡出（`afade=t=in:d=0.015:curve=iqsin` 与 `afade=t=out:d=0.015:curve=oqsin`），消除音频跳接爆音。
6. **多平台 NLE 时间线互通**：导出 **FCP7 XML**（Premiere Pro / DaVinci Resolve）、**FCPXML**（Final Cut Pro）、**CMX 3600 EDL** 与 **CSV**。

---

## 安装与 Google Cloud 环境配置

```bash
# 1. 安装 FFmpeg 与 Python 依赖
brew install ffmpeg
git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/plugins/video-trimmer
pip install -r requirements.txt
pip install mlx-whisper

# 2. 授权 ADC 并运行 setup.sh
gcloud auth application-default login
chmod +x setup.sh
./setup.sh --project YOUR_GCP_PROJECT_ID --region us-central1
```

---

## 命令行使用说明 (CLI Usage)

```bash
# 标准视频粗剪（默认启用 Agentic 视频理解模式）
python3 video_trimmer.py -i "raw_footage.mp4" --agentic

# 配合拍摄脚本对齐
python3 video_trimmer.py -i "raw_footage.mp4" --script "shooting_script.md" --agentic

# 静态抽帧模式（仅在需要 Static Multimodal 时省略 --agentic）
python3 video_trimmer.py -i "raw_footage.mp4"

# 直接从 Google Drive 分享链接进行粗剪
python3 video_trimmer.py -i "https://drive.google.com/file/d/FILE_ID/view?usp=sharing" --agentic -o output/
```

---

## Google Drive 直连与 GCS 双层生命周期规则

| GCS 路径前缀 (`matchesPrefix`) | 存储对象 | 保留天数 (`age`) | 说明 |
| :--- | :--- | :--- | :--- |
| **`raw/`** | 暂存原始视频 (`raw/<filename>.mp4`) | **2 天 (`age: 2`)** | 推理后立即删除，并以 2 天自动清理规则作为安全兜底。 |
| **`output/`**、**`deliverables/`**、**`trimmed/`** | 粗剪视频、XML/FCPXML 时间线与报告 | **15 天 (`age: 15`)** | 保留 15 天供团队审阅，到期自动清理。 |

---

## 许可证 (License)

本项目采用 [MIT License](LICENSE) 授权。

# Video Trimmer (`video-trimmer`)

> **AI 驱动的智能视频粗剪与修剪引擎**
> 
> 专为口播视频（Talking-head videos）、音视频播客、教程讲解与公开演讲打造的端到端智能视频粗剪工具。
> 结合 **Gemini 3.8 Flash 多模态原生视频理解**、**Whisper 词级声学时间戳对齐**（支持 Apple Silicon Metal GPU 硬件加速 `mlx-whisper`）与 **声学起振自动吸附（Acoustic Onset Snapping / Smart Gap Shortening）**。

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

[English (en)](README.md) | [繁體中文 (zh-TW)](README.zh-TW.md) | [简体中文 (zh-CN)](README.zh-CN.md) | [日本語 (ja)](README.ja.md) | [한국어 (ko)](README.ko.md)

---

## 核心特色与技术创新

1. **原生多模态视频理解（Gemini 3.8 Flash）**：
   - 不单纯依赖脆弱的语音文字转录（ASR）比对。
   - 通过 Gemini Files API / Interactions API 直传原始视频，同步评估**讲者眼神注视、面部表情、卡壳卡顿与重录片段**。
2. **智能“最终录制优先（Last Take Wins）”算法**：
   - 自动识别讲者口误、忘词、语气练习或同段落重复录制，精准保留最后一次流畅成功的录制（Take）。
3. **多模态动态讲者分离（Multimodal Active Speaker Diarization）**：
   - 同步解析镜头视觉线索（眼神注视、唇形同步、肢体语言）与音频声学特征（胸前领夹麦克风 vs 远处空间残响）。
   - 准确区分画面主角讲者与场外工作人员喊话（例如“Action”、“CTA-S2”、“CDA84”）或镜头外的幕后花絮闲聊，无需依赖繁重的本地语者分离模型即可达成角色分离。
4. **声学起振自动吸附（Acoustic Onset Snapping / Smart Gap Shortening）**：
   - Whisper 转录时间戳往往比说话者真正发声早 0.3 秒至 0.8 秒。`video-trimmer` 动态扫描波形能量，将剪切起点精确吸附在**声带起振前 80ms**，消除说话前尴尬的多余停顿。
5. **自适应环境底噪与尾音追踪**：
   - 动态分析局部空间底噪，保护轻柔鼻音尾音（例如日语“〜ございます”、发音弱化字尾）不被截断。
6. **15ms 音频等功率微淡入淡出（Audio Equal-Power Micro-Crossfade）**：
   - FFmpeg 渲染输出时，在每个剪辑接点自动应用 15ms 微淡入淡出（`afade`），彻底消除数字爆音（Click/Pop）与底噪阶梯断差。
7. **脚本引导对齐（`--script`）**：
   - 可选传入拍摄脚本或逐字稿，引导分段对齐，避免遗漏关键脚本内容。
8. **一键导出主流 NLE 剪辑工程**：
   - 生成业界标准 **FCP 7 XML**（兼容 Adobe Premiere Pro 与 DaVinci Resolve）与 **FCPXML**（Final Cut Pro X），并可直接渲染输出高质量 **MP4**。

---

## 项目结构

本项目完全符合 [Agent Plugins 1.0 Specification](https://agent-plugins.org/) 与 [Agent Skills Specification](https://agentskills.io/specification)：

```text
video-trimmer/
├── plugin.json                 # Agent Plugins 1.0 规范清单
├── rules/
│   └── AGENTS.md               # 外部 AI Client 运行期守则（严格只读与 Fail-Fast 防护）
├── skills/
│   └── video-trimmer/          # 插件技能组合包（SKILL.md、scripts、prompts）
├── AGENTS.md                   # 项目长期维护与开发守则（ASD-STE100 英文标准）
├── SKILL.md                    # Agent Skill 规范文件与执行指引
├── README.md                   # 公开说明文件（英文）
├── README.zh-TW.md             # 公开说明文件（繁体中文）
├── README.zh-CN.md             # 公开说明文件（简体中文）
├── README.ja.md                # 公开说明文件（日文）
├── README.ko.md                # 公开说明文件（韩文）
├── LICENSE                     # MIT 开源许可证
├── .env.example                # Vertex AI 与 GCS 环境变量模板
├── setup.sh                    # 100% Native gcloud GCP 资源配置脚本（零 Terraform 依赖）
├── pyproject.toml              # PEP 621 Python 打包与 CLI 命令配置
├── requirements.txt            # Python 运行环境依赖
├── video_trimmer.py            # 主 CLI 命令转发器
├── auto_rough_cut.py           # 向后兼容包装器
├── scripts/                    # 核心引擎模块
│   ├── __init__.py
│   ├── video_trimmer.py        # CLI 参数解析与主要管线编排
│   ├── constants.py            # 集中管理的命名常量
│   ├── exceptions.py           # 自定义异常层级架构
│   ├── acoustic.py             # CPS、动态保护边距、声学能量检测
│   ├── transcribe.py           # Whisper 语音转录、语义句合并、剪辑区间对齐
│   ├── gemini_client.py        # Vertex AI (ADC) 客户端与多模态推理
│   ├── gcs_utils.py            # GCS 临时上传与临时文件清理
│   ├── exporters.py            # FCP7 XML / FCPXML / CSV 导出器
│   └── render.py               # ffprobe 规格检验与 ffmpeg 渲染输出
├── tests/                      # 离线单元测试套件
├── prompts/
│   └── video_cut_prompt.md     # 多模态剪辑 Prompt 规范
└── examples/                   # 示例工程输出（EDL、XML、FCPXML、JSON）
```

---

## 快速开始

### 1. 系统环境要求

请确保系统已安装 [FFmpeg](https://ffmpeg.org/) 并已配置于 `PATH`：

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg
```

### 2. 安装与部署

#### 方法 A：Google Antigravity 与 Agent Plugins 1.0 安装（AI Agent 推荐方式）

可直接安装至 Google Antigravity 或任何支持 [Agent Plugins 1.0](https://agent-plugins.org/) 规范的 AI Client：

1. **安装为 Agent Plugin（推荐：自动加载 `plugin.json` 与 `rules/AGENTS.md` 只读保护）**：
   - **全局插件（Global Plugin）**（所有项目与工作区通用）：
     ```bash
     git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/plugins/video-trimmer
     ```
   - **工作区插件（Workspace Plugin）**（仅限当前工作区）：
     ```bash
     git clone https://github.com/sylphlin/video-trimmer.git .agents/plugins/video-trimmer
     ```

2. **或安装为 Agent Skill**：
   - **全局技能（Global Skill）**：
     ```bash
     git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/skills/video-trimmer
     ```
   - **工作区技能（Workspace Skill）**：
     ```bash
     git clone https://github.com/sylphlin/video-trimmer.git .agent/skills/video-trimmer
     ```

#### 方法 B：独立 Python CLI 安装

克隆仓库并在本地安装依赖：

```bash
git clone https://github.com/sylphlin/video-trimmer.git
cd video-trimmer

# 安装核心依赖
pip install -r requirements.txt

# （推荐 macOS Apple Silicon 用户）安装 mlx-whisper 以获得 Metal GPU 硬件加速
pip install mlx-whisper

# 或以可编辑模式安装 CLI 工具
pip install -e .
```

### 3. Google Cloud 原生资源配置（100% Native gcloud，零 Terraform 依赖）

本工具专用于 **Google Cloud Vertex AI** 配合 **应用程序默认凭据（ADC）** 以及 **Google Cloud Storage (GCS)** 进行多模态视频暂存。

所有云端资源（GCS 存储桶、CORS 设置、2 天临时文件自动清理生命周期、专属 Service Account 与最小权限 IAM 角色）均直接通过原生 `gcloud` 命令配置——**无任何外部 Terraform 依赖，100% Cloud Shell Ready**。

#### 选项 A：一键自动化配置（推荐）

直接执行随附的自动化配置脚本：

```bash
# 赋予执行权限（仅需执行一次）
chmod +x setup.sh

# 自动化配置（自动读取现有 gcloud 项目并配置 .env）：
./setup.sh

# 或显式指定 GCP 项目与区域：
./setup.sh --project 你的项目ID --region us-central1

# 干跑预览（不修改任何云端资源）：
./setup.sh --dry-run
```

该脚本将自动完成：
1. 创建并验证 GCS 存储桶 `gs://video-preprocessing-${PROJECT_ID}`，启用 `--uniform-bucket-level-access` 与 `--public-access-prevention`。
2. 配置 24 小时缓存的 CORS 规则（支持 `GET` 与 `HEAD`），便于 Signed URL 视频流式传输。
3. 为暂存目录 `raw/` 配置 **2 天自动清理生命周期规则**，避免云端存储累积费用。
4. 创建专属 Service Account `video-trimmer-sa` 并授予最小权限（`roles/storage.objectUser`、`roles/aiplatform.user`、`roles/logging.logWriter`）。
5. 自动写入本地 `.env` 配置文件。

#### 选项 B：手动原生 gcloud 配置

若偏好在终端中手动执行原生命令：

```bash
export PROJECT_ID="你的GCP项目ID"
export REGION="us-central1"
export BUCKET_NAME="video-preprocessing-${PROJECT_ID}"
export SA="video-trimmer-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# 1. 创建 GCS 暂存存储桶
gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access \
    --public-access-prevention

# 2. 配置暂存视频 2 天自动清理规则
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

# 3. 创建专属 Service Account 并绑定最小权限 IAM 角色
gcloud iam service-accounts create video-trimmer-sa \
    --display-name="Video Trimmer Service Account" \
    --project="${PROJECT_ID}"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA}" \
    --role="roles/aiplatform.user"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
    --member="serviceAccount:${SA}" \
    --role="roles/storage.objectUser"

# 4. 在本地进行 ADC 凭据登录验证
gcloud auth application-default login

# 5. 配置本地 .env
cp .env.example .env
# 在 .env 中填写 GOOGLE_CLOUD_PROJECT、VIDEO_TRIMMER_BUCKET 等参数
```

---

## 使用方法

对原始拍摄视频执行粗剪管线：

```bash
# 使用已安装的 CLI 命令：
video-trimmer -i "/路径/至/raw_footage.mp4"

# 或直接使用 Python 执行：
python video_trimmer.py -i "/路径/至/raw_footage.mp4"
```

### 进阶示例

```bash
# 1. 提供拍摄脚本以进行章节引导对齐
video-trimmer -i "take 1.mp4" --script "script.md"

# 2. 启用 Agentic 视频理解（动态多轮帧检索）
video-trimmer -i "interview.mp4" --agentic

# 3. 采用紧凑节奏模式（适合高节奏知识型 YouTube 讲解或科技评测）
video-trimmer -i "news.mp4" --pacing compact --suffix "fast"

# 4. 使用已缓存的 EDL JSON 重新渲染剪辑（本地秒级处理，无需重新调用云端 API）
video-trimmer -i "take 1.mp4" --cached-json "take 1_agentic_edl.json" --suffix "fine_tuned"
```

---

## CLI 命令行参数参考

| 参数 | 缩写 | 默认值 | 说明 |
| :--- | :---: | :---: | :--- |
| `--input` | `-i` | *(必需)* | 输入的原始视频文件路径（`.mp4`、`.mov`）。 |
| `--output-dir` | `-o` | 视频同目录 | 保存所有输出文件的目录路径。 |
| `--model` | `-m` | `gemini-3.8-flash` | Gemini 模型标识符（默认自 `$MODEL_NAME` 读取）。 |
| `--project` | | `None` | Google Cloud 项目 ID（默认自 `$GOOGLE_CLOUD_PROJECT` 或 ADC 读取）。 |
| `--region` | | `None` | Vertex AI 地理位置（默认自 `$GOOGLE_CLOUD_LOCATION` 或 `global` 读取）。 |
| `--bucket` | | `None` | 用于视频暂存的 GCS 存储桶名称（默认自 `$VIDEO_TRIMMER_BUCKET` 读取）。 |
| `--keep-gcs-upload` | | `False` | 保留 GCS 上的暂存视频，不于推理完成后立即删除。 |
| `--script` | `-s` | `None` | 拍摄脚本或逐字稿文本文件路径（`.md` / `.txt`）。 |
| `--agentic` | | `False` | 启用动态帧检索的多轮 Agentic 视频理解模式。 |
| `--pacing` | `-p` | `auto` | 语音节奏保护策略：`auto`（动态 CPS）、`compact`（紧凑）、`breathing`（放宽呼吸停顿）。 |
| `--cached-json`| | `None` | 传入现有 EDL JSON 文件以跳过云端 Gemini 推理，直接进行本地剪辑渲染。 |
| `--suffix` | | `None` | 为输出的文件名称加入自定义后缀标签。 |
| `--crf` | | `18` | FFmpeg H.264 渲染的 CRF 画质参数（18 为视觉无损）。 |
| `--skip-whisper`| | `False` | 跳过本地 Whisper 转录（改用语音能量 Fallback 模式）。 |
| `--verbose` | | `False` | 输出详细 DEBUG 层级日志（默认为 INFO）。 |

---

## 输出文件说明

针对输入视频 `take 1.mp4`，`video-trimmer` 将生成以下文件：

1. **`take 1_<tag>_trimmed.mp4`**：已自动完成粗剪拼接、包含等功率音频微淡入淡出的最终渲染视频。
2. **`take 1_<tag>_edl.xml`**：标准 FCP 7 XML 剪辑时间线，兼容 **Adobe Premiere Pro** 与 **DaVinci Resolve**。
3. **`take 1_<tag>_edl.fcpxml`**：专用于 **Final Cut Pro X** 的 Apple FCPXML 剪辑时间线。
4. **`take 1_<tag>_edl.json`**：包含保留语句、讲者语速 CPS、时间戳的结构化剪辑决策 JSON。
5. **`take 1_<tag>_edl.csv`**：可用于表格查看的剪辑片段对照表，含视觉与声学验证备注。
6. **`take 1_whisper_sentences.json`**：包含词级时间戳的完整语音转录分析记录。
7. **`usage_log.jsonl`**（位于输出目录）：每次调用 Gemini API 的详细记录，追踪 Token 消耗量与调用耗时，便于用量统计。

---

## 导入非线性编辑软件（NLE）

- **DaVinci Resolve**：
  1. 在媒体池（Media Pool）右键 ➜ `时间线` ➜ `导入` ➜ `AAF / EDL / XML...`（快捷键 `Ctrl+Shift+I` / `Cmd+Shift+I`）。
  2. 选择生成的 `_edl.xml`，即可瞬间建立链接回原始视频的精准粗剪时间线。
- **Adobe Premiere Pro**：
  1. 依次点击菜单 `文件` ➜ `导入...`（快捷键 `Cmd+I` / `Ctrl+I`）。
  2. 选择 `_edl.xml`，在项目面板中双击生成的序列即可直接编辑。
- **Final Cut Pro X**：
  1. 依次点击菜单 `文件` ➜ `导入` ➜ `XML...`。
  2. 选择生成的 `_edl.fcpxml` 即可直接载入。

---

## 开发与测试

运行离线单元测试（使用合成音频与固定测试夹具，无需连接真实视频或 API）：

```bash
pip install -e ".[dev]"
pytest tests/
# 或
python3 -m unittest discover tests
```

---

## 许可证

[MIT License](LICENSE) © 2026 sylphlin

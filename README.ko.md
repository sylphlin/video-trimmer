# Video Trimmer (`video-trimmer`)

> **AI 기반 스마트 비디오 러프컷 및 트리밍 엔진**
> 
> 토킹헤드 영상, 비디오 팟캐스트, 튜토리얼 강의, 발표 영상 편집을 위해 설계된 엔드투엔드 스마트 비디오 러프컷 도구입니다.
> **Gemini 3.8 Flash 멀티모달 네이티브 비디오 이해**, **Whisper 단어 단위 음향 타임스탬프 동기화** (Apple Silicon Metal GPU 가속 `mlx-whisper` 지원), 및 **음향 온셋 자동 스냅 (Acoustic Onset Snapping / Smart Gap Shortening)** 기술을 통합했습니다.

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

[English (en)](README.md) | [繁體中文 (zh-TW)](README.zh-TW.md) | [简体中文 (zh-CN)](README.zh-CN.md) | [日本語 (ja)](README.ja.md) | [한국어 (ko)](README.ko.md)

---

## 주요 기능 및 기술 혁신

1. **멀티모달 네이티브 비디오 이해 (Gemini 3.8 Flash)**:
   - 단순한 음성 인식(ASR) 텍스트 매칭에만 의존하지 않습니다.
   - Gemini Files API / Interactions API를 통해 원본 비디오를 직접 업로드하여 **발표자의 시선 처리, 표정, 말더듬, 재촬영(Take)** 여부를 동시에 종합 판단합니다.
2. **지능형 '마지막 테이크 우선 (Last Take Wins)' 알고리즘**:
   - 발표자의 실수, 대사 연습, 동일 구간의 반복 촬영을 자동으로 감지하여 가장 자연스럽고 완성도 높은 마지막 테이크만 정확히 추출합니다.
3. **멀티모달 활성 화자 분리 (Multimodal Active Speaker Diarization)**:
   - 영상 시각 단서(시선, 입모양 싱크, 제스처)와 오디오 음향 특성(핀마이크 근접음 vs 공간 잔향)을 연동하여 분석합니다.
   - 카메라 앞의 메인 발표자와 카메라 밖 연출진의 큐 사인(예: "Action", "CTA-S2", "CDA84")이나 테이크 간 잡담을 명확하게 분리하여 무거운 로컬 화자 분리 모델 없이도 정확한 화자 구분을 제공합니다.
4. **음향 온셋 자동 스냅 (Acoustic Onset Snapping / Smart Gap Shortening)**:
   - Whisper 타임스탬프는 화자가 실제 발성하기 0.3초~0.8초 전에 시작되는 경향이 있습니다. `video-trimmer`는 파형 에너지를 동적으로 스캔하여 컷 시작점을 **성대 진동 시작 80ms 전**에 정밀하게 스냅하여 발화 전의 불필요한 침묵을 제거합니다.
5. **적응형 배경 노이즈 및 어미 감쇠 트래킹**:
   - 주변 배경 소음을 동적으로 분석하여 미세한 비음이나 흐려지는 말끝 어미가 잘려 나가지 않도록 보호합니다.
6. **15ms 오디오 등파워 마이크로 크로스페이드 (Audio Equal-Power Micro-Crossfade)**:
   - FFmpeg 렌더링 시 모든 컷 경계에 15ms 마이크로 페이드(`afade`)를 자동 적용하여 점프컷 사이의 디지털 팝 노이즈 및 배경 소음 단차를 완벽하게 차단합니다.
7. **촬영 대본 기반 정렬 (`--script`)**:
   - 촬영 대본이나 스크립트 텍스트(`.md` / `.txt`)를 지정하여 챕터 및 구성에 맞춘 정밀 컷 편집을 지원합니다.
8. **주요 NLE 프로젝트 원클릭 내보내기**:
   - 업계 표준 **FCP 7 XML** (Adobe Premiere Pro 및 DaVinci Resolve 지원) 및 **FCPXML** (Final Cut Pro X 지원)을 생성하며, 고화질 **MP4** 직접 렌더링도 지원합니다.

---

## 디렉터리 구조

본 프로젝트는 [Agent Plugins 1.0 Specification](https://agent-plugins.org/) 및 [Agent Skills Specification](https://agentskills.io/specification)을 준수합니다:

```text
video-trimmer/
├── plugin.json                 # Agent Plugins 1.0 규격 매니페스트
├── rules/
│   └── AGENTS.md               # 외부 AI 클라이언트 실행 불변 규칙 (엄격한 읽기 전용 및 Fail-Fast)
├── AGENTS.md                   # 프로젝트 유지보수 및 개발 운영 지침 (ASD-STE100 영어 기준)
├── SKILL.md                    # Agent Skill 규격 정의서 및 가이드
├── README.md                   # 공개 문서 (영어)
├── README.zh-TW.md             # 공개 문서 (번체 중국어)
├── README.zh-CN.md             # 공개 문서 (간체 중국어)
├── README.ja.md                # 공개 문서 (일본어)
├── README.ko.md                # 공개 문서 (한국어)
├── LICENSE                     # MIT 라이선스
├── .env.example                # Vertex AI 및 GCS 환경 변수 템플릿
├── setup.sh                    # 100% 네이티브 gcloud GCP 리소스 프로비저닝 스크립트 (Terraform 불필요)
├── pyproject.toml              # PEP 621 Python 패키징 및 CLI 콘솔 스크립트 설정
├── requirements.txt            # Python 런타임 종속성
├── video_trimmer.py            # 메인 CLI 명령 포워더
├── auto_rough_cut.py           # 하위 호환성 래퍼
├── scripts/                    # 코어 엔진 모듈
│   ├── __init__.py
│   ├── video_trimmer.py        # CLI 인자 분석 및 파이프라인 오케스트레이션
│   ├── constants.py            # 중앙 집중식 명명 상수
│   ├── exceptions.py           # 사용자 정의 예외 계층
│   ├── acoustic.py             # CPS, 동적 마진, 음향 에너지 감지
│   ├── transcribe.py           # Whisper 음성 전사, 문장 결합, 클립 동기화
│   ├── gemini_client.py        # Vertex AI (ADC) 클라이언트 및 멀티모달 추론
│   ├── gcs_utils.py            # GCS 임시 업로드 및 정리
│   ├── exporters.py            # FCP7 XML / FCPXML / CSV 내보내기
│   └── render.py               # ffprobe 검증 및 ffmpeg 렌더링
├── tests/                      # 오프라인 단위 테스트
├── prompts/
│   └── video_cut_prompt.md     # 멀티모달 컷 편집 프롬프트 사양
└── examples/                   # 샘플 결과물 파일 (EDL, XML, FCPXML, JSON)
```

---

## 빠른 시작

### 1. 시스템 요구 사항

시스템에 [FFmpeg](https://ffmpeg.org/)가 설치되어 있고 `PATH`에 등록되어 있는지 확인하세요:

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg
```

### 2. 설치 및 배포

#### 방법 A: Google Antigravity 및 Agent Plugins 1.0 설치 (AI 에이전트 권장)

Google Antigravity 또는 [Agent Plugins 1.0](https://agent-plugins.org/) 호환 AI 클라이언트에 직접 설치합니다:

1. **Agent Plugin으로 설치 (권장: `plugin.json` 및 `rules/AGENTS.md` 읽기 전용 보호 자동 로드)**:
   - **전역 플러그인 (Global Plugin)** (모든 프로젝트 및 워크스페이스 공용, 권장):
     ```bash
     git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/plugins/video-trimmer
     ```
   - **워크스페이스 플러그인 (Workspace Plugin)** (현재 워크스페이스 전용):
     ```bash
     git clone https://github.com/sylphlin/video-trimmer.git .agents/plugins/video-trimmer
     ```

2. **또는 Agent Skill로 설치**:
   - **전역 스킬 (Global Skill)**:
     ```bash
     git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/skills/video-trimmer
     ```
   - **워크스페이스 스킬 (Workspace Skill)**:
     ```bash
     git clone https://github.com/sylphlin/video-trimmer.git .agent/skills/video-trimmer
     ```

#### 방법 B: 독립형 Python CLI 설치

저장소를 클론하고 로컬 환경에 종속성을 설치합니다:

```bash
git clone https://github.com/sylphlin/video-trimmer.git
cd video-trimmer

# 코어 종속성 설치
pip install -r requirements.txt

# (macOS Apple Silicon 권장) Metal GPU 하드웨어 가속용 mlx-whisper 설치
pip install mlx-whisper

# 또는 편집 가능 모드로 CLI 도구 설치
pip install -e .
```

### 3. Google Cloud 네이티브 환경 설정 (100% Native gcloud, Terraform 불필요)

본 도구는 비디오 임시 스테이징을 위해 **Google Cloud Storage (GCS)**를 사용하고, 멀티모달 추론에 **Google Cloud Vertex AI** (**Application Default Credentials: ADC**)를 사용합니다.

필요한 모든 클라우드 리소스(GCS 버킷, CORS 설정, 2일 임시 파일 자동 정리 수명 주기, 전용 서비스 계정, 최소 권한 IAM 역할)는 순수 `gcloud` 명령어로 프로비저닝됩니다 (**Terraform 불필요, Cloud Shell 지원**).

#### 옵션 A: 원클릭 자동 프로비저닝 (권장)

포함된 자동 환경 구성 스크립트를 실행합니다:

```bash
# 실행 권한 부여 (최초 1회)
chmod +x setup.sh

# 자동 프로비저닝 (기존 gcloud 설정을 읽어 .env 자동 생성):
./setup.sh

# 또는 GCP 프로젝트 및 리전을 명시적으로 지정:
./setup.sh --project YOUR_PROJECT_ID --region us-central1

# 드라이런 확인 (리소스 변경 없이 명령어 미리보기):
./setup.sh --dry-run
```

이 스크립트는 다음 작업을 자동으로 처리합니다:
1. GCS 버킷 `gs://video-preprocessing-${PROJECT_ID}` 생성 및 확인 (`--uniform-bucket-level-access` 및 `--public-access-prevention` 적용).
2. Signed URL 스트리밍 재생을 위한 24시간 캐시 CORS 설정 (`GET`, `HEAD` 허용).
3. `raw/` 경로에 대한 **2일 자동 삭제 수명 주기 규칙** 설정 (스토리지 비용 누적 방지).
4. 전용 서비스 계정 `video-trimmer-sa` 생성 및 최소 권한 부여 (`roles/storage.objectUser`, `roles/aiplatform.user`, `roles/logging.logWriter`).
5. 로컬 `.env` 환경 설정 파일 자동 생성 및 업데이트.

#### 옵션 B: 수동 네이티브 gcloud 명령어 설정

터미널에서 직접 설정하려는 경우:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export BUCKET_NAME="video-preprocessing-${PROJECT_ID}"
export SA="video-trimmer-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# 1. GCS 임시 버킷 생성
gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access \
    --public-access-prevention

# 2. 임시 비디오 2일 자동 정리 수명 주기 구성
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

# 3. 전용 서비스 계정 생성 및 최소 권한 IAM 역할 바인딩
gcloud iam service-accounts create video-trimmer-sa \
    --display-name="Video Trimmer Service Account" \
    --project="${PROJECT_ID}"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA}" \
    --role="roles/aiplatform.user"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
    --member="serviceAccount:${SA}" \
    --role="roles/storage.objectUser"

# 4. 로컬 ADC 자격 증명 로그인
gcloud auth application-default login

# 5. 로컬 .env 구성
cp .env.example .env
# .env에 GOOGLE_CLOUD_PROJECT, VIDEO_TRIMMER_BUCKET 등 설정
```

---

## 사용 방법

촬영된 원본 비디오에 러프컷 파이프라인을 실행합니다:

```bash
# 설치된 CLI 명령어로 실행:
video-trimmer -i "/path/to/raw_footage.mp4"

# 또는 Python으로 직접 실행:
python video_trimmer.py -i "/path/to/raw_footage.mp4"
```

### 고급 예제

```bash
# 1. 촬영 대본을 지정하여 구성 정렬 유도
video-trimmer -i "take 1.mp4" --script "script.md"

# 2. Agentic 비디오 이해 모드 활성화 (동적 멀티턴 프레임 탐색)
video-trimmer -i "interview.mp4" --agentic

# 3. 빠른 템포의 유튜브 테크/과학 설명 영상용 컴팩트 페이싱 적용
video-trimmer -i "news.mp4" --pacing compact --suffix "fast"

# 4. 캐시된 EDL JSON으로 즉시 재렌더링 (로컬에서 몇 초 만에 완료)
video-trimmer -i "take 1.mp4" --cached-json "take 1_agentic_edl.json" --suffix "fine_tuned"
```

---

## CLI 옵션 레퍼런스

| 옵션 | 단축형 | 기본값 | 설명 |
| :--- | :---: | :---: | :--- |
| `--input` | `-i` | *(필수)* | 입력 원본 비디오 파일 경로 (`.mp4`, `.mov`). |
| `--output-dir` | `-o` | 비디오와 동일 경로 | 모든 결과 파일이 저장될 디렉터리 경로. |
| `--model` | `-m` | `gemini-3.8-flash` | Gemini 모델 식별자 (기본값: `$MODEL_NAME`). |
| `--project` | | `None` | Google Cloud 프로젝트 ID (기본값: `$GOOGLE_CLOUD_PROJECT` 또는 ADC). |
| `--region` | | `None` | Vertex AI 리전 (기본값: `$GOOGLE_CLOUD_LOCATION` 또는 `global`). |
| `--bucket` | | `None` | 비디오 스테이징용 GCS 버킷 (기본값: `$VIDEO_TRIMMER_BUCKET`). |
| `--keep-gcs-upload` | | `False` | 추론 완료 후 GCS에 업로드된 임시 비디오를 삭제하지 않고 유지. |
| `--script` | `-s` | `None` | 촬영 대본 또는 스크립트 텍스트 파일 경로 (`.md` / `.txt`). |
| `--agentic` | | `False` | 동적 프레임 탐색을 수행하는 Agentic 비디오 이해 모드 활성화. |
| `--pacing` | `-p` | `auto` | 호흡 및 페이싱 전략: `auto` (동적 CPS), `compact` (밀착), `breathing` (호흡 여유). |
| `--cached-json`| | `None` | 기존 EDL JSON을 지정하여 클라우드 추론을 건너뛰고 로컬 렌더링만 수행. |
| `--suffix` | | `None` | 생성되는 파일명에 추가할 사용자 정의 접미사 태그. |
| `--crf` | | `18` | FFmpeg H.264 렌더링 CRF 화질 값 (18 = 시각적 무손실). |
| `--skip-whisper`| | `False` | 로컬 Whisper 전사 건너뛰기 (음향 에너지 폴백 사용). |
| `--verbose` | | `False` | 상세 DEBUG 레벨 로그 출력 (기본값: INFO). |

---

## 생성되는 결과 파일

입력 파일 `take 1.mp4`에 대해 `video-trimmer`는 다음 파일을 생성합니다:

1. **`take 1_<tag>_trimmed.mp4`**: 등파워 마이크로 페이드가 적용된 최종 결합 렌더링 비디오.
2. **`take 1_<tag>_edl.xml`**: **Adobe Premiere Pro** 및 **DaVinci Resolve**와 호환되는 표준 FCP 7 XML 타임라인.
3. **`take 1_<tag>_edl.fcpxml`**: **Final Cut Pro X** 전용 Apple FCPXML 타임라인.
4. **`take 1_<tag>_edl.json`**: 선택된 문장, 화자 발화 속도 CPS, 타임스탬프를 포함하는 구조화된 편집 결정 JSON.
5. **`take 1_<tag>_edl.csv`**: 스프레드시트 검토용 컷 목록 (시각 및 음향 검증 메모 포함).
6. **`take 1_whisper_sentences.json`**: 단어 단위 타임스탬프를 포함하는 전체 음성 전사 분석 데이터.
7. **`usage_log.jsonl`** (출력 디렉터리 내): Gemini API 호출별 토큰 사용량 및 소요 시간 로그.

---

## NLE 소프트웨어로 가져오기

- **DaVinci Resolve**:
  1. 미디어 풀(Media Pool) 우클릭 ➜ `타임라인` ➜ `가져오기` ➜ `AAF / EDL / XML...` (단축키: `Ctrl+Shift+I` / `Cmd+Shift+I`).
  2. 생성된 `_edl.xml`을 선택하면 원본 비디오에 연결된 러프컷 타임라인이 즉시 생성됩니다.
- **Adobe Premiere Pro**:
  1. `파일` ➜ `가져오기...` (단축키: `Cmd+I` / `Ctrl+I`).
  2. `_edl.xml`을 선택하고, 프로젝트 패널에 추가된 시퀀스를 더블 클릭하여 즉시 편집합니다.
- **Final Cut Pro X**:
  1. `파일` ➜ `가져오기` ➜ `XML...`.
  2. 생성된 `_edl.fcpxml`을 선택하여 가져옵니다.

---

## 개발 및 테스트

오프라인 단위 테스트 실행 (합성 오디오 및 고정 테스트 픽스처 사용, 실제 비디오 및 API 호출 불필요):

```bash
pip install -e ".[dev]"
pytest tests/
# 또는
python3 -m unittest discover tests
```

---

## 라이선스

[MIT License](LICENSE) © 2026 sylphlin

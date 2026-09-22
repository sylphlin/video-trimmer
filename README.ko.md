# Video Trimmer (`video-trimmer`)

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sylphlin/video-trimmer/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Apple Silicon Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-orange.svg)]()
[![FFmpeg](https://img.shields.io/badge/FFmpeg-5.0+-red.svg)](https://ffmpeg.org/)

[English (en)](README.md) | [繁體中文 (zh-TW)](README.zh-TW.md) | [简体中文 (zh-CN)](README.zh-CN.md) | [日本語 (ja)](README.ja.md) | [한국어 (ko)](README.ko.md)

---

## 개요 (Overview)

**Video Trimmer**는 토킹헤드 비디오, 튜토리얼 및 발표 녹화물을 위한 AI 자동 러프컷 및 트리밍 엔진입니다. **Google Vertex AI Gemini 3.8 Flash** 멀티모달 비디오 추론, **Whisper 단어 단위 음향 타임스탬프**(`mlx-whisper`) 및 **음향 온셋 스내핑(Acoustic Onset Snapping)**을 결합하여 NG 테이크, 말더듬, 무음 구간을 제거하고 NLE 타임라인(`.xml`, `.fcpxml`, `.edl`, `.csv`)과 렌더링된 MP4 비디오를 생성합니다.

---

## 핵심 기능

1. **멀티모달 비디오 추론 (`gemini-3.8-flash`)**: 발표자의 시선, 표정, 말실수 및 재촬영 구간을 동시에 평가합니다.
2. **Last-Take-Wins (최종 성공 테이크 선택)**: 동일 대사의 반복 촬영을 감지하여 마지막 성공 테이크만 유지합니다.
3. **음향 온셋 스내핑 (`tighten_clip_to_speech`)**: 성대 진동 80 ms 전에 컷 포인트를 배치하여 첫 음소를 자르지 않고 무음을 제거합니다.
4. **15 ms 등전력 오디오 마이크로 크로스페이드**: 모든 편집 경계에 15 ms 마이크로 페이드(`afade=t=in:d=0.015:curve=iqsin` 및 `afade=t=out:d=0.015:curve=oqsin`)를 적용하여 팝 노이즈를 방지합니다.
5. **멀티 NLE 타임라인 지원**: **FCP7 XML**(Premiere Pro / DaVinci Resolve), **FCPXML**(Final Cut Pro), **CMX 3600 EDL** 및 **CSV**를 내보냅니다.

---

## 설치 및 Google Cloud 설정

```bash
# 1. FFmpeg 및 Python 패키지 설치
brew install ffmpeg
git clone https://github.com/sylphlin/video-trimmer.git ~/.gemini/config/plugins/video-trimmer
pip install -r requirements.txt
pip install mlx-whisper

# 2. ADC 인증 및 setup.sh 실행
gcloud auth application-default login
chmod +x setup.sh
./setup.sh --project YOUR_GCP_PROJECT_ID --region us-central1
```

---

## 명령줄 사용법 (CLI Usage)

```bash
# 표준 러프컷 실행 (Agentic 비디오 이해 모드 기본 사용)
python3 video_trimmer.py -i "raw_footage.mp4" --agentic

# 촬영 대본 정렬
python3 video_trimmer.py -i "raw_footage.mp4" --script "shooting_script.md" --agentic

# 정적 멀티모달 모드 (static 모드 사용 시에만 --agentic 생략)
python3 video_trimmer.py -i "raw_footage.mp4"

# Google Drive 공유 링크에서 직접 러프컷 실행
python3 video_trimmer.py -i "https://drive.google.com/file/d/FILE_ID/view?usp=sharing" --agentic -o output/
```

---

## Google Drive 연동 및 GCS 2단계 수명 주기 정책

| GCS 경로 접두사 (`matchesPrefix`) | 저장 객체 | 보관 기간 (`age`) | 목적 |
| :--- | :--- | :--- | :--- |
| **`raw/`** | 스테이징된 원본 비디오 (`raw/<filename>.mp4`) | **2일 (`age: 2`)** | 추론 완료 후 즉시 삭제되며 안전 백업으로 2일 후 자동 삭제됩니다. |
| **`output/`**, **`deliverables/`**, **`trimmed/`** | 트리밍된 비디오, XML/FCPXML 타임라인 | **15일 (`age: 15`)** | 팀 검토를 위해 15일간 보관한 후 자동 삭제합니다. |

---

## 라이선스 (License)

이 프로젝트는 [MIT License](LICENSE)에 따라 라이선스가 부여됩니다.

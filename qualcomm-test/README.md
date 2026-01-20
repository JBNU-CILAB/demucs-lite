# Qualcomm AI Hub Demucs Test

Qualcomm AI Hub를 사용하여 Demucs ONNX 모델을 실제 모바일 NPU에서 테스트하는 스크립트 모음입니다.

## 개요

이 디렉토리는 Demucs 모델을 Qualcomm NPU (Samsung Galaxy S24)에서 실행하기 위한 전체 파이프라인을 제공합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                       전체 파이프라인                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   MP3/WAV 파일                                                  │
│       │                                                         │
│       ▼ preprocess_demucs.py (선택적)                           │
│   input.raw + x.raw                                             │
│       │                                                         │
│       ▼ run_qai_hub_*.py                                        │
│   ┌─────────────────────────────────────┐                       │
│   │ 1. ONNX → QNN DLC 컴파일             │                       │
│   │ 2. Qualcomm NPU에서 추론             │                       │
│   │ 3. 결과 다운로드                      │                       │
│   └─────────────────────────────────────┘                       │
│       │                                                         │
│       ▼ output/add_67.raw                                       │
│       │                                                         │
│       ▼ postprocess_demucs.py                                   │
│   stems/                                                        │
│   ├── drums.wav                                                 │
│   ├── bass.wav                                                  │
│   ├── other.wav                                                 │
│   └── vocals.wav                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 파일 설명

### 추론 스크립트

| 파일 | 설명 | 양자화 | 속도 |
|------|------|--------|------|
| `run_qai_hub.py` | 단일 청크 테스트용 | FP16 | 기본 |
| `run_qai_hub_chunked.py` | 순차 처리 (1개씩) | FP16 | 느림 |
| `run_qai_hub_batch.py` | 배치 처리 (50개씩) | FP16 | 빠름 |
| `run_qai_hub_batch_int8.py` | **INT8 PTQ 배치 처리** | INT8 | 빠름 |

### 전처리/후처리 스크립트

| 파일 | 설명 |
|------|------|
| `preprocess_demucs.py` | 오디오 → `input.raw` + `x.raw` 변환 |
| `postprocess_demucs.py` | `add_67.raw` → 4개 WAV 파일 변환 |

## 사용법

### 방법 1: FP16 배치 추론

전체 곡을 FP16으로 처리합니다.

```bash
# 1. 배치 추론 실행
python run_qai_hub_batch.py ../day6-happy.mp3

# 2. 결과를 WAV로 변환
python postprocess_demucs.py output stems
```

### 방법 2: INT8 PTQ 배치 추론

INT8 양자화로 더 가볍고 빠르게 처리합니다.

```bash
# 1. INT8 PTQ 배치 추론 실행
python run_qai_hub_batch_int8.py ../day6-happy.mp3

# 2. 결과를 WAV로 변환
python postprocess_demucs.py output stems_int8 --raw-file add_67_int8.raw
```

**INT8 PTQ 동작 원리:**
1. 오디오에서 calibration 샘플 50개 추출
2. FP16 모델 + calibration 데이터로 컴파일 시 INT8 양자화 자동 적용
3. INT8 양자화된 모델로 추론

### 방법 3: 순차 추론

각 청크를 순차적으로 처리합니다. 느리지만 안정적입니다.

```bash
python run_qai_hub_chunked.py ../day6-happy.mp3
python postprocess_demucs.py output stems
```

### 방법 4: 단일 청크 테스트

모델 동작 확인용입니다.

```bash
# 1. 전처리
python preprocess_demucs.py ../day6-happy.mp3

# 2. 단일 추론
python run_qai_hub.py

# 3. 후처리
python postprocess_demucs.py output stems
```

## 핵심 동작 원리

### 1. 오디오 청킹

긴 오디오를 1초 단위로 분할하여 NPU 메모리 제한을 우회합니다.

```
오디오 (190초)
    │
    ▼ chunk_audio()
    │
253개 청크 (1초씩, 25% 오버랩)

┌────────────────────────────────────────────┐
│  |----1초----|                             │  Chunk 0
│        |----1초----|                       │  Chunk 1
│              |----1초----|                 │  Chunk 2
│                    ...                     │
│  └─ 0.25초 오버랩 ─┘                       │
└────────────────────────────────────────────┘
```

### 2. 듀얼 입력 구조

Demucs 모델은 두 가지 입력을 동시에 받습니다:

| 입력 | Shape | 설명 |
|------|-------|------|
| `input` | (1, 2, 44100) | 시간 도메인 (waveform) |
| `x` | (1, 4, 2048, 44) | 주파수 도메인 (STFT) |

```
chunk_audio (1초)
     │
     ├──▶ input: 스테레오 waveform
     │
     └──▶ compute_stft_for_chunk()
              │
              ▼
          x: [L_real, L_imag, R_real, R_imag]
```

### 3. 배치 처리

50개씩 묶어서 병렬로 Qualcomm AI Hub에 제출합니다.

```
253개 청크
    │
    ▼ 50개씩 분할
    │
Batch 1: 1-50   ──▶ 제출 ──▶ 결과 수집
Batch 2: 51-100 ──▶ 제출 ──▶ 결과 수집
Batch 3: 101-150 ──▶ 제출 ──▶ 결과 수집
...
```

### 4. Overlap-Add 병합

오버랩 구간은 크로스페이드로 부드럽게 연결합니다.

```
     ████████████░░░░                   Chunk 0
          ░░░░████████████░░░░          Chunk 1
               ░░░░████████████░░░░     Chunk 2
     ─────────────────────────────────  최종 출력
     ▲ fade-in   ▲ crossfade   ▲ fade-out
```

### 5. 출력 구조

| 출력 | Shape | 설명 |
|------|-------|------|
| `add_67.raw` | (1, 4, 2, N) | 4개 stem × 스테레오 |

- Stem 0: drums
- Stem 1: bass
- Stem 2: other
- Stem 3: vocals

## 모델 설정

```python
# 모델 경로
MODEL_PATH = "htdemucs_model/htdemucs_seg1.0s_fp16.onnx"

# 디바이스
DEVICE_NAME = "Samsung Galaxy S24 (Family)"

# 오디오 설정
SAMPLE_RATE = 44100
CHUNK_SIZE = 44100
OVERLAP = 11025
BATCH_SIZE = 50
```

## INT8 양자화 기술 노트

### FP16 vs INT8 양자화의 차이

**FP16 변환**은 단순히 숫자 포맷만 바꾸는 것입니다:
- 모델 구조 변경 없음
- QNN이 그대로 해석 가능

**INT8 양자화**는 연산 자체가 바뀝니다:
- 각 레이어에 scale/zero_point 파라미터 추가
- QuantizeLinear/DequantizeLinear 노드 삽입

### 왜 Qualcomm PTQ를 사용해야 하는가?

| 방법 | 모델 구조 | QNN 호환 |
|-----|---------|---------|
| ONNX Runtime `quantize_dynamic` | QDQ 노드 추가 | X |
| Qualcomm AI Hub PTQ | QNN 내부 처리 | O |

ONNX Runtime의 INT8 양자화는 ONNX Runtime용으로 최적화되어 있어 QNN 컴파일러가 인식하지 못합니다. 따라서 Qualcomm에서는 **컴파일 시 calibration 데이터로 PTQ**를 수행해야 합니다.

```python
hub.submit_compile_job(
    model=fp16_model,
    calibration_data=calibration_data,
    options="--quantize_full_type int8"
)
```

## 요구 사항

```bash
pip install qai-hub librosa numpy soundfile
```

다음의 과정을 통해 Qualcomm AI Hub를 설정합니다.
```
pip3 install qai-hub
qai-hub configure --api_token (API_TOKEN)
qai-hub list-devices
```

- 지원 디바이스: Samsung Galaxy S24 등 Qualcomm Snapdragon 탑재 기기
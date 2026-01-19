# Demucs Model Conversion

[Demucs v4 Hybrid Transformer](https://github.com/facebookresearch/demucs) 모델을 모바일 기기에서 실행하기 위한 경량화 및 최적화 프로젝트입니다.

## 프로젝트 개요

[sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx) 프로젝트를 기반으로, 실제 모바일 NPU(Qualcomm)에서 동작할 수 있도록 모델을 최적화했습니다.

### 주요 작업 내용

1. **FP16 양자화**: 모델을 Float32 → Float16으로 변환하여 메모리 사용량 절반 감소
2. **INT8 PTQ 양자화**: 컴파일 시 calibration 데이터로 INT8 양자화 적용 (추가 경량화)
3. **청킹 처리**: NPU 메모리 제한을 우회하기 위해 오디오를 1초 단위로 분할 처리
4. **배치 추론**: Qualcomm AI Hub에서 병렬 처리로 추론 시간 단축
5. **Qualcomm NPU 테스트**: Samsung Galaxy S24에서 실제 추론 테스트 수행

## Qualcomm AI Hub 테스트 결과

### 테스트 환경
- **모델**: HTDemucs (FP16)
- **디바이스**: Samsung Galaxy S24
- **테스트 곡**: 데이식스 - Happy (약 3분)

### 메모리 문제 해결 과정

| 시도 | 청크 크기 | 결과 |
|------|----------|------|
| 전체 곡 처리 | - | ❌ OOM (메모리 부족) |
| 2초 청킹 | 88,200 samples | ❌ OOM |
| 1초 청킹 | 44,100 samples | ✅ 성공 |

### 추론 시간 분석

**Qualcomm AI Hub 테스트 (디바이스 프로비저닝 포함)**
- 청크당 처리 시간: 약 4분 (프로비저닝 대기 시간 포함)
- 3분 곡 (253개 청크): 약 28시간 예상
- 병목: 클라우드 테스트 환경의 디바이스 프로비저닝 대기

**실제 기기 추론 시간 (프로비저닝 제외)**
- 청크당 순수 추론 시간: **약 100ms**
- 3분 곡 예상 처리 시간: **약 30~35초**
- 5분 곡 예상 처리 시간: **약 50~60초**

## 디렉토리 구조

```
demucs.onnx/
├── scripts/                    # 모델 변환 스크립트
│   ├── convert-pth-to-onnx.py          # PyTorch → ONNX 변환
│   ├── convert-pth-to-onnx-chunked.py  # 청크용 ONNX 변환
│   └── convert_to_fp16.py              # FP16 양자화
├── qualcomm-test/              # Qualcomm AI Hub 테스트
│   ├── run_qai_hub.py                  # 단일 청크 테스트
│   ├── run_qai_hub_chunked.py          # 순차 처리
│   ├── run_qai_hub_batch.py            # FP16 배치 처리
│   ├── run_qai_hub_batch_int8.py       # INT8 PTQ 배치 처리 (권장)
│   ├── preprocess_demucs.py            # 전처리 (STFT)
│   └── postprocess_demucs.py           # 후처리 (WAV 변환, ISTFT)
├── src/                        # C++ 구현
│   └── dsp.cpp                         # STFT/iSTFT 구현
├── demucs-for-onnx/            # 수정된 Demucs (STFT 분리)
└── onnx-models/                # 변환된 ONNX 모델
```

## 사용 방법

### 1. 모델 변환

```bash
# PyTorch → ONNX 변환
python scripts/convert-pth-to-onnx-chunked.py

# FP16 양자화
python scripts/convert_to_fp16.py onnx-models/htdemucs.onnx onnx-models/htdemucs_fp16.onnx
```

### 2. Qualcomm AI Hub 테스트

```bash
cd qualcomm-test

# FP16 배치 추론
python run_qai_hub_batch.py ../day6-happy.mp3
python postprocess_demucs.py output stems

# INT8 PTQ 배치 추론 (권장 - 더 가볍고 빠름)
python run_qai_hub_batch_int8.py ../day6-happy.mp3
python postprocess_demucs.py output stems_int8 --raw-file add_67_int8.raw
```

### 3. Qualcomm AI Hub 설정

```bash
pip install qai-hub
qai-hub configure --api_token YOUR_API_TOKEN
qai-hub list-devices
```

## 기술적 세부사항

### STFT/iSTFT 분리

원본 Demucs 모델은 STFT/iSTFT를 내부에서 수행하지만, 이 연산은 ONNX로 내보낼 수 없습니다. 따라서 STFT/iSTFT를 모델 외부로 분리했습니다.

```
오디오 → [STFT (외부)] → ONNX 모델 → [iSTFT (외부)] → 분리된 stems
```

### 청킹 파라미터

```python
CHUNK_SIZE = 44100   # 1초 (44.1kHz)
OVERLAP = 11025      # 25% 오버랩
HOP_SIZE = 33075     # 청크 간 이동 거리
```

### 모델 입출력

| 입력 | Shape | 설명 |
|------|-------|------|
| input | (1, 2, 44100) | 시간 도메인 스테레오 오디오 |
| x | (1, 4, 2048, 44) | STFT 결과 (Real/Imag 분리) |

| 출력 | Shape | 설명 |
|------|-------|------|
| add_67 | (1, 4, 2, 44100) | 4개 stem × 스테레오 |

### INT8 양자화 (PTQ)

**FP16 변환**은 숫자 포맷만 바꾸므로 QNN 호환이 됩니다. 하지만 **INT8 양자화**는 각 런타임마다 포맷이 다릅니다:

| 방법 | QNN 호환 |
|-----|---------|
| ONNX Runtime `quantize_dynamic` | ❌ (QDQ 노드가 QNN에서 인식 안 됨) |
| Qualcomm AI Hub PTQ | ✅ |

따라서 Qualcomm에서는 **컴파일 시 calibration 데이터로 PTQ**를 수행합니다:

```python
hub.submit_compile_job(
    model=fp16_model,
    calibration_data=calibration_data,
    options="--quantize_full_type int8"
)
```

## 결론

- FP16 양자화만으로는 모바일 NPU 메모리 한계를 극복할 수 없었음
- 1초 단위 청킹으로 메모리 문제 해결
- **INT8 PTQ**: 컴파일 시 calibration 데이터로 INT8 양자화 적용 가능
- 실제 기기에서 **청크당 약 100ms** 추론 가능
- **3분 곡 기준 약 30~35초** 처리 예상

## Credits

Based on [sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx) (MIT License)

"""
Demucs ONNX 모델 출력 후처리 스크립트

추론 결과를 WAV 파일로 변환하여 검증:
- 4개의 stem (drums, bass, other, vocals) 분리
- output (주파수 도메인) 또는 add_67 (시간 도메인) 처리
"""

import librosa
import numpy as np
import os
import sys
import soundfile as sf

# Demucs 모델 상수
SAMPLE_RATE = 44100
TIME_BRANCH_LEN = 343980
FREQ_BRANCH_LEN = 336
FREQ_BINS = 2048
NB_SOURCES = 4
FFT_WINDOW_SIZE = 4096
FFT_HOP_SIZE = 1024

# Stem 이름
STEM_NAMES = ["drums", "bass", "other", "vocals"]


def load_time_domain_output(output_path: str) -> np.ndarray:
    """
    시간 도메인 출력 로드 (add_67)
    Shape: (1, 4, 2, time_samples) -> (4, 2, time_samples)
    
    배치 처리된 출력은 다양한 길이를 가질 수 있으므로 동적으로 처리
    """
    data = np.fromfile(output_path, dtype=np.float32)
    
    # 동적 길이 계산: total_size = 1 * 4 * 2 * time_samples
    # time_samples = total_size / 8
    if data.size % 8 != 0:
        raise ValueError(f"Invalid file size: {data.size} elements (must be divisible by 8)")
    
    time_samples = data.size // 8
    print(f"  Detected time samples: {time_samples} ({time_samples/SAMPLE_RATE:.2f}s)")
    
    output = data.reshape(1, NB_SOURCES, 2, time_samples)
    return output[0]  # Remove batch dimension: (4, 2, time_samples)


def load_freq_domain_output(output_path: str) -> np.ndarray:
    """
    주파수 도메인 출력 로드 (output)
    Shape: (1, 4, 4, 2048, 336) -> (4, 4, 2048, 336)
    """
    data = np.fromfile(output_path, dtype=np.float32)
    expected_size = 1 * NB_SOURCES * 4 * FREQ_BINS * FREQ_BRANCH_LEN
    
    if data.size != expected_size:
        raise ValueError(f"Expected {expected_size} elements, got {data.size}")
    
    output = data.reshape(1, NB_SOURCES, 4, FREQ_BINS, FREQ_BRANCH_LEN)
    return output[0]  # Remove batch dimension: (4, 4, 2048, 336)


def freq_to_time_domain(freq_output: np.ndarray) -> np.ndarray:
    """
    주파수 도메인 출력을 시간 도메인으로 변환 (iSTFT)
    
    입력: (4, 4, 2048, 336) - 4 stems, 4 channels (L_real, L_imag, R_real, R_imag)
    출력: (4, 2, time_samples) - 4 stems, stereo
    """
    time_outputs = []
    
    for stem_idx in range(NB_SOURCES):
        stem_freq = freq_output[stem_idx]  # (4, 2048, 336)
        
        # Real/Imag 채널을 complex로 결합
        L_complex = stem_freq[0] + 1j * stem_freq[1]  # (2048, 336)
        R_complex = stem_freq[2] + 1j * stem_freq[3]  # (2048, 336)
        
        # DC bin 추가 (0으로 패딩)
        L_full = np.zeros((FREQ_BINS + 1, FREQ_BRANCH_LEN), dtype=np.complex64)
        R_full = np.zeros((FREQ_BINS + 1, FREQ_BRANCH_LEN), dtype=np.complex64)
        L_full[1:2049, :] = L_complex
        R_full[1:2049, :] = R_complex
        
        # iSTFT 수행
        L_time = librosa.istft(
            L_full,
            hop_length=FFT_HOP_SIZE,
            win_length=FFT_WINDOW_SIZE,
            window='hann',
            center=True,
            length=TIME_BRANCH_LEN
        )
        R_time = librosa.istft(
            R_full,
            hop_length=FFT_HOP_SIZE,
            win_length=FFT_WINDOW_SIZE,
            window='hann',
            center=True,
            length=TIME_BRANCH_LEN
        )
        
        time_outputs.append(np.stack([L_time, R_time], axis=0))
    
    return np.stack(time_outputs, axis=0)  # (4, 2, time_samples)


def save_stems_as_wav(stems: np.ndarray, output_dir: str, original_length: int = None):
    """
    4개의 stem을 개별 WAV 파일로 저장
    
    입력: stems (4, 2, time_samples)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for stem_idx, stem_name in enumerate(STEM_NAMES):
        stem_audio = stems[stem_idx]  # (2, time_samples)
        
        # 원본 길이로 트림 (필요한 경우)
        if original_length is not None and stem_audio.shape[1] > original_length:
            stem_audio = stem_audio[:, :original_length]
        
        # WAV 파일 저장 (stereo)
        output_path = os.path.join(output_dir, f"{stem_name}.wav")
        sf.write(output_path, stem_audio.T, SAMPLE_RATE)
        print(f"  Saved: {output_path}")


def postprocess(output_dir: str = "output", wav_output_dir: str = "stems", raw_file: str = "add_67.raw"):
    """메인 후처리 함수"""
    print("="*60)
    print("Demucs Output Postprocessing")
    print("="*60)
    
    time_output_path = os.path.join(output_dir, raw_file)
    freq_output_path = os.path.join(output_dir, "output.raw")
    
    stems = None
    
    if os.path.exists(time_output_path):
        print(f"\nLoading time-domain output: {time_output_path}")
        try:
            stems = load_time_domain_output(time_output_path)
            print(f"  Loaded shape: {stems.shape}")
        except Exception as e:
            print(f"  Error: {e}")
    
    if stems is None and os.path.exists(freq_output_path):
        print(f"\nLoading frequency-domain output: {freq_output_path}")
        try:
            freq_output = load_freq_domain_output(freq_output_path)
            print(f"  Loaded shape: {freq_output.shape}")
            print("  Converting to time domain (iSTFT)...")
            stems = freq_to_time_domain(freq_output)
            print(f"  Converted shape: {stems.shape}")
        except Exception as e:
            print(f"  Error: {e}")
    
    if stems is None:
        print("\nError: No valid output files found!")
        print(f"  Expected: {time_output_path} or {freq_output_path}")
        sys.exit(1)
    
    # WAV 파일로 저장
    print(f"\nSaving stems to: {wav_output_dir}/")
    save_stems_as_wav(stems, wav_output_dir)
    
    print("\n" + "="*60)
    print("Postprocessing complete!")
    print("="*60)
    print(f"\nGenerated files:")
    for stem_name in STEM_NAMES:
        print(f"  - {wav_output_dir}/{stem_name}.wav")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Demucs output postprocessing")
    parser.add_argument("output_dir", nargs="?", default="output",
                        help="Directory containing raw output files")
    parser.add_argument("wav_output_dir", nargs="?", default="stems",
                        help="Directory to save WAV files")
    parser.add_argument("--raw-file", type=str, default="add_67.raw",
                        help="Raw output filename (default: add_67.raw)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory not found: {args.output_dir}")
        print("Please run run_qai_hub.py first to generate outputs.")
        sys.exit(1)
    
    postprocess(args.output_dir, args.wav_output_dir, args.raw_file)


if __name__ == "__main__":
    main()

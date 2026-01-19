"""
Qualcomm AI Hub를 사용한 Demucs ONNX 모델 테스트 스크립트 (배치 버전)
"""

import qai_hub as hub
import numpy as np
import librosa
import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Demucs 모델 상수 (1초 segment 모델)
SAMPLE_RATE = 44100
ORIGINAL_TIME_BRANCH_LEN = 343980
FFT_WINDOW_SIZE = 4096
FFT_HOP_SIZE = 1024
FREQ_BINS = 2048
NB_SOURCES = 4

# 청킹 파라미터
CHUNK_SIZE = 44100  # 1초
OVERLAP = 11025     # 25% 오버랩
HOP_SIZE = CHUNK_SIZE - OVERLAP
FREQ_CHUNK_LEN = 44

# 모델 및 디바이스 설정
MODEL_PATH = "htdemucs_model/htdemucs_seg1.0s_fp16.onnx"
DEVICE_NAME = "Samsung Galaxy S24 (Family)"

# 배치 처리 설정
BATCH_SIZE = 50  # 한 번에 제출할 작업 수


def load_audio(file_path: str) -> np.ndarray:
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    if audio.shape[0] > 2:
        audio = audio[:2]
    return audio


def chunk_audio(audio: np.ndarray, chunk_size: int, hop_size: int) -> list:
    total_samples = audio.shape[1]
    chunks = []
    
    start = 0
    while start < total_samples:
        end = min(start + chunk_size, total_samples)
        chunk = audio[:, start:end]
        
        if chunk.shape[1] < chunk_size:
            padding = chunk_size - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, padding)), mode='constant')
        
        chunks.append((start, chunk))
        start += hop_size
        
        if end >= total_samples:
            break
    
    return chunks


def compute_stft_for_chunk(chunk: np.ndarray) -> np.ndarray:
    n_fft = FFT_WINDOW_SIZE
    hop_length = FFT_HOP_SIZE
    
    stft_results = []
    for ch in range(2):
        stft = librosa.stft(chunk[ch], n_fft=n_fft, hop_length=hop_length,
                           window='hann', center=True, pad_mode='reflect')
        stft_results.append(stft)
    
    stft_array = np.stack(stft_results, axis=0)
    stft_array = stft_array[:, 1:FREQ_BINS+1, :]
    
    n_frames = stft_array.shape[2]
    if n_frames > FREQ_CHUNK_LEN:
        stft_array = stft_array[:, :, :FREQ_CHUNK_LEN]
    elif n_frames < FREQ_CHUNK_LEN:
        padding = FREQ_CHUNK_LEN - n_frames
        stft_array = np.pad(stft_array, ((0, 0), (0, 0), (0, padding)), mode='constant')
    
    real_part = np.real(stft_array)
    imag_part = np.imag(stft_array)
    
    x = np.zeros((4, FREQ_BINS, FREQ_CHUNK_LEN), dtype=np.float32)
    x[0] = real_part[0]
    x[1] = imag_part[0]
    x[2] = real_part[1]
    x[3] = imag_part[1]
    
    return x.reshape(1, 4, FREQ_BINS, FREQ_CHUNK_LEN).astype(np.float32)


def freq_to_time_domain(freq_output: np.ndarray) -> np.ndarray:
    """
    주파수 도메인 출력을 시간 도메인으로 변환 (iSTFT)
    
    입력: (4, 4, 2048, 44) - 4 sources, 4 channels (L_real, L_imag, R_real, R_imag)
    출력: (4, 2, CHUNK_SIZE) - 4 sources, stereo
    """
    time_outputs = []
    
    for source_idx in range(NB_SOURCES):
        source_freq = freq_output[source_idx]  # (4, 2048, 44)
        
        # Real/Imag 채널을 complex로 결합
        L_complex = source_freq[0] + 1j * source_freq[1]  # (2048, 44)
        R_complex = source_freq[2] + 1j * source_freq[3]  # (2048, 44)
        
        # DC bin 추가 (0으로 패딩) - STFT에서 제거했던 DC bin 복원
        L_full = np.zeros((FREQ_BINS + 1, FREQ_CHUNK_LEN), dtype=np.complex64)
        R_full = np.zeros((FREQ_BINS + 1, FREQ_CHUNK_LEN), dtype=np.complex64)
        L_full[1:FREQ_BINS+1, :] = L_complex
        R_full[1:FREQ_BINS+1, :] = R_complex
        
        # iSTFT 수행
        L_time = librosa.istft(
            L_full,
            hop_length=FFT_HOP_SIZE,
            win_length=FFT_WINDOW_SIZE,
            window='hann',
            center=True,
            length=CHUNK_SIZE
        )
        R_time = librosa.istft(
            R_full,
            hop_length=FFT_HOP_SIZE,
            win_length=FFT_WINDOW_SIZE,
            window='hann',
            center=True,
            length=CHUNK_SIZE
        )
        
        time_outputs.append(np.stack([L_time, R_time], axis=0))
    
    return np.stack(time_outputs, axis=0).astype(np.float32)  # (4, 2, CHUNK_SIZE)


def create_overlap_window(chunk_size: int, overlap: int) -> np.ndarray:
    """Hann 기반 overlap-add 윈도우 생성 (합산 시 정확히 1이 되도록)"""
    window = np.ones(chunk_size)
    
    # Hann 윈도우의 절반 사용: sin²(x) + cos²(x) = 1
    # fade_in: 0 → 1 (부드러운 시작)
    fade_in = np.sin(np.linspace(0, np.pi/2, overlap)) ** 2
    # fade_out: 1 → 0 (부드러운 끝)
    fade_out = np.cos(np.linspace(0, np.pi/2, overlap)) ** 2
    
    window[:overlap] = fade_in
    window[-overlap:] = fade_out
    return window


def merge_outputs_with_overlap(outputs: list, original_length: int) -> np.ndarray:
    merged = np.zeros((NB_SOURCES, 2, original_length), dtype=np.float32)
    weights = np.zeros(original_length, dtype=np.float32)
    window = create_overlap_window(CHUNK_SIZE, OVERLAP)
    
    for start_idx, output in outputs:
        end_idx = min(start_idx + CHUNK_SIZE, original_length)
        valid_len = end_idx - start_idx
        
        for source in range(NB_SOURCES):
            for ch in range(2):
                merged[source, ch, start_idx:end_idx] += output[source, ch, :valid_len] * window[:valid_len]
        
        weights[start_idx:end_idx] += window[:valid_len]
    
    weights = np.maximum(weights, 1e-8)
    for source in range(NB_SOURCES):
        for ch in range(2):
            merged[source, ch] /= weights
    
    return merged


def compile_model_for_chunk(device: hub.Device):
    print("\n" + "="*60)
    print("STEP 1: Compiling model")
    print("="*60)
    
    compile_job = hub.submit_compile_job(
        model=MODEL_PATH,
        device=device,
        options="--target_runtime qnn_dlc",
        input_specs=dict(
            input=((1, 2, CHUNK_SIZE), "float16"),
            x=((1, 4, FREQ_BINS, FREQ_CHUNK_LEN), "float16")
        ),
    )
    
    print(f"Compile job: {compile_job.job_id}")
    target_model = compile_job.get_target_model()
    
    if target_model is None:
        print("❌ Compilation failed!")
        return None, None
    
    print(f"✅ Compilation successful!")
    return compile_job, target_model


def submit_inference_batch(target_model, device, chunks_batch, batch_start_idx):
    jobs = []
    
    for i, (start_idx, chunk_audio) in enumerate(chunks_batch):
        input_tensor = chunk_audio.reshape(1, 2, CHUNK_SIZE).astype(np.float16)
        x_tensor = compute_stft_for_chunk(chunk_audio).astype(np.float16)
        
        inference_job = hub.submit_inference_job(
            model=target_model,
            device=device,
            inputs=dict(input=[input_tensor], x=[x_tensor]),
        )
        
        chunk_idx = batch_start_idx + i + 1
        jobs.append((start_idx, inference_job, chunk_idx))
        print(f"  Submitted chunk {chunk_idx}: {inference_job.job_id}")
    
    return jobs


def collect_results(jobs):
    results = []
    failed = []
    
    print(f"\nCollecting {len(jobs)} results...")
    
    for start_idx, job, chunk_idx in jobs:
        try:
            output_data = job.download_output_data()
            
            if output_data is None:
                print(f"  Chunk {chunk_idx}: ❌ No output")
                failed.append(chunk_idx)
                continue
            
            # HTDemucs는 두 가지 출력을 합산해야 함
            # output_0 = 주파수 도메인 (1, 4, 4, 2048, 44) → iSTFT 필요
            # output_1 = 시간 도메인 (1, 4, 2, 44100) → 직접 사용
            
            output_time = None
            output_freq = None
            
            # 시간 도메인 출력 추출
            if 'output_1' in output_data:
                output_time = np.array(output_data['output_1'][0])
                if output_time.ndim == 4:
                    output_time = output_time[0]  # (4, 2, chunk_samples)
            elif 'xt' in output_data:
                output_time = np.array(output_data['xt'][0])
                if output_time.ndim == 4:
                    output_time = output_time[0]
            
            # 주파수 도메인 출력 추출
            if 'output_0' in output_data:
                output_freq = np.array(output_data['output_0'][0])
                if output_freq.ndim == 5:
                    output_freq = output_freq[0]  # (4, 4, 2048, 44)
            elif 'output' in output_data:
                output_freq = np.array(output_data['output'][0])
                if output_freq.ndim == 5:
                    output_freq = output_freq[0]
            
            if output_time is None:
                print(f"    Available outputs: {list(output_data.keys())}")
                raise ValueError(f"Time domain output not found in: {list(output_data.keys())}")
            
            # 주파수 도메인이 있으면 iSTFT 변환 후 합산
            if output_freq is not None:
                wav_from_spec = freq_to_time_domain(output_freq)
                output = wav_from_spec + output_time
            else:
                # 주파수 도메인이 없으면 시간 도메인만 사용
                print(f"  Chunk {chunk_idx}: ⚠️ No freq output, using time only")
                output = output_time
            
            results.append((start_idx, output))
            print(f"  Chunk {chunk_idx}: ✅")
            
        except Exception as e:
            print(f"  Chunk {chunk_idx}: ❌ Error: {e}")
            failed.append(chunk_idx)
    
    return results, failed


def run_batch_inference(target_model, device, chunks, original_length, batch_size=50):
    print("\n" + "="*60)
    print(f"STEP 2: Batch inference for {len(chunks)} chunks")
    print(f"  Batch size: {batch_size}")
    print("="*60)
    
    all_results = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]
        
        print(f"\n--- Batch {batch_num + 1}/{total_batches} (chunks {start+1}-{end}) ---")
        
        # 배치 제출
        print("Submitting jobs...")
        jobs = submit_inference_batch(target_model, device, batch, start)
        
        # 결과 수집
        results, failed = collect_results(jobs)
        all_results.extend(results)
        
        if failed:
            print(f"⚠️ {len(failed)} chunks failed in this batch")
        
        print(f"Batch {batch_num + 1} complete: {len(results)}/{len(batch)} successful")
    
    if len(all_results) < len(chunks):
        print(f"\n⚠️ Warning: Only {len(all_results)}/{len(chunks)} chunks processed")
    
    # 결과 병합
    print("\nMerging results with overlap-add...")
    merged = merge_outputs_with_overlap(all_results, original_length)
    print(f"  Merged shape: {merged.shape}")
    
    return merged


def save_stems_as_raw(stems: np.ndarray, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "add_67.raw")
    output_with_batch = stems.reshape(1, *stems.shape).astype(np.float32)
    output_with_batch.tofile(output_path)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch Demucs inference on Qualcomm AI Hub")
    parser.add_argument("audio_file", nargs="?", default="../day6-happy.mp3")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Jobs per batch (default: {BATCH_SIZE})")
    args = parser.parse_args()
    
    batch_size = args.batch_size
    
    print("="*60)
    print("Demucs - Batch Inference on Qualcomm AI Hub")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE_NAME}")
    print(f"Chunk size: {CHUNK_SIZE/SAMPLE_RATE:.2f}s")
    print(f"Batch size: {BATCH_SIZE}")
    
    device = hub.Device(DEVICE_NAME)
    
    # 1. 컴파일
    compile_job, target_model = compile_model_for_chunk(device)
    if target_model is None:
        sys.exit(1)
    
    # 2. 오디오 로드
    print("\n" + "="*60)
    print("Loading audio")
    print("="*60)
    
    if not os.path.exists(args.audio_file):
        print(f"Error: File not found: {args.audio_file}")
        sys.exit(1)
    
    audio = load_audio(args.audio_file)
    original_length = audio.shape[1]
    print(f"  File: {args.audio_file}")
    print(f"  Duration: {original_length/SAMPLE_RATE:.2f}s")
    
    chunks = chunk_audio(audio, CHUNK_SIZE, HOP_SIZE)
    print(f"  Chunks: {len(chunks)}")
    
    # 3. 배치 추론
    start_time = time.time()
    merged_output = run_batch_inference(target_model, device, chunks, original_length, batch_size)
    elapsed = time.time() - start_time
    
    print(f"\nTotal inference time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    
    # 4. 저장
    print("\n" + "="*60)
    print("STEP 3: Saving outputs")
    print("="*60)
    save_stems_as_raw(merged_output)
    
    print("\nDone! Run postprocess_demucs.py to convert to WAV")


if __name__ == "__main__":
    main()

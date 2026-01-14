"""
1초 청크 Demucs ONNX 모델 웹 테스터 백엔드
Overlap-add 방식으로 청크를 병합하여 자연스러운 결과 생성
"""
import os
import shutil
import tempfile
from pathlib import Path
import torch
import torchaudio
import onnxruntime as ort
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

# Demucs 모듈 임포트
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "demucs-for-onnx"))
from demucs.htdemucs import standalone_spec, standalone_magnitude, standalone_ispec, standalone_mask

# Initialize FastAPI
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR.parent / "qualcomm-test/htdemucs_model/htdemucs_seg1.0s_fp16.onnx"
OUTPUT_DIR = BASE_DIR / "results"
FRONTEND_DIR = BASE_DIR / "frontend"

# 1초 청크 모델 파라미터
SAMPLE_RATE = 44100
SEGMENT_LENGTH = 44100  # 1초
OVERLAP = 11025  # 25% 오버랩
HOP_SIZE = SEGMENT_LENGTH - OVERLAP  # 33075
NB_SOURCES = 4  # drums, bass, other, vocals

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load ONNX Model
print(f"Loading 1-second chunk ONNX Model from {MODEL_PATH}...")
sess = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
print("Model Loaded.")


def create_overlap_window(chunk_size: int, overlap: int) -> np.ndarray:
    """Overlap-add용 윈도우 생성 (fade-in/fade-out)"""
    window = np.ones(chunk_size)
    
    # Fade-in at start
    fade_in = np.linspace(0, 1, overlap)
    window[:overlap] = fade_in
    
    # Fade-out at end
    fade_out = np.linspace(1, 0, overlap)
    window[-overlap:] = fade_out
    
    return window


def merge_outputs_with_overlap(outputs: list, original_length: int) -> np.ndarray:
    """
    청크 출력을 overlap-add 방식으로 병합
    
    Args:
        outputs: [(start_idx, output_tensor), ...] 
                 output_tensor shape: (1, 4, 2, chunk_samples)
        original_length: 원본 오디오 길이
    
    Returns:
        merged: (1, 4, 2, original_length)
    """
    merged = np.zeros((1, NB_SOURCES, 2, original_length), dtype=np.float32)
    weights = np.zeros(original_length, dtype=np.float32)
    
    window = create_overlap_window(SEGMENT_LENGTH, OVERLAP)
    
    for start_idx, output in outputs:
        end_idx = min(start_idx + SEGMENT_LENGTH, original_length)
        valid_len = end_idx - start_idx
        
        # 윈도우 적용
        for source in range(NB_SOURCES):
            for ch in range(2):
                merged[0, source, ch, start_idx:end_idx] += output[0, source, ch, :valid_len] * window[:valid_len]
        
        weights[start_idx:end_idx] += window[:valid_len]
    
    # 가중치로 정규화
    weights = np.maximum(weights, 1e-8)
    for source in range(NB_SOURCES):
        for ch in range(2):
            merged[0, source, ch] /= weights
    
    return merged


@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Load Audio
        wav_np, sr = sf.read(tmp_path)
        wav = torch.from_numpy(wav_np).float()
        
        # Handle shape (Time, Channels) -> (Channels, Time)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.t()
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            sr = SAMPLE_RATE
        
        # Ensure Stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]

        # Normalize
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)
        
        original_length = wav.shape[-1]

        # 청킹 로직 (오버랩 포함)
        chunks = []
        start = 0
        while start < original_length:
            end = min(start + SEGMENT_LENGTH, original_length)
            chunk = wav[:, start:end]
            
            # 마지막 청크가 짧으면 패딩
            if chunk.shape[1] < SEGMENT_LENGTH:
                chunk = torch.nn.functional.pad(chunk, (0, SEGMENT_LENGTH - chunk.shape[1]))
            
            chunks.append((start, chunk))
            start += HOP_SIZE
            
            if end >= original_length:
                break
        
        print(f"Processing {len(chunks)} chunks for {original_length/SAMPLE_RATE:.2f}s audio...")
        
        out_chunks = []
        
        for i, (start_idx, chunk) in enumerate(chunks):
            chunk = chunk.unsqueeze(0)  # (1, 2, SEG)
            
            # Preprocess
            spec = standalone_spec(chunk)
            magspec = standalone_magnitude(spec)
            
            # Prepare inputs (float16)
            chunk_np = chunk.numpy().astype(np.float16)
            magspec_np = magspec.numpy().astype(np.float16)
            
            # Run Inference
            outputs = sess.run(None, {'input': chunk_np, 'x': magspec_np})
            
            output_spec = torch.from_numpy(outputs[0].astype(np.float32))  # (1, 4, 4, Fr, T)
            output_time = torch.from_numpy(outputs[1].astype(np.float32))  # (1, 4, 2, L)
            
            # Postprocess
            spec_complex = standalone_mask(None, output_spec, cac=True)
            wav_from_spec = standalone_ispec(spec_complex, length=SEGMENT_LENGTH)
            
            # Combine
            out_chunk = wav_from_spec + output_time
            out_chunks.append((start_idx, out_chunk.numpy()))
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(chunks)} chunks...")
        
        # Overlap-add 병합
        out_wav = merge_outputs_with_overlap(out_chunks, original_length)
        out_wav = torch.from_numpy(out_wav)
        
        # Save files
        file_id = os.path.splitext(os.path.basename(file.filename))[0]
        out_paths = {}
        sources = ['drums', 'bass', 'other', 'vocals']
        
        for idx, source in enumerate(sources):
            stem = out_wav[0, idx, :, :]
            stem_path = OUTPUT_DIR / f"{file_id}_{source}.wav"
            sf.write(str(stem_path), stem.t().numpy(), SAMPLE_RATE)
            out_paths[source] = f"/results/{stem_path.name}"
            
        # Cleanup temp file
        os.unlink(tmp_path)
        
        print(f"Done! Saved stems to {OUTPUT_DIR}")
        return out_paths

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return FileResponse(FRONTEND_DIR / "index.html")


# Serve static files
app.mount("/results", StaticFiles(directory=OUTPUT_DIR), name="results")

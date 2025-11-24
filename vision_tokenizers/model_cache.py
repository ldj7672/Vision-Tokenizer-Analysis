"""
Model Cache Configuration

이 모듈은 모든 모델 다운로드 경로를 ./model_weights로 설정합니다.
"""

import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 모델 가중치 저장 경로
MODEL_WEIGHTS_DIR = PROJECT_ROOT / "model_weights"

# 디렉토리 생성
MODEL_WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)

# Hugging Face 캐시 경로 설정
os.environ['HF_HOME'] = str(MODEL_WEIGHTS_DIR / "huggingface")
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_WEIGHTS_DIR / "huggingface" / "transformers")
os.environ['HF_DATASETS_CACHE'] = str(MODEL_WEIGHTS_DIR / "huggingface" / "datasets")
os.environ['HUGGINGFACE_HUB_CACHE'] = str(MODEL_WEIGHTS_DIR / "huggingface" / "hub")

# Torch Hub 캐시 경로 설정
os.environ['TORCH_HOME'] = str(MODEL_WEIGHTS_DIR / "torch")

print(f"✓ Model cache directory set to: {MODEL_WEIGHTS_DIR}")
print(f"  - Hugging Face: {MODEL_WEIGHTS_DIR / 'huggingface'}")
print(f"  - Torch Hub: {MODEL_WEIGHTS_DIR / 'torch'}")


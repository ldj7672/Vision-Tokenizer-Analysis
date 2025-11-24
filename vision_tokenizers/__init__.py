"""
Vision Tokenizers Package

이 패키지는 다양한 Vision Tokenizer 구현을 제공합니다.

Discrete Tokenizers (자동 다운로드):
- TATokTokenizer: TA-Tok (Text-Aligned Tokenizer)
- ARDTokDetokenizer: AR-DTok (Autoregressive De-Tokenizer)
- SANADetokenizer: SANA Dif-DTok (Diffusion De-Tokenizer)
- Lumina2Detokenizer: Lumina2 Dif-DTok (Diffusion De-Tokenizer, 추천 🌟)

Discrete Tokenizers (수동 다운로드):
- VQGANTokenizer: VQ-GAN (Vector Quantized GAN)

Continuous Tokenizers (Baseline):
- VAELDMTokenizer: Stable Diffusion VAE (with optional quantization)
"""

# 모델 캐시 경로 설정 (가장 먼저 실행)
from . import model_cache

from .base import VisionTokenizerBase
from .vae_ldm import VAELDMTokenizer
from .tatok import TATokTokenizer
from .ar_dtok import ARDTokDetokenizer
from .sana_dtok import SANADetokenizer
from .lumina2_dtok import Lumina2Detokenizer
from .magvit2 import MAGVIT2Tokenizer
from .titok import TiTokTokenizer
from .tatok_combined import (
    TATokARDTokTokenizer,
    TATokSANATokenizer,
    TATokLumina2Tokenizer
)

__all__ = [
    'VisionTokenizerBase',
    'VAELDMTokenizer',
    'TATokTokenizer',
    'ARDTokDetokenizer',
    'SANADetokenizer',
    'Lumina2Detokenizer',
    'VQGANTokenizer',
    'MAGVIT2Tokenizer',
    'TiTokTokenizer',
    'TATokARDTokTokenizer',
    'TATokSANATokenizer',
    'TATokLumina2Tokenizer',
]

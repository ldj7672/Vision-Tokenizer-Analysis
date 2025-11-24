"""
TiTok Tokenizer Wrapper

이 모듈은 TiTok Vision Tokenizer를 래핑합니다.

원리:
- "An Image is Worth 32 Tokens for Reconstruction and Generation"
- 매우 적은 토큰으로 고품질 재구성 (32 tokens!)
- 1D latent tokenization
- Multi-scale patch aggregation

연동 방법:
- GitHub: https://github.com/bytedance/1d-tokenizer
- Hugging Face: fun-research/TiTok
- 자동 다운로드 지원

특징:
- 극도로 압축된 표현 (32 tokens for 256x256 image)
- 1D latent space
- Codebook size: 4096
- 빠른 inference

참고:
- 논문: "An Image is Worth 32 Tokens for Reconstruction and Generation" (ByteDance, NeurIPS 2024)
- GitHub: https://github.com/bytedance/1d-tokenizer
- 자동 다운로드: ✅
"""

from typing import Dict, Any, Union
import torch
from PIL import Image
import numpy as np
import sys
from pathlib import Path

from .base import VisionTokenizerBase


class TiTokTokenizer(VisionTokenizerBase):
    """
    TiTok 기반 Vision Tokenizer
    
    매우 적은 토큰으로 이미지를 표현합니다 (32 tokens for 256x256).
    """
    
    def __init__(self,
                 image_size: int = 256,
                 num_latent_tokens: int = 32,
                 codebook_size: int = 4096,
                 device: str = "cuda",
                 model_name: str = "yucornetto/tokenizer_titok_l32_imagenet",
                 **kwargs):
        """
        Args:
            image_size: 입력 이미지 크기 (256)
            num_latent_tokens: Latent 토큰 개수 (32, 64, or 128)
            codebook_size: Codebook 크기 (4096)
            device: 디바이스 ("cuda" or "cpu")
            model_name: Hugging Face 모델 이름
        """
        super().__init__(device=device, **kwargs)
        self.image_size = image_size
        self.num_latent_tokens = num_latent_tokens
        self.codebook_size = codebook_size
        self.model_name = model_name
        
        try:
            # Import from extracted modules
            from .titok_modules import TiTok
            
            print(f"Loading TiTok from {model_name}...")
            
            # Load pretrained model
            self.model = TiTok.from_pretrained(model_name)
            self.model = self.model.to(device)
            self.model.eval()
            self.model.requires_grad_(False)
            
            # Apply optimizations
            if self.use_bf16:
                self.model = self.model.to(torch.bfloat16)
            
            if self.use_compile:
                self.model.encoder = torch.compile(self.model.encoder, mode='reduce-overhead')
                self.model.decoder = torch.compile(self.model.decoder, mode='reduce-overhead')
            
            print(f"✓ Loaded TiTok (tokens={num_latent_tokens}, codebook={codebook_size})")
            
        except Exception as e:
            print(f"✗ Failed to load TiTok: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def encode(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        이미지를 TiTok 토큰으로 인코딩
        
        Returns:
            Dict with keys:
                - 'tokens': torch.Tensor, shape (B, 1, N) - discrete token indices (1D)
                - 'token_count': int
                - 'latent_shape': Tuple[int] - 1D shape
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        
        # Preprocess image
        if isinstance(image, Image.Image):
            # Convert PIL to tensor [0, 1]
            image_np = np.array(image.resize((self.image_size, self.image_size))).astype(np.float32)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0) / 255.0
            image_tensor = image_tensor.to(self.device)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image_tensor = torch.from_numpy(image).float() / 255.0
            else:
                image_tensor = torch.from_numpy(image).float()
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
        else:
            image_tensor = image.to(self.device)
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
        
        # Convert to BF16 if enabled
        if self.use_bf16:
            image_tensor = image_tensor.to(torch.bfloat16)
        
        with torch.no_grad():
            # TiTok tokenization
            if self.model.quantize_mode == "vq":
                encoded_tokens = self.model.encode(image_tensor)[1]["min_encoding_indices"]
            elif self.model.quantize_mode == "vae":
                posteriors = self.model.encode(image_tensor)[1]
                encoded_tokens = posteriors.sample()
            else:
                # Default: assume it returns tokens directly
                encoded_tokens = self.model.encode(image_tensor)[1]["min_encoding_indices"]
        
        # encoded_tokens shape: (B, 1, N) where N is num_latent_tokens
        B = encoded_tokens.shape[0]
        if encoded_tokens.ndim == 3:
            N = encoded_tokens.shape[2]
        else:
            N = encoded_tokens.shape[1]
            encoded_tokens = encoded_tokens.unsqueeze(1)
        
        return {
            'tokens': encoded_tokens,  # (B, 1, N) - 1D tokens
            'indices': encoded_tokens,
            'token_count': N,
            'latent_shape': (N,),  # 1D latent
        }
    
    def decode(self, tokens: Union[torch.Tensor, Dict[str, Any]]) -> Image.Image:
        """
        TiTok 토큰을 이미지로 디코딩
        
        Args:
            tokens: Dict with 'tokens' key or torch.Tensor (token indices)
        
        Returns:
            PIL.Image: 재구성된 이미지
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        
        # Extract tokens
        if isinstance(tokens, dict):
            token_tensor = tokens['tokens']
        else:
            token_tensor = tokens
        
        with torch.no_grad():
            # Decode from token indices
            reconstructed = self.model.decode_tokens(token_tensor)
            reconstructed = torch.clamp(reconstructed, 0.0, 1.0)
        
        # postprocess handles batch dimension removal and BF16->float32 conversion
        return self.postprocess(reconstructed)
    
    def info(self) -> Dict[str, Any]:
        """TiTok 토크나이저 정보 반환"""
        return {
            'name': 'TiTok',
            'type': '1D-Latent-Tokenization',
            'codebook_size': self.codebook_size,
            'num_latent_tokens': self.num_latent_tokens,
            'image_size': (self.image_size, self.image_size),
            'compression_ratio': (self.image_size ** 2) / self.num_latent_tokens,  # e.g., 256^2 / 32 = 2048x
            'num_parameters': self.count_parameters(),
            'description': f'TiTok: An Image is Worth {self.num_latent_tokens} Tokens (극도로 압축된 표현)',
            'paper': 'An Image is Worth 32 Tokens for Reconstruction and Generation (ByteDance, NeurIPS 2024)',
            'github': 'https://github.com/bytedance/1d-tokenizer',
            'huggingface': 'https://huggingface.co/fun-research/TiTok',
            'auto_download': True,
        }

"""
Stable Diffusion / LDM VAE Tokenizer Wrapper

이 모듈은 Stable Diffusion에서 사용되는 VAE (Variational Autoencoder)를 
Vision Tokenizer로 래핑합니다.

원리:
- Stable Diffusion의 VAE는 이미지를 연속적인 latent space로 인코딩합니다.
- 입력 이미지 (3, 512, 512)를 (4, 64, 64) latent로 압축 (8x downsampling)
- KL-divergence를 사용한 정규화로 안정적인 latent space 학습
- 높은 재구성 품질과 perceptual quality를 제공

연동 방법:
- diffusers 라이브러리의 AutoencoderKL 모델 사용
- Hugging Face Hub에서 사전학습된 가중치 로드
- 지원 모델: stabilityai/sd-vae-ft-mse, stabilityai/sd-vae-ft-ema 등

특징:
- 연속적인 latent representation (discrete tokens 아님)
- 높은 재구성 품질 (PSNR ~30dB)
- 8x spatial compression
- 4-channel latent space
"""

from typing import Dict, Any, Union
import torch
from PIL import Image
import numpy as np

from .base import VisionTokenizerBase


class VAELDMTokenizer(VisionTokenizerBase):
    """
    Stable Diffusion VAE 기반 Vision Tokenizer
    
    이 토크나이저는 연속적인 latent space를 사용하므로
    discrete tokens를 직접 생성하지 않습니다.
    대신 latent representation을 양자화하여 토큰으로 변환할 수 있습니다.
    """
    
    def __init__(self, 
                 model_name: str = "stabilityai/sd-vae-ft-mse",
                 device: str = "cuda",
                 use_quantization: bool = False,
                 num_quantization_bins: int = 256,
                 **kwargs):
        """
        Args:
            model_name: Hugging Face model ID
            device: 디바이스 ("cuda" or "cpu")
            use_quantization: latent를 discrete tokens로 양자화할지 여부
            num_quantization_bins: 양자화 bin 수 (use_quantization=True일 때)
        """
        super().__init__(device=device, **kwargs)
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.num_quantization_bins = num_quantization_bins
        
        try:
            from diffusers import AutoencoderKL
            self.model = AutoencoderKL.from_pretrained(model_name).to(device)
            self.model.eval()
            
            # Apply optimizations
            if self.use_bf16:
                self.model = self.model.to(torch.bfloat16)
            
            if self.use_compile:
                self.model.encoder = torch.compile(self.model.encoder, mode='reduce-overhead')
                self.model.decoder = torch.compile(self.model.decoder, mode='reduce-overhead')
            
            print(f"✓ Loaded VAE model from {model_name}")
        except Exception as e:
            print(f"✗ Failed to load VAE model: {e}")
            print(f"  Please install: pip install diffusers")
            self.model = None
        
    def encode(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        이미지를 VAE latent로 인코딩
        
        Returns:
            Dict with keys:
                - 'latent': torch.Tensor, shape (B, 4, H/8, W/8) - 연속 latent
                - 'tokens': torch.Tensor (optional) - 양자화된 토큰 (use_quantization=True)
                - 'token_count': int
                - 'latent_shape': Tuple[int, int]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please install diffusers: pip install diffusers")
        
        # VAE는 8의 배수 크기가 필요하므로 리사이즈
        # 원본 이미지 크기를 8의 배수로 조정
        if isinstance(image, Image.Image):
            orig_w, orig_h = image.size
            # 8의 배수로 조정
            new_h = (orig_h // 8) * 8
            new_w = (orig_w // 8) * 8
            image_tensor = self.preprocess(image, target_size=(new_h, new_w))
        else:
            image_tensor = self.preprocess(image)
        
        # Convert to BF16 if enabled
        if self.use_bf16:
            image_tensor = image_tensor.to(torch.bfloat16)
        
        with torch.no_grad():
            # VAE encoding
            posterior = self.model.encode(image_tensor).latent_dist
            latent = posterior.sample()  # or .mode() for deterministic
            
            # Optional: scale latent by scaling factor
            if hasattr(self.model.config, 'scaling_factor'):
                latent = latent * self.model.config.scaling_factor
        
        result = {
            'latent': latent,
            'latent_shape': (latent.shape[2], latent.shape[3]),
            'token_count': latent.shape[2] * latent.shape[3]
        }
        
        if self.use_quantization:
            # Quantize continuous latent to discrete tokens
            tokens = self._quantize_latent(latent)
            result['tokens'] = tokens
        
        return result
    
    def decode(self, tokens: Union[torch.Tensor, Dict[str, Any]]) -> Image.Image:
        """
        latent를 이미지로 디코딩
        
        Args:
            tokens: Dict with 'latent' key or torch.Tensor (latent or quantized tokens)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please install diffusers: pip install diffusers")
        
        if isinstance(tokens, dict):
            latent = tokens['latent']
        else:
            if self.use_quantization:
                # Dequantize tokens back to continuous latent
                latent = self._dequantize_tokens(tokens)
            else:
                latent = tokens
        
        # Unscale latent
        if hasattr(self.model.config, 'scaling_factor'):
            latent = latent / self.model.config.scaling_factor
        
        with torch.no_grad():
            # VAE decoding
            image_tensor = self.model.decode(latent).sample
        
        return self.postprocess(image_tensor)
    
    def info(self) -> Dict[str, Any]:
        """VAE 토크나이저 정보 반환"""
        return {
            'name': 'Stable Diffusion VAE',
            'type': 'VAE',
            'model_name': self.model_name,
            'codebook_size': self.num_quantization_bins if self.use_quantization else None,
            'latent_channels': 4,
            'latent_shape': (64, 64),  # for 512x512 input
            'compression_ratio': 8.0,  # 8x spatial downsampling
            'num_parameters': self.count_parameters(),
            'input_size': (512, 512),
            'pretrained': True,
            'use_quantization': self.use_quantization,
            'description': 'Continuous latent space VAE from Stable Diffusion'
        }
    
    def _quantize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        연속 latent를 discrete tokens로 양자화
        
        Args:
            latent: shape (B, C, H, W)
        
        Returns:
            tokens: shape (B, C, H, W) with integer values [0, num_bins-1]
        """
        # Simple uniform quantization
        # 1. Normalize latent to [0, 1]
        latent_min = latent.min()
        latent_max = latent.max()
        normalized = (latent - latent_min) / (latent_max - latent_min + 1e-8)
        
        # 2. Multiply by num_bins and round
        tokens = (normalized * (self.num_quantization_bins - 1)).round().long()
        
        # 3. Clip to [0, num_bins-1]
        tokens = torch.clamp(tokens, 0, self.num_quantization_bins - 1)
        
        # Store normalization params for dequantization
        self._latent_min = latent_min
        self._latent_max = latent_max
        
        return tokens
    
    def _dequantize_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        discrete tokens를 연속 latent로 역양자화
        
        Args:
            tokens: shape (B, C, H, W) with integer values
        
        Returns:
            latent: shape (B, C, H, W) with continuous values
        """
        # Reverse quantization
        normalized = tokens.float() / (self.num_quantization_bins - 1)
        
        # Denormalize
        if hasattr(self, '_latent_min') and hasattr(self, '_latent_max'):
            latent = normalized * (self._latent_max - self._latent_min) + self._latent_min
        else:
            # Fallback: assume [-1, 1] range
            latent = normalized * 2.0 - 1.0
        
        return latent


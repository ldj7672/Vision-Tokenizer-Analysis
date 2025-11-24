"""
MAGVIT-v2 Tokenizer Wrapper

이 모듈은 Open-MAGVIT2 Vision Tokenizer를 래핑합니다.

원리:
- Lookup Free Quantization (LFQ) 사용
- Video/Image tokenization SOTA
- Codebook size: 262144 (18-bit)
- Discriminator 기반 adversarial training

연동 방법:
- GitHub: https://github.com/TencentARC/Open-MAGVIT2
- Hugging Face: TencentARC/Open-MAGVIT2
- 자동 다운로드 지원

특징:
- Lookup Free Quantization (VQ보다 효율적)
- Video와 Image 모두 지원
- Codebook size: 262144 (18-bit)
- 매우 높은 재구성 품질

참고:
- 논문: "Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation"
- GitHub: https://github.com/TencentARC/Open-MAGVIT2
- 자동 다운로드: ✅
"""

from typing import Dict, Any, Union
import torch
from PIL import Image
import numpy as np
import sys
from pathlib import Path

from .base import VisionTokenizerBase


class MAGVIT2Tokenizer(VisionTokenizerBase):
    """
    Open-MAGVIT2 기반 Vision Tokenizer
    
    Lookup Free Quantization을 사용하여 이미지를 discrete token으로 변환합니다.
    """
    
    def __init__(self,
                 image_size: int = 256,
                 codebook_size: int = 262144,  # 18-bit (2^18)
                 device: str = "cuda",
                 model_name: str = "TencentARC/Open-MAGVIT2",
                 **kwargs):
        """
        Args:
            image_size: 입력 이미지 크기 (256 or 512)
            codebook_size: Codebook 크기 (262144 for Open-MAGVIT2)
            device: 디바이스 ("cuda" or "cpu")
            model_name: Hugging Face 모델 이름
        """
        super().__init__(device=device, **kwargs)
        self.image_size = image_size
        self.codebook_size = codebook_size
        self.model_name = model_name
        
        try:
            # Import from extracted modules
            import os
            from huggingface_hub import hf_hub_download
            from omegaconf import OmegaConf
            
            from .magvit2_modules import VQModel
            
            print(f"Loading Open-MAGVIT2 (image_size={image_size})...")
            
            # Download checkpoint
            if image_size == 256:
                ckpt_name = "imagenet_256_L.ckpt"
            elif image_size == 128:
                ckpt_name = "imagenet_128_L.ckpt"
            else:
                # Use 256 model for other sizes
                ckpt_name = "imagenet_256_L.ckpt"
            
            ckpt_path = hf_hub_download(model_name, ckpt_name)
            
            # Use default config (no need for config file anymore)
            config = OmegaConf.create({
                    "model": {
                        "init_args": {
                            "embed_dim": 18 if image_size == 256 else 8,
                            "n_embed": codebook_size,
                            "sample_minimization_weight": 1.0,
                            "batch_maximization_weight": 1.0,
                            "ddconfig": {
                                "double_z": False,
                                "z_channels": 18 if image_size == 256 else 8,
                                "resolution": image_size,
                                "in_channels": 3,
                                "out_ch": 3,
                                "ch": 128,
                                "ch_mult": [1, 1, 2, 2, 4],
                                "num_res_blocks": 4 if image_size == 256 else 2,
                            },
                            "lossconfig": {
                                "target": "vision_tokenizers.magvit2_modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
                                "params": {
                                    "disc_conditional": False,
                                    "disc_in_channels": 3,
                                    "disc_start": 0,
                                    "disc_weight": 0.8,
                                    "gen_loss_weight": 0.1,
                                    "lecam_loss_weight": 0.05,
                                    "codebook_weight": 0.1,
                                    "commit_weight": 0.25,
                                }
                            }
                        }
                    }
                })
            
            # Load model
            self.model = VQModel(**config.model.init_args)
            
            # Load checkpoint
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            
            self.model = self.model.to(device)
            self.model.eval()
            
            # Apply optimizations
            if self.use_bf16:
                self.model = self.model.to(torch.bfloat16)
            
            if self.use_compile:
                self.model.encoder = torch.compile(self.model.encoder, mode='reduce-overhead')
                self.model.decoder = torch.compile(self.model.decoder, mode='reduce-overhead')
            
            print(f"✓ Loaded Open-MAGVIT2 (image_size={image_size}, codebook_size={codebook_size})")
            
        except Exception as e:
            print(f"✗ Failed to load Open-MAGVIT2: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def encode(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        이미지를 MAGVIT-v2 토큰으로 인코딩
        
        Returns:
            Dict with keys:
                - 'tokens': torch.Tensor, shape (B, H, W) - discrete token indices
                - 'token_count': int
                - 'latent_shape': Tuple[int, int]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        
        # Preprocess image
        if isinstance(image, Image.Image):
            image_tensor = self.preprocess(image, target_size=(self.image_size, self.image_size))
        else:
            image_tensor = self.preprocess(image)
        
        # Convert to BF16 if enabled
        if self.use_bf16:
            image_tensor = image_tensor.to(torch.bfloat16)
        
        with torch.no_grad():
            # Open-MAGVIT2 encoding
            if self.model.use_ema:
                with self.model.ema_scope():
                    quant, diff, indices, _ = self.model.encode(image_tensor)
            else:
                quant, diff, indices, _ = self.model.encode(image_tensor)
            
            # indices shape: (B, H, W) or (B, N) or (N,)
            if indices.ndim == 3:
                B, H, W = indices.shape
                latent_shape = (H, W)
                token_count = H * W
            elif indices.ndim == 2:
                B, N = indices.shape
                H = W = int(N ** 0.5)
                latent_shape = (H, W)
                token_count = N
                indices = indices.reshape(B, H, W)
            elif indices.ndim == 1:
                # Single image without batch dimension
                N = indices.shape[0]
                H = W = int(N ** 0.5)
                latent_shape = (H, W)
                token_count = N
                indices = indices.reshape(1, H, W)  # Add batch dimension
            else:
                raise ValueError(f"Unexpected token shape: {indices.shape}")
        
        return {
            'tokens': indices,
            'indices': indices,
            'token_count': token_count,
            'latent_shape': latent_shape,
            'quantized': quant,  # For decode
        }
    
    def decode(self, tokens: Union[torch.Tensor, Dict[str, Any]]) -> Image.Image:
        """
        MAGVIT-v2 토큰을 이미지로 디코딩
        
        Args:
            tokens: Dict with 'quantized' key or torch.Tensor (token indices)
        
        Returns:
            PIL.Image: 재구성된 이미지
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        
        # Extract quantized features or indices
        if isinstance(tokens, dict):
            if 'quantized' in tokens:
                quant = tokens['quantized']
            else:
                # Need to quantize indices
                token_indices = tokens['tokens']
                # This requires embedding lookup - use model's quantize
                quant = self.model.quantize.get_codebook_entry(token_indices.flatten(), token_indices.shape)
        else:
            # Assume it's quantized features
            quant = tokens
        
        with torch.no_grad():
            # Decode
            if self.model.use_ema:
                with self.model.ema_scope():
                    reconstructed = self.model.decode(quant)
            else:
                reconstructed = self.model.decode(quant)
        
        return self.postprocess(reconstructed)
    
    def info(self) -> Dict[str, Any]:
        """MAGVIT-v2 토크나이저 정보 반환"""
        return {
            'name': 'Open-MAGVIT2',
            'type': 'Lookup-Free-Quantization',
            'codebook_size': self.codebook_size,
            'image_size': (self.image_size, self.image_size),
            'compression_ratio': 16.0,  # 16x spatial compression
            'num_parameters': self.count_parameters(),
            'description': 'Open-MAGVIT2 with Lookup Free Quantization (18-bit codebook)',
            'paper': 'Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation',
            'github': 'https://github.com/TencentARC/Open-MAGVIT2',
            'huggingface': 'https://huggingface.co/TencentARC/Open-MAGVIT2',
            'auto_download': True,
        }

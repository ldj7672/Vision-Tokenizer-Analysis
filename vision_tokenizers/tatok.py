"""
TA-Tok (Text-Aligned Tokenizer) Wrapper

이 모듈은 TA-Tok Vision Tokenizer를 래핑합니다.

원리:
- Text-aligned representation learning을 통한 이미지 토큰화
- SigLIP2 기반 인코더로 시각-언어 정렬된 표현 학습
- Vision과 Language를 통합하는 "dialect" 접근법
- Understanding과 Generation을 모두 지원

연동 방법:
- 공식 GitHub: https://github.com/csuhan/Tar
- Hugging Face Hub: ByteDance-Seed/Tar-TA-Tok
- 자동 다운로드 지원 (transformers AutoModel)

특징:
- Text-aligned representations (시각-언어 통합)
- Codebook size: 65536 (매우 큰 vocabulary)
- Input size: 384px
- SigLIP2-SO400M-patch14-384 인코더 사용
- Understanding (Image-to-Text)와 Generation (Text-to-Image) 모두 지원

참고:
- 논문: "Vision as a Dialect: Unifying Visual Understanding and Generation via Text-Aligned Representations" (NeurIPS 2025)
- GitHub: https://github.com/csuhan/Tar
- 프로젝트: tar.csuhan.com
- 자동 다운로드: ✅ (transformers 사용)
"""

from typing import Dict, Any, Union
import torch
from PIL import Image
import numpy as np
import sys
from pathlib import Path

from .base import VisionTokenizerBase

# Tar 프로젝트의 tok 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from tok.ta_tok import TextAlignedTokenizer


class TATokTokenizer(VisionTokenizerBase):
    """
    TA-Tok 기반 Vision Tokenizer
    
    Text-aligned representation을 사용하여 이미지를 discrete token으로 변환합니다.
    """
    
    def __init__(self,
                 checkpoint_path: str = None,
                 model_name: str = "ByteDance-Seed/Tar-TA-Tok",
                 device: str = "cuda",
                 **kwargs):
        """
        Args:
            checkpoint_path: 로컬 체크포인트 경로 (예: model_weights/ta_tok.pth)
            model_name: Hugging Face model name (checkpoint_path가 None일 때 사용)
            device: 디바이스 ("cuda" or "cpu")
        """
        super().__init__(device=device, **kwargs)
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        
        # TA-Tok 설정
        self.codebook_size = 65536  # TA-Tok의 실제 codebook size
        self.input_size = 384  # SigLIP2 input size
        
        try:
            # 로컬 체크포인트 우선 사용
            if checkpoint_path and Path(checkpoint_path).exists():
                print(f"Loading TA-Tok from local checkpoint: {checkpoint_path}")
                self.model = TextAlignedTokenizer.from_checkpoint(
                    checkpoint_path,
                    load_teacher=False
                ).to(device)
                self.model.eval()
                self.model.set_vq_eval_deterministic(deterministic=True)
                
                # Apply optimizations
                if self.use_bf16:
                    self.model = self.model.to(torch.bfloat16)
                
                if self.use_compile:
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                
                print(f"✓ Loaded TA-Tok from {checkpoint_path}")
            else:
                # Hugging Face Hub에서 다운로드 시도
                print(f"⚠ Local checkpoint not found: {checkpoint_path}")
                print(f"  TA-Tok requires manual checkpoint download from:")
                print(f"  https://huggingface.co/csuhan/TA-Tok/resolve/main/ta_tok.pth")
                print(f"  Save to: model_weights/ta_tok.pth")
                self.model = None
                
        except Exception as e:
            print(f"✗ Failed to load TA-Tok: {e}")
            print(f"  Please download checkpoint from:")
            print(f"  https://huggingface.co/csuhan/TA-Tok/resolve/main/ta_tok.pth")
            import traceback
            traceback.print_exc()
            self.model = None
        
    def encode(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        이미지를 TA-Tok 토큰으로 인코딩
        
        Returns:
            Dict with keys:
                - 'tokens': torch.Tensor, shape (B, H, W) - discrete token indices
                - 'token_count': int - number of tokens
                - 'latent_shape': Tuple[int, int] - spatial shape
                - 'embeddings': torch.Tensor - continuous embeddings before quantization
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please install transformers: pip install transformers")
        
        # Preprocess image to 384x384
        if isinstance(image, Image.Image):
            image_tensor = self.preprocess(image, target_size=(384, 384))
        else:
            image_tensor = self.preprocess(image)
        
        # Convert to BF16 if enabled
        if self.use_bf16:
            image_tensor = image_tensor.to(torch.bfloat16)
        
        with torch.no_grad():
            # TA-Tok encoding
            output = self.model.encode(image_tensor)
            
            # Extract tokens and embeddings
            tokens = output.get('bottleneck_rep')  # discrete token indices
            embeddings = output.get('encoded')  # continuous embeddings
            regularized_z = output.get('regularized_z')  # quantized embeddings
        
        # Get spatial shape
        if tokens.ndim == 2:  # (B, N)
            B, N = tokens.shape
            H = W = int(N ** 0.5)
            latent_shape = (H, W)
        else:
            raise ValueError(f"Unexpected token shape: {tokens.shape}")
        
        return {
            'tokens': tokens,  # (B, N) discrete indices
            'indices': tokens,
            'token_count': N,
            'latent_shape': latent_shape,
            'embeddings': embeddings,  # (B, N, C) continuous
            'regularized_z': regularized_z  # (B, N, C) quantized
        }
    
    def decode(self, tokens: Union[torch.Tensor, Dict[str, Any]]) -> Image.Image:
        """
        TA-Tok 토큰을 이미지로 디코딩
        
        Note: TA-Tok은 인코더만 제공하며, 디코딩은 별도의 De-Tokenizer가 필요합니다:
        - AR-DTok: Autoregressive De-Tokenizer
        - Dif-DTok: Diffusion-based De-Tokenizer (SANA, Lumina2)
        
        Args:
            tokens: Dict with 'tokens' key or torch.Tensor (token indices)
        """
        raise NotImplementedError(
            "TA-Tok does not include a decoder. "
            "Please use a separate De-Tokenizer:\n"
            "  - AR-DTok: Use ARDTokDetokenizer\n"
            "  - Dif-DTok (SANA): Use SANADetokenizer\n"
            "  - Dif-DTok (Lumina2): Use Lumina2Detokenizer"
        )
    
    def info(self) -> Dict[str, Any]:
        """TA-Tok 토크나이저 정보 반환"""
        return {
            'name': 'TA-Tok (Text-Aligned Tokenizer)',
            'type': 'Text-Aligned-VQ',
            'codebook_size': self.codebook_size,  # 65536
            'encoder': 'SigLIP2-SO400M-patch14-384',
            'latent_channels': 1152,  # SigLIP2 embedding dim
            'latent_shape': (27, 27),  # 384px / 14 patch size
            'compression_ratio': 196.0,  # 384^2 / 27^2
            'num_parameters': self.count_parameters(),
            'input_size': (self.input_size, self.input_size),
            'pretrained': self.model is not None,
            'model_name': self.model_name,
            'description': 'Text-aligned tokenizer for unified visual understanding and generation',
            'paper': 'Vision as a Dialect (NeurIPS 2025)',
            'github': 'https://github.com/csuhan/Tar',
            'hf_repo': self.model_name,
            'auto_download': True,
            'detokenizers': {
                'AR-DTok': 'Autoregressive de-tokenizer (256/512/1024px)',
                'Dif-DTok': 'Diffusion-based de-tokenizer (SANA, Lumina2)'
            }
        }
    
    def _load_model(self, checkpoint_path: str):
        """체크포인트에서 모델 로드"""
        # TODO: 구현
        # checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # model = TATokModel(**checkpoint['config'])
        # model.load_state_dict(checkpoint['model_state_dict'])
        # return model
        pass
    
    def _load_pretrained(self, model_name: str):
        """사전학습된 모델 로드 (Hugging Face Hub 등)"""
        # TODO: 구현
        # from transformers import AutoModel
        # model = AutoModel.from_pretrained(f"tatok/{model_name}")
        # return model
        pass
    
    def visualize_token_allocation(self, image: Union[Image.Image, torch.Tensor], 
                                   encoded_result: Dict[str, Any]) -> Image.Image:
        """
        Adaptive token allocation 시각화
        
        Args:
            image: 원본 이미지
            encoded_result: encode() 메서드의 출력
        
        Returns:
            PIL.Image: token importance map이 오버레이된 이미지
        """
        # TODO: 구현
        # importance_map을 heatmap으로 변환하여 원본 이미지에 오버레이
        pass


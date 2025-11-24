"""
SANA Dif-DTok (Diffusion De-Tokenizer) Wrapper

이 모듈은 SANA 기반 Diffusion De-Tokenizer를 래핑합니다.

원리:
- Diffusion 방식으로 TA-Tok 토큰을 이미지로 디코딩
- SANA-600M 모델 사용
- 512px, 1024px 출력 지원

연동 방법:
- 공식 GitHub: https://github.com/csuhan/Tar
- Hugging Face Hub: csuhan/Tar-SANA-600M-512px, csuhan/Tar-SANA-600M-1024px
- 자동 다운로드 지원 (snapshot_download)

특징:
- Diffusion-based generation (high quality)
- Multiple resolution support (512/1024px)
- Text conditioning support
- Compatible with TA-Tok tokens

참고:
- 논문: "Vision as a Dialect" (NeurIPS 2025)
- GitHub: https://github.com/csuhan/Tar
- 자동 다운로드: ✅ (diffusers 사용)
"""

from typing import Dict, Any, Union, Optional
import torch
from PIL import Image
import numpy as np

from .base import VisionTokenizerBase


class SANADetokenizer(VisionTokenizerBase):
    """
    SANA Dif-DTok 기반 De-Tokenizer
    
    TA-Tok 토큰을 diffusion 방식으로 이미지로 디코딩합니다.
    """
    
    def __init__(self,
                 model_name: str = "csuhan/Tar-SANA-600M-1024px",
                 ta_tok_path: str = None,
                 device: str = "cuda",
                 use_hf: bool = True,
                 **kwargs):
        """
        Args:
            model_name: Hugging Face model name (csuhan/Tar-SANA-600M-512px or 1024px)
            ta_tok_path: TA-Tok 체크포인트 경로 (None이면 HF에서 자동 다운로드)
            device: 디바이스 ("cuda" or "cpu")
            use_hf: Hugging Face Hub에서 자동 다운로드 여부
        """
        super().__init__(device=device, **kwargs)
        self.model_name = model_name
        self.ta_tok_path = ta_tok_path
        self.use_hf = use_hf
        
        # SANA 설정
        self.codebook_size = 65536  # TA-Tok codebook size
        self.output_size = 1024 if "1024" in model_name else 512
        
        try:
            if use_hf:
                from huggingface_hub import snapshot_download, hf_hub_download
                
                # SANA 모델 다운로드
                sana_path = snapshot_download(model_name)
                print(f"✓ Downloaded SANA Dif-DTok ({self.output_size}px) from Hugging Face Hub")
                
                # TA-Tok 다운로드
                if ta_tok_path is None:
                    ta_tok_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth")
                    print(f"✓ Downloaded TA-Tok from Hugging Face Hub")
                
                self.sana_path = sana_path
                self.ta_tok_path = ta_tok_path
                
                # Load SANA pipeline
                import os
                from diffusers import SanaPipeline
                self.pipeline = SanaPipeline.from_pretrained(sana_path, torch_dtype=torch.float32).to(device)
                self.pipeline.text_encoder.to(torch.bfloat16)
                self.pipeline.transformer = self.pipeline.transformer.to(torch.bfloat16)
                
                # Load TA-Tok
                from tok.ta_tok import TextAlignedTokenizer
                self.ta_tok = TextAlignedTokenizer.from_checkpoint(
                    ta_tok_path,
                    load_teacher=False,
                    input_type='indices'
                ).to(device)
                self.ta_tok.eval()
                
                # Load negative prompt embeddings
                self.neg_embeds = torch.load(os.path.join(sana_path, 'negative_prompt_embeds.pth'), weights_only=False)[None].to(device)
                self.neg_attn_mask = torch.ones(self.neg_embeds.shape[:2], dtype=torch.int16, device=device)
                
                print(f"✓ Loaded SANA Dif-DTok ({self.output_size}px)")
            else:
                self.pipeline = None
                self.ta_tok = None
                
        except Exception as e:
            print(f"✗ Failed to load SANA Dif-DTok: {e}")
            print(f"  Install: pip install diffusers huggingface_hub")
            self.pipeline = None
            self.ta_tok = None
    
    def encode(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        SANA Dif-DTok은 디코더 전용이므로 encoding을 지원하지 않습니다.
        TA-Tok을 사용하여 이미지를 토큰으로 인코딩하세요.
        """
        raise NotImplementedError(
            "SANA Dif-DTok is a decoder-only model. "
            "Use TATokTokenizer to encode images to tokens."
        )
    
    def decode(self, 
               tokens: Union[torch.Tensor, Dict[str, Any]],
               prompt: Optional[str] = None,
               num_inference_steps: int = 50) -> Image.Image:
        """
        TA-Tok 토큰을 이미지로 디코딩
        
        Args:
            tokens: Dict with 'tokens' key or torch.Tensor (token indices from TA-Tok)
            prompt: Optional text prompt for conditioning
            num_inference_steps: Number of diffusion steps
        
        Returns:
            PIL.Image: 생성된 이미지 (output_size x output_size)
        """
        if self.pipeline is None or self.ta_tok is None:
            raise RuntimeError(
                "Models not loaded. Please check installation.\n"
                "Required: pip install diffusers huggingface_hub\n"
                "Full implementation: https://github.com/csuhan/Tar"
            )
        
        # Extract tokens
        if isinstance(tokens, dict):
            token_tensor = tokens['tokens']
        else:
            token_tensor = tokens
        
        with torch.no_grad():
            # Step 1: TA-Tok decodes indices to features
            caption_embeds = self.ta_tok.decode_from_bottleneck(token_tensor).to(torch.bfloat16)
            caption_embeds_mask = torch.ones(caption_embeds.shape[:2], dtype=torch.int16, device=caption_embeds.device)
            
            # Step 2: SANA generates image from features
            image = self.pipeline(
                prompt_embeds=caption_embeds,
                prompt_attention_mask=caption_embeds_mask,
                negative_prompt_embeds=self.neg_embeds,
                negative_prompt_attention_mask=self.neg_attn_mask,
                negative_prompt=None,
                height=self.output_size,
                width=self.output_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=1.0
            ).images[0]
            
            return image
    
    def info(self) -> Dict[str, Any]:
        """SANA Dif-DTok 디토크나이저 정보 반환"""
        return {
            'name': f'SANA Dif-DTok (Diffusion De-Tokenizer {self.output_size}px)',
            'type': 'Diffusion-Detokenizer',
            'input_codebook_size': self.codebook_size,  # TA-Tok tokens
            'output_size': (self.output_size, self.output_size),
            'architecture': 'SANA-600M',
            'num_parameters': self.count_parameters(),
            'pretrained': self.model_name,
            'checkpoint_path': getattr(self, 'sana_path', None),
            'ta_tok_path': self.ta_tok_path,
            'description': f'Diffusion-based de-tokenizer for {self.output_size}px image generation',
            'paper': 'Vision as a Dialect (NeurIPS 2025)',
            'github': 'https://github.com/csuhan/Tar',
            'hf_repo': self.model_name,
            'compatible_with': 'TA-Tok tokens',
            'auto_download': True,
            'supports_text_conditioning': True
        }






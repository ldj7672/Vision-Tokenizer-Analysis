"""
AR-DTok (Autoregressive De-Tokenizer) Wrapper

이 모듈은 AR-DTok De-Tokenizer를 래핑합니다.

원리:
- Autoregressive 방식으로 TA-Tok 토큰을 이미지로 디코딩
- LlamaGen 기반 VQ decoder 사용
- 256px, 512px, 1024px 출력 지원

연동 방법:
- 공식 GitHub: https://github.com/csuhan/Tar
- Hugging Face Hub: csuhan/TA-Tok
- 자동 다운로드 지원 (hf_hub_download)

특징:
- Autoregressive generation
- Multiple resolution support (256/512/1024px)
- Fast inference with caching
- Compatible with TA-Tok tokens

참고:
- 논문: "Vision as a Dialect" (NeurIPS 2025)
- GitHub: https://github.com/csuhan/Tar
- 자동 다운로드: ✅ (hf_hub_download 사용)
"""

from typing import Dict, Any, Union
import torch
from PIL import Image
import numpy as np

from .base import VisionTokenizerBase


class ARDTokDetokenizer(VisionTokenizerBase):
    """
    AR-DTok 기반 De-Tokenizer
    
    TA-Tok 토큰을 autoregressive 방식으로 이미지로 디코딩합니다.
    """
    
    def __init__(self,
                 ar_dtok_path: str = None,
                 vq_decoder_path: str = None,
                 ta_tok_path: str = None,
                 output_size: int = 1024,
                 device: str = "cuda",
                 use_hf: bool = True,
                 **kwargs):
        """
        Args:
            ar_dtok_path: AR-DTok 체크포인트 경로 (None이면 HF에서 자동 다운로드)
            vq_decoder_path: VQ decoder 경로 (None이면 HF에서 자동 다운로드)
            ta_tok_path: TA-Tok 체크포인트 경로 (None이면 HF에서 자동 다운로드)
            output_size: 출력 이미지 크기 (256, 512, 1024)
            device: 디바이스 ("cuda" or "cpu")
            use_hf: Hugging Face Hub에서 자동 다운로드 여부
        """
        super().__init__(device=device, **kwargs)
        self.ar_dtok_path = ar_dtok_path
        self.vq_decoder_path = vq_decoder_path
        self.ta_tok_path = ta_tok_path
        self.output_size = output_size
        self.use_hf = use_hf
        
        # AR-DTok 설정
        self.codebook_size = 65536  # TA-Tok codebook size
        
        try:
            # AR-DTok 모델 로드
            if use_hf:
                from huggingface_hub import hf_hub_download
                
                # AR-DTok 다운로드
                if ar_dtok_path is None:
                    if output_size == 256:
                        filename = "ar_dtok_lp_256px.pth"
                    elif output_size == 512:
                        filename = "ar_dtok_lp_512px.pth"
                    elif output_size == 1024:
                        filename = "ar_dtok_lp_1024px.pth"
                    else:
                        raise ValueError(f"Unsupported output size: {output_size}. Choose from [256, 512, 1024]")
                    
                    ar_dtok_path = hf_hub_download("csuhan/TA-Tok", filename)
                    print(f"✓ Downloaded AR-DTok ({output_size}px) from Hugging Face Hub")
                
                # VQ Decoder 다운로드
                if vq_decoder_path is None:
                    vq_decoder_path = hf_hub_download("peizesun/llamagen_t2i", "vq_ds16_t2i.pt")
                    print(f"✓ Downloaded VQ Decoder from Hugging Face Hub")
            
            # Load models
            if ar_dtok_path and vq_decoder_path:
                # Load AR model using from_checkpoint (recommended way)
                from tok.ar_dtok.ar_model import ARModel
                self.ar_model = ARModel.from_checkpoint(ar_dtok_path, load_state_dict=True)
                self.ar_model.to(device)
                self.ar_model.eval()
                
                # Load TA-Tok encoder (needed to convert indices to embeddings)
                from tok.ta_tok import TextAlignedTokenizer
                if self.ta_tok_path is None:
                    from huggingface_hub import hf_hub_download
                    self.ta_tok_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth")
                
                self.ta_tok = TextAlignedTokenizer.from_checkpoint(
                    self.ta_tok_path, load_teacher=False, input_type='indices'
                ).to(device)
                self.ta_tok.eval()
                
                # Build VQ Decoder
                from tok.ar_dtok.vqvae import VQVAE
                self.vq_decoder = VQVAE.from_checkpoint(
                    ckpt=vq_decoder_path,
                    codebook_size=16384,
                    codebook_embed_dim=8,
                    bottleneck_token_num=256,
                    input_size=output_size
                ).to(device)
                self.vq_decoder.eval()
                
                print(f"✓ Loaded AR-DTok ({output_size}px, model_type: {self.ar_model.model_type})")
                print(f"✓ Loaded TA-Tok encoder")
                print(f"✓ Loaded VQ Decoder")
            else:
                self.ar_model = None
                self.vq_decoder = None
                self.ta_tok = None
                
        except Exception as e:
            print(f"✗ Failed to load AR-DTok: {e}")
            print(f"  Install: pip install huggingface_hub")
            import traceback
            traceback.print_exc()
            self.ar_model = None
            self.vq_decoder = None
            self.ta_tok = None
    
    def encode(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        AR-DTok은 디코더 전용이므로 encoding을 지원하지 않습니다.
        TA-Tok을 사용하여 이미지를 토큰으로 인코딩하세요.
        """
        raise NotImplementedError(
            "AR-DTok is a decoder-only model. "
            "Use TATokTokenizer to encode images to tokens."
        )
    
    def decode(self, tokens: Union[torch.Tensor, Dict[str, Any]]) -> Image.Image:
        """
        TA-Tok 토큰을 이미지로 디코딩
        
        Args:
            tokens: Dict with 'tokens' key or torch.Tensor (token indices from TA-Tok)
        
        Returns:
            PIL.Image: 생성된 이미지 (output_size x output_size)
        """
        if self.ar_model is None or self.vq_decoder is None or self.ta_tok is None:
            raise RuntimeError(
                "Models not loaded. Please check installation.\n"
                "Required: pip install huggingface_hub\n"
                "Full implementation: https://github.com/csuhan/Tar"
            )
        
        # Extract tokens
        if isinstance(tokens, dict):
            token_tensor = tokens['tokens']
        else:
            token_tensor = tokens
        
        with torch.no_grad():
            # Step 1: Convert TA-Tok indices to continuous embeddings
            caption_embeds = self.ta_tok.decode_from_bottleneck(token_tensor)
            
            # Step 2: AR model generates VQ image tokens from embeddings
            image_tokens = self.ar_model.sample(
                c=caption_embeds,
                cfg_scale=1.0,
                cfg_interval=-1,
                temperature=1.0,
                top_k=200,
                top_p=1.0,
            )
            
            # Step 3: VQ Decoder decodes image tokens to image
            image_tensor = self.vq_decoder.decode_from_bottleneck(image_tokens)
            
            # Image is already in [0, 1] range from VQVAE
            image_tensor = image_tensor.clamp(0, 1)
            
            return self.postprocess(image_tensor)
    
    def info(self) -> Dict[str, Any]:
        """AR-DTok 디토크나이저 정보 반환"""
        return {
            'name': f'AR-DTok (Autoregressive De-Tokenizer {self.output_size}px)',
            'type': 'Autoregressive-Detokenizer',
            'input_codebook_size': self.codebook_size,  # TA-Tok tokens
            'output_size': (self.output_size, self.output_size),
            'architecture': 'LlamaGen-based',
            'vq_decoder': 'vq_ds16_t2i',
            'num_parameters': self.count_parameters(),
            'pretrained': self.ar_dtok_path is not None,
            'checkpoint_path': self.ar_dtok_path,
            'vq_decoder_path': self.vq_decoder_path,
            'description': f'Autoregressive de-tokenizer for {self.output_size}px image generation',
            'paper': 'Vision as a Dialect (NeurIPS 2025)',
            'github': 'https://github.com/csuhan/Tar',
            'hf_repo': 'csuhan/TA-Tok',
            'compatible_with': 'TA-Tok tokens',
            'auto_download': True
        }






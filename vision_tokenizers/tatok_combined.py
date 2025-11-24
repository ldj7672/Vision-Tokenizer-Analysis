"""
TA-Tok Combined Tokenizer (TA-Tok + De-Tokenizer)

이 모듈은 TA-Tok 인코더와 De-Tokenizer를 결합한 통합 tokenizer를 제공합니다.

조합:
- TA-Tok + AR-DTok: Autoregressive de-tokenization
- TA-Tok + SANA: Diffusion-based de-tokenization
- TA-Tok + Lumina2: Diffusion-based de-tokenization (highest quality)
"""

from typing import Dict, Any, Union
import torch
from PIL import Image
import numpy as np

from .base import VisionTokenizerBase
from .tatok import TATokTokenizer
from .ar_dtok import ARDTokDetokenizer
from .sana_dtok import SANADetokenizer
from .lumina2_dtok import Lumina2Detokenizer


class TATokARDTokTokenizer(VisionTokenizerBase):
    """TA-Tok + AR-DTok 조합"""
    
    def __init__(self,
                 ta_tok_checkpoint: str = None,
                 ta_tok_model: str = "ByteDance-Seed/Tar-TA-Tok",
                 ar_dtok_size: int = 1024,
                 device: str = "cuda",
                 use_bf16: bool = False,
                 use_compile: bool = False,
                 **kwargs):
        super().__init__(device=device, use_bf16=use_bf16, use_compile=use_compile, **kwargs)
        
        # TA-Tok (encoder)
        self.encoder = TATokTokenizer(
            checkpoint_path=ta_tok_checkpoint,
            model_name=ta_tok_model,
            device=device,
            use_bf16=use_bf16,
            use_compile=use_compile
        )
        
        # AR-DTok (decoder)
        self.decoder = ARDTokDetokenizer(
            ta_tok_path=ta_tok_checkpoint,
            output_size=ar_dtok_size,
            device=device,
            use_hf=True,
            use_bf16=use_bf16,
            use_compile=use_compile
        )
        
        self.codebook_size = 65536
    
    def encode(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """TA-Tok으로 인코딩"""
        return self.encoder.encode(image)
    
    def decode(self, tokens: Union[torch.Tensor, Dict[str, Any]]) -> Image.Image:
        """AR-DTok으로 디코딩"""
        return self.decoder.decode(tokens)
    
    def info(self) -> Dict[str, Any]:
        encoder_info = self.encoder.info()
        decoder_info = self.decoder.info()
        
        return {
            'name': f'TA-Tok + AR-DTok ({self.decoder.output_size}px)',
            'type': 'Combined (Encoder + Decoder)',
            'encoder': encoder_info['name'],
            'decoder': decoder_info['name'],
            'codebook_size': self.codebook_size,
            'output_size': (self.decoder.output_size, self.decoder.output_size),
            'auto_download': True,
            'description': 'TA-Tok encoder with AR-DTok autoregressive decoder'
        }


class TATokSANATokenizer(VisionTokenizerBase):
    """TA-Tok + SANA Dif-DTok 조합"""
    
    def __init__(self,
                 ta_tok_checkpoint: str = None,
                 ta_tok_model: str = "ByteDance-Seed/Tar-TA-Tok",
                 sana_model: str = "csuhan/Tar-SANA-600M-1024px",
                 device: str = "cuda",
                 use_bf16: bool = False,
                 use_compile: bool = False,
                 **kwargs):
        super().__init__(device=device, use_bf16=use_bf16, use_compile=use_compile, **kwargs)
        
        # TA-Tok (encoder)
        self.encoder = TATokTokenizer(
            checkpoint_path=ta_tok_checkpoint,
            model_name=ta_tok_model,
            device=device,
            use_bf16=use_bf16,
            use_compile=use_compile
        )
        
        # SANA Dif-DTok (decoder)
        self.decoder = SANADetokenizer(
            model_name=sana_model,
            device=device,
            use_hf=True,
            use_bf16=use_bf16,
            use_compile=use_compile
        )
        
        self.codebook_size = 65536
    
    def encode(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """TA-Tok으로 인코딩"""
        return self.encoder.encode(image)
    
    def decode(self, tokens: Union[torch.Tensor, Dict[str, Any]]) -> Image.Image:
        """SANA Dif-DTok으로 디코딩"""
        return self.decoder.decode(tokens)
    
    def info(self) -> Dict[str, Any]:
        encoder_info = self.encoder.info()
        decoder_info = self.decoder.info()
        
        return {
            'name': f'TA-Tok + SANA Dif-DTok ({self.decoder.output_size}px)',
            'type': 'Combined (Encoder + Decoder)',
            'encoder': encoder_info['name'],
            'decoder': decoder_info['name'],
            'codebook_size': self.codebook_size,
            'output_size': (self.decoder.output_size, self.decoder.output_size),
            'auto_download': True,
            'description': 'TA-Tok encoder with SANA diffusion-based decoder'
        }


class TATokLumina2Tokenizer(VisionTokenizerBase):
    """TA-Tok + Lumina2 Dif-DTok 조합 🌟"""
    
    def __init__(self,
                 ta_tok_checkpoint: str = None,
                 ta_tok_model: str = "ByteDance-Seed/Tar-TA-Tok",
                 lumina2_model: str = "csuhan/Tar-Lumina2",
                 device: str = "cuda",
                 use_bf16: bool = False,
                 use_compile: bool = False,
                 **kwargs):
        super().__init__(device=device, use_bf16=use_bf16, use_compile=use_compile, **kwargs)
        
        # TA-Tok (encoder)
        self.encoder = TATokTokenizer(
            checkpoint_path=ta_tok_checkpoint,
            model_name=ta_tok_model,
            device=device,
            use_bf16=use_bf16,
            use_compile=use_compile
        )
        
        # Lumina2 Dif-DTok (decoder)
        self.decoder = Lumina2Detokenizer(
            model_name=lumina2_model,
            device=device,
            use_hf=True,
            use_bf16=use_bf16,
            use_compile=use_compile
        )
        
        self.codebook_size = 65536
    
    def encode(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """TA-Tok으로 인코딩"""
        return self.encoder.encode(image)
    
    def decode(self, tokens: Union[torch.Tensor, Dict[str, Any]]) -> Image.Image:
        """Lumina2 Dif-DTok으로 디코딩"""
        return self.decoder.decode(tokens)
    
    def info(self) -> Dict[str, Any]:
        encoder_info = self.encoder.info()
        decoder_info = self.decoder.info()
        
        return {
            'name': f'TA-Tok + Lumina2 Dif-DTok (1024px) 🌟',
            'type': 'Combined (Encoder + Decoder)',
            'encoder': encoder_info['name'],
            'decoder': decoder_info['name'],
            'codebook_size': self.codebook_size,
            'output_size': (1024, 1024),
            'auto_download': True,
            'recommended': True,
            'description': 'TA-Tok encoder with Lumina2 diffusion-based decoder (highest quality, 2.6B params)'
        }






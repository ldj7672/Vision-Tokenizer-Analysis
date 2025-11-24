"""
Lumina2 Dif-DTok (Diffusion De-Tokenizer) Wrapper

이 모듈은 Lumina2 기반 Diffusion De-Tokenizer를 래핑합니다.

원리:
- Diffusion 방식으로 TA-Tok 토큰을 이미지로 디코딩
- Lumina2-2.6B 모델 사용 (SANA보다 큰 모델)
- 1024px 고품질 출력

연동 방법:
- 공식 GitHub: https://github.com/csuhan/Tar
- Hugging Face Hub: csuhan/Tar-Lumina2
- 자동 다운로드 지원 (snapshot_download)

특징:
- Diffusion-based generation (highest quality)
- 1024px high-resolution output
- Text conditioning support
- 2.6B parameters (larger than SANA)
- Compatible with TA-Tok tokens

참고:
- 논문: "Vision as a Dialect" (NeurIPS 2025)
- GitHub: https://github.com/csuhan/Tar
- 자동 다운로드: ✅ (diffusers 사용)
- 추천 모델: 🌟 (Tar 논문 권장)
"""

from typing import Dict, Any, Union, Optional
import torch
from PIL import Image
import numpy as np

from .base import VisionTokenizerBase


class Lumina2Detokenizer(VisionTokenizerBase):
    """
    Lumina2 Dif-DTok 기반 De-Tokenizer
    
    TA-Tok 토큰을 diffusion 방식으로 고품질 이미지로 디코딩합니다.
    """
    
    def __init__(self,
                 model_name: str = "csuhan/Tar-Lumina2",
                 ta_tok_path: str = None,
                 device: str = "cuda",
                 use_hf: bool = True,
                 **kwargs):
        """
        Args:
            model_name: Hugging Face model name
            ta_tok_path: TA-Tok 체크포인트 경로 (None이면 HF에서 자동 다운로드)
            device: 디바이스 ("cuda" or "cpu")
            use_hf: Hugging Face Hub에서 자동 다운로드 여부
        """
        super().__init__(device=device, **kwargs)
        self.model_name = model_name
        self.ta_tok_path = ta_tok_path
        self.use_hf = use_hf
        
        # Lumina2 설정
        self.codebook_size = 65536  # TA-Tok codebook size
        self.output_size = 1024
        self.num_params = 2.6e9  # 2.6B parameters
        
        try:
            if use_hf:
                from huggingface_hub import snapshot_download, hf_hub_download
                
                # Lumina2 모델 다운로드
                lumina2_path = snapshot_download(model_name)
                print(f"✓ Downloaded Lumina2 Dif-DTok from Hugging Face Hub")
                
                # TA-Tok 다운로드
                if ta_tok_path is None:
                    ta_tok_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth")
                    print(f"✓ Downloaded TA-Tok from Hugging Face Hub")
                
                self.lumina2_path = lumina2_path
                self.ta_tok_path = ta_tok_path
                
                # Load models
                import os
                from diffusers.models import AutoencoderKL
                from transformers import AutoModel, AutoTokenizer
                import tok.lumina2_model as models
                from tok.transport import Sampler, create_transport
                from tok.ta_tok import TextAlignedTokenizer
                from tok.utils import ScalingLayer
                
                dtype = torch.float32  # Use float32 instead of bfloat16 to avoid dtype mismatch
                
                # Load tokenizer & text encoder from local cache
                import os
                gemma_cache_dir = os.path.join(
                    os.environ.get('HF_HOME', './model_weights/huggingface'),
                    'models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf'
                )
                
                if os.path.exists(gemma_cache_dir):
                    # Use local cached files
                    self.tokenizer = AutoTokenizer.from_pretrained(gemma_cache_dir)
                    self.text_encoder = AutoModel.from_pretrained(
                        gemma_cache_dir, 
                        torch_dtype=dtype, 
                        device_map=device
                    ).eval()
                else:
                    # Fallback to online download (requires HF login)
                    self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
                    self.text_encoder = AutoModel.from_pretrained(
                        "google/gemma-2-2b", 
                        torch_dtype=dtype, 
                        device_map=device
                    ).eval()
                
                self.tokenizer.padding_side = "right"
                
                # Load VAE from local cache
                flux_cache_dir = os.path.join(
                    os.environ.get('HF_HOME', './model_weights/huggingface'),
                    'models--black-forest-labs--FLUX.1-dev/snapshots'
                )
                
                # Find the latest snapshot
                if os.path.exists(flux_cache_dir):
                    snapshots = [d for d in os.listdir(flux_cache_dir) if os.path.isdir(os.path.join(flux_cache_dir, d))]
                    if snapshots:
                        flux_vae_path = os.path.join(flux_cache_dir, snapshots[0], "vae")
                        self.vae = AutoencoderKL.from_pretrained(
                            flux_vae_path, torch_dtype=dtype
                        ).to(device)
                    else:
                        # Fallback to online download
                        self.vae = AutoencoderKL.from_pretrained(
                            "black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype
                        ).to(device)
                else:
                    # Fallback to online download
                    self.vae = AutoencoderKL.from_pretrained(
                        "black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype
                    ).to(device)
                self.vae.requires_grad_(False)
                
                # Load TA-Tok
                self.ta_tok = TextAlignedTokenizer.from_checkpoint(
                    ta_tok_path, load_teacher=False, input_type='rec'
                )
                self.ta_tok.scale_layer = ScalingLayer(mean=[0., 0., 0.], std=[1.0, 1.0, 1.0])
                self.ta_tok.eval().to(device=device, dtype=dtype)
                
                # Load Lumina2 model
                train_args = torch.load(os.path.join(lumina2_path, "model_args.pth"), weights_only=False)
                self.model = models.__dict__[train_args.model](
                    in_channels=16,
                    cond_in_channels=1152,
                    qk_norm=train_args.qk_norm,
                    cap_feat_dim=self.text_encoder.config.hidden_size,
                )
                self.model.eval().to(device, dtype=dtype)
                
                # Load checkpoint
                ckpt = torch.load(
                    os.path.join(lumina2_path, "consolidated_ema.00-of-01.pth"),
                    map_location='cpu',
                    weights_only=False
                )
                self.model.load_state_dict(ckpt, strict=True)
                
                # Setup sampler
                self.transport = create_transport("Linear", "velocity", None, None, None)
                self.sampler = Sampler(self.transport)
                
                # Default prompt for reconstruction
                self.default_prompt = "You are an assistant designed to generate superior images with the highest degree of image alignment based on the SigLIP2 features of an original image. <Prompt Start> "
                
                print(f"✓ Loaded Lumina2 Dif-DTok (1024px, 2.6B params) 🌟")
            else:
                self.pipeline = None
                self.ta_tok = None
                
        except Exception as e:
            print(f"✗ Failed to load Lumina2 Dif-DTok: {e}")
            print(f"  Install: pip install diffusers huggingface_hub")
            import traceback
            traceback.print_exc()
            self.pipeline = None
            self.ta_tok = None
            self.model = None
            self.vae = None
            self.text_encoder = None
            self.tokenizer = None
    
    def encode(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        Lumina2 Dif-DTok은 디코더 전용이므로 encoding을 지원하지 않습니다.
        TA-Tok을 사용하여 이미지를 토큰으로 인코딩하세요.
        """
        raise NotImplementedError(
            "Lumina2 Dif-DTok is a decoder-only model. "
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
            PIL.Image: 생성된 이미지 (1024x1024)
        """
        if not hasattr(self, 'model') or self.model is None or self.ta_tok is None:
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
            import torch.nn.functional as F
            from torchvision.transforms.functional import to_pil_image
            
            dtype = torch.bfloat16
            
            # Use default prompt if not provided
            if prompt is None:
                prompt = self.default_prompt
            
            # Encode text prompt
            text_inputs = self.tokenizer(
                [prompt],
                padding=True,
                pad_to_multiple_of=8,
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.device)
            prompt_masks = text_inputs.attention_mask.to(self.device)
            
            cap_feats = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=prompt_masks,
                output_hidden_states=True,
            ).hidden_states[-2]
            
            # Preprocess condition (TA-Tok tokens -> features)
            cond_in = self.ta_tok.decode_from_bottleneck(token_tensor.to(self.device))
            B, L, D = cond_in.shape
            H = W = int(L**0.5)
            cond_in = cond_in.permute(0, 2, 1).reshape(B, D, H, W)
            cond_in = F.interpolate(cond_in, size=(128, 128), mode='bilinear')
            cond_ins_all = [[cond_in[0].to(dtype)]]  # Convert to bfloat16
            
            # Prepare latent
            latent_w = latent_h = self.output_size // 8
            z = torch.randn([1, 16, latent_w, latent_h], device=self.device, dtype=dtype)
            
            # Model kwargs
            model_kwargs = dict(
                cap_feats=cap_feats,
                cap_mask=prompt_masks.to(cap_feats.device),
                cond=cond_ins_all,
                position_type=[["aligned"]],
            )
            
            # Sample with ODE
            sample_fn = self.sampler.sample_ode(
                sampling_method="euler",
                num_steps=num_inference_steps,
                atol=1e-6,
                rtol=1e-3,
                reverse=False,
                time_shifting_factor=6.0,
            )
            samples = sample_fn(z, self.model.forward, **model_kwargs)[-1]
            
            # Decode with VAE
            samples = samples[:1]
            samples = self.vae.decode(samples / self.vae.config.scaling_factor + self.vae.config.shift_factor)[0]
            samples = (samples + 1.0) / 2.0
            samples.clamp_(0.0, 1.0)
            
            img = to_pil_image(samples[0].float())
            return img
    
    def info(self) -> Dict[str, Any]:
        """Lumina2 Dif-DTok 디토크나이저 정보 반환"""
        return {
            'name': 'Lumina2 Dif-DTok (Diffusion De-Tokenizer 1024px) 🌟',
            'type': 'Diffusion-Detokenizer',
            'input_codebook_size': self.codebook_size,  # TA-Tok tokens
            'output_size': (self.output_size, self.output_size),
            'architecture': 'Lumina2-2.6B',
            'num_parameters': int(self.num_params),
            'pretrained': self.model_name,
            'checkpoint_path': getattr(self, 'lumina2_path', None),
            'ta_tok_path': self.ta_tok_path,
            'description': 'Diffusion-based de-tokenizer for 1024px high-quality image generation',
            'paper': 'Vision as a Dialect (NeurIPS 2025)',
            'github': 'https://github.com/csuhan/Tar',
            'hf_repo': self.model_name,
            'compatible_with': 'TA-Tok tokens',
            'auto_download': True,
            'supports_text_conditioning': True,
            'recommended': True  # 🌟 Tar 논문 권장 모델
        }






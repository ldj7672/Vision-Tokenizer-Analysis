"""
Vision Tokenizer Base Interface

이 모듈은 모든 Vision Tokenizer가 구현해야 하는 공통 인터페이스를 정의합니다.
다양한 토크나이저(VAE, VQ-KD, TA-Tok 등)를 통일된 방식으로 사용할 수 있도록 합니다.

주요 기능:
- encode: 이미지를 토큰으로 변환
- decode: 토큰을 이미지로 복원
- info: 토크나이저의 메타정보 반환 (codebook size, latent shape, 파라미터 수 등)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple
import torch
from PIL import Image
import numpy as np


class VisionTokenizerBase(ABC):
    """
    모든 Vision Tokenizer의 기본 인터페이스
    
    이 추상 클래스를 상속받아 구체적인 토크나이저를 구현합니다.
    각 토크나이저는 encode, decode, info 메서드를 반드시 구현해야 합니다.
    """
    
    def __init__(self, device: str = "cuda", use_bf16: bool = False, use_compile: bool = False, **kwargs):
        """
        Args:
            device: 모델을 로드할 디바이스 ("cuda" 또는 "cpu")
            use_bf16: BFloat16 사용 여부 (메모리 절약, 속도 향상)
            use_compile: torch.compile 사용 여부 (PyTorch 2.0+, 속도 향상)
            **kwargs: 토크나이저별 추가 설정
        """
        self.device = device
        self.use_bf16 = use_bf16
        self.use_compile = use_compile
        self.model = None
        self.dtype = torch.bfloat16 if use_bf16 else torch.float32
        
    @abstractmethod
    def encode(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        이미지를 토큰으로 인코딩
        
        Args:
            image: 입력 이미지 (PIL Image, torch.Tensor, 또는 numpy array)
                   - PIL Image: RGB 형식
                   - torch.Tensor: shape (B, C, H, W) 또는 (C, H, W), 값 범위 [0, 1] 또는 [-1, 1]
                   - numpy array: shape (H, W, C), 값 범위 [0, 255]
        
        Returns:
            Dict containing:
                - 'tokens': torch.Tensor, shape (B, N) 또는 (B, H, W) - 이산 토큰 인덱스
                - 'token_count': int - 토큰 개수
                - 'latent_shape': Tuple[int, ...] - latent representation의 shape (H, W) 또는 (H, W, D)
                - 'latent': torch.Tensor (optional) - 연속적인 latent representation
                - 'indices': torch.Tensor (optional) - codebook indices (VQ 기반 모델용)
                - 'quantized': torch.Tensor (optional) - quantized latent (VQ 기반 모델용)
        
        Example:
            >>> result = tokenizer.encode(image)
            >>> print(result['tokens'].shape)  # (1, 256)
            >>> print(result['token_count'])    # 256
            >>> print(result['latent_shape'])   # (16, 16)
        """
        pass
    
    @abstractmethod
    def decode(self, tokens: Union[torch.Tensor, Dict[str, Any]]) -> Image.Image:
        """
        토큰을 이미지로 디코딩
        
        Args:
            tokens: 토큰 정보
                   - torch.Tensor: shape (B, N) 또는 (B, H, W) - 토큰 인덱스
                   - Dict: encode() 메서드의 출력 결과 (latent, quantized 등 포함 가능)
        
        Returns:
            PIL.Image: 복원된 RGB 이미지
        
        Example:
            >>> encoded = tokenizer.encode(image)
            >>> reconstructed = tokenizer.decode(encoded['tokens'])
            >>> reconstructed.save('output.png')
        """
        pass
    
    @abstractmethod
    def info(self) -> Dict[str, Any]:
        """
        토크나이저의 메타정보 반환
        
        Returns:
            Dict containing:
                - 'name': str - 토크나이저 이름
                - 'type': str - 토크나이저 타입 (예: 'VAE', 'VQ-VAE', 'VQ-KD', 'TA-Tok')
                - 'codebook_size': int - codebook 크기 (VQ 기반 모델의 경우)
                - 'latent_channels': int - latent space의 채널 수
                - 'latent_shape': Tuple[int, int] - 기본 latent shape (H, W)
                - 'compression_ratio': float - 압축률 (원본 대비)
                - 'num_parameters': int - 모델 파라미터 수
                - 'input_size': Tuple[int, int] - 기본 입력 이미지 크기 (H, W)
                - 'pretrained': bool - 사전학습된 가중치 사용 여부
                - 'checkpoint_path': str (optional) - 체크포인트 경로
        
        Example:
            >>> info = tokenizer.info()
            >>> print(f"Tokenizer: {info['name']}")
            >>> print(f"Codebook size: {info['codebook_size']}")
            >>> print(f"Compression ratio: {info['compression_ratio']:.2f}x")
        """
        pass
    
    def preprocess(self, image: Union[Image.Image, torch.Tensor, np.ndarray], target_size: tuple = None) -> torch.Tensor:
        """
        이미지를 모델 입력 형식으로 전처리
        
        Args:
            image: 입력 이미지
            target_size: (H, W) 리사이즈 타겟 크기 (None이면 리사이즈 안함)
        
        Returns:
            torch.Tensor: shape (B, C, H, W), 값 범위는 모델에 따라 다름
        
        TODO: 각 토크나이저에서 필요시 오버라이드
        """
        if isinstance(image, Image.Image):
            # Resize to target size if specified
            if target_size is not None:
                image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)
            
            # PIL Image -> Tensor
            image = np.array(image)
            if image.ndim == 2:  # Grayscale
                image = np.stack([image] * 3, axis=-1)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image = image.unsqueeze(0)  # Add batch dimension
        elif isinstance(image, np.ndarray):
            # Numpy array -> Tensor
            if image.ndim == 3 and image.shape[-1] == 3:  # (H, W, C)
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                image = image.unsqueeze(0)
            elif image.ndim == 2:  # Grayscale (H, W)
                image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1).float() / 255.0
                image = image.unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            # Tensor: ensure batch dimension
            if image.ndim == 3:  # (C, H, W)
                image = image.unsqueeze(0)
        
        return image.to(self.device)
    
    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """
        모델 출력을 PIL Image로 변환
        
        Args:
            tensor: shape (B, C, H, W) 또는 (C, H, W)
        
        Returns:
            PIL.Image: RGB 이미지
        
        TODO: 각 토크나이저에서 필요시 오버라이드
        """
        if tensor.ndim == 4:
            tensor = tensor[0]  # Remove batch dimension
        
        # Clamp to [0, 1] or [-1, 1] depending on model output
        # Check the actual range of the tensor
        if tensor.min() < -0.5:  # Likely [-1, 1] range
            tensor = (tensor + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        
        tensor = torch.clamp(tensor, 0, 1)
        
        # Tensor -> numpy -> PIL
        # Convert BF16 to float32 before converting to numpy (numpy doesn't support BF16)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        image_np = (tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)
    
    def count_parameters(self) -> int:
        """
        모델의 총 파라미터 수 계산
        
        Returns:
            int: 파라미터 수
        """
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def __repr__(self) -> str:
        info = self.info()
        return f"{self.__class__.__name__}(name='{info.get('name', 'Unknown')}', type='{info.get('type', 'Unknown')}')"


# TODO: 추가 유틸리티 함수들
def calculate_compression_ratio(input_shape: Tuple[int, ...], 
                                latent_shape: Tuple[int, ...],
                                codebook_size: int = None) -> float:
    """
    압축률 계산
    
    Args:
        input_shape: 입력 이미지 shape (C, H, W)
        latent_shape: latent representation shape (H', W') 또는 (H', W', D)
        codebook_size: codebook 크기 (VQ 기반 모델의 경우)
    
    Returns:
        float: 압축률 (원본 대비 몇 배 압축되었는지)
    """
    # 입력 이미지의 비트 수 (RGB 8-bit per channel)
    input_bits = input_shape[0] * input_shape[1] * input_shape[2] * 8
    
    # Latent의 비트 수
    if codebook_size is not None:
        # VQ 기반: 각 토큰은 log2(codebook_size) 비트
        import math
        bits_per_token = math.log2(codebook_size)
        latent_bits = latent_shape[0] * latent_shape[1] * bits_per_token
    else:
        # VAE 등: 연속 값 (32-bit float 가정)
        if len(latent_shape) == 2:
            latent_bits = latent_shape[0] * latent_shape[1] * 32
        else:
            latent_bits = latent_shape[0] * latent_shape[1] * latent_shape[2] * 32
    
    return input_bits / latent_bits



"""
Reconstruction Quality Metrics

이 모듈은 이미지 재구성 품질을 평가하는 다양한 메트릭을 제공합니다.

지원 메트릭:
1. PSNR (Peak Signal-to-Noise Ratio)
   - 픽셀 단위 재구성 오차 측정
   - 높을수록 좋음 (일반적으로 20-40 dB)
   - 수식: 10 * log10(MAX^2 / MSE)

2. SSIM (Structural Similarity Index)
   - 구조적 유사도 측정 (luminance, contrast, structure)
   - 범위: [-1, 1], 1에 가까울수록 좋음
   - 인간의 시각적 인지와 높은 상관관계

3. LPIPS (Learned Perceptual Image Patch Similarity)
   - 딥러닝 기반 perceptual distance
   - 낮을수록 좋음 (0에 가까울수록 유사)
   - VGG, AlexNet 등의 feature space에서 거리 측정

4. MS-SSIM (Multi-Scale SSIM)
   - 여러 스케일에서 SSIM 계산
   - 다양한 해상도에서의 품질 평가

사용법:
    from metrics.reconstruction import calculate_psnr, calculate_ssim, calculate_lpips
    
    psnr = calculate_psnr(original, reconstructed)
    ssim = calculate_ssim(original, reconstructed)
    lpips = calculate_lpips(original, reconstructed, net='alex')
"""

from typing import Union, Tuple
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def calculate_psnr(img1: Union[torch.Tensor, np.ndarray, Image.Image],
                   img2: Union[torch.Tensor, np.ndarray, Image.Image],
                   max_value: float = 1.0) -> float:
    """
    PSNR (Peak Signal-to-Noise Ratio) 계산
    
    Args:
        img1: 원본 이미지
        img2: 재구성된 이미지
        max_value: 이미지의 최대 픽셀 값 (1.0 for [0,1], 255 for [0,255])
    
    Returns:
        float: PSNR 값 (dB)
    
    Example:
        >>> psnr = calculate_psnr(original, reconstructed)
        >>> print(f"PSNR: {psnr:.2f} dB")
    """
    # Convert to tensor
    tensor1 = _to_tensor(img1)
    tensor2 = _to_tensor(img2)
    
    # Calculate MSE
    mse = F.mse_loss(tensor1, tensor2)
    
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 10 * torch.log10((max_value ** 2) / mse)
    return psnr.item()


def calculate_ssim(img1: Union[torch.Tensor, np.ndarray, Image.Image],
                   img2: Union[torch.Tensor, np.ndarray, Image.Image],
                   window_size: int = 11,
                   channel: int = 3) -> float:
    """
    SSIM (Structural Similarity Index) 계산
    
    Args:
        img1: 원본 이미지
        img2: 재구성된 이미지
        window_size: Gaussian window 크기
        channel: 채널 수
    
    Returns:
        float: SSIM 값 [0, 1]
    
    Example:
        >>> ssim = calculate_ssim(original, reconstructed)
        >>> print(f"SSIM: {ssim:.4f}")
    """
    try:
        # Option 1: Use pytorch-msssim library
        from pytorch_msssim import ssim
        tensor1 = _to_tensor(img1)
        tensor2 = _to_tensor(img2)
        ssim_val = ssim(tensor1, tensor2, data_range=1.0, size_average=True)
        return ssim_val.item()
    except ImportError:
        try:
            # Option 2: Use scikit-image
            from skimage.metrics import structural_similarity
            arr1 = _to_numpy(img1)
            arr2 = _to_numpy(img2)
            ssim_val = structural_similarity(arr1, arr2, multichannel=True, channel_axis=-1)
            return float(ssim_val)
        except ImportError:
            raise ImportError("Please install pytorch-msssim or scikit-image: pip install pytorch-msssim")


def calculate_lpips(img1: Union[torch.Tensor, np.ndarray, Image.Image],
                    img2: Union[torch.Tensor, np.ndarray, Image.Image],
                    net: str = 'alex',
                    device: str = 'cuda') -> float:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) 계산
    
    Args:
        img1: 원본 이미지
        img2: 재구성된 이미지
        net: 사용할 네트워크 ('alex', 'vgg', 'squeeze')
        device: 디바이스
    
    Returns:
        float: LPIPS distance (낮을수록 유사)
    
    Example:
        >>> lpips = calculate_lpips(original, reconstructed, net='alex')
        >>> print(f"LPIPS: {lpips:.4f}")
    """
    try:
        import lpips
        
        # Initialize LPIPS model
        loss_fn = lpips.LPIPS(net=net).to(device)
        
        # Convert to tensor [-1, 1]
        tensor1 = _to_tensor(img1, normalize=True).to(device)
        tensor2 = _to_tensor(img2, normalize=True).to(device)
        
        with torch.no_grad():
            distance = loss_fn(tensor1, tensor2)
        
        return distance.item()
    except ImportError:
        raise ImportError("Please install lpips: pip install lpips")


def calculate_ms_ssim(img1: Union[torch.Tensor, np.ndarray, Image.Image],
                      img2: Union[torch.Tensor, np.ndarray, Image.Image]) -> float:
    """
    MS-SSIM (Multi-Scale SSIM) 계산
    
    Args:
        img1: 원본 이미지
        img2: 재구성된 이미지
    
    Returns:
        float: MS-SSIM 값 [0, 1]
    """
    try:
        from pytorch_msssim import ms_ssim
        tensor1 = _to_tensor(img1)
        tensor2 = _to_tensor(img2)
        ms_ssim_val = ms_ssim(tensor1, tensor2, data_range=1.0, size_average=True)
        return ms_ssim_val.item()
    except ImportError:
        raise ImportError("Please install pytorch-msssim: pip install pytorch-msssim")


def calculate_all_metrics(img1: Union[torch.Tensor, np.ndarray, Image.Image],
                          img2: Union[torch.Tensor, np.ndarray, Image.Image],
                          device: str = 'cuda') -> dict:
    """
    모든 재구성 메트릭을 한 번에 계산
    
    Args:
        img1: 원본 이미지
        img2: 재구성된 이미지
        device: 디바이스
    
    Returns:
        dict: 모든 메트릭 값
            - 'psnr': float
            - 'ssim': float
            - 'lpips': float
            - 'ms_ssim': float
    
    Example:
        >>> metrics = calculate_all_metrics(original, reconstructed)
        >>> print(f"PSNR: {metrics['psnr']:.2f} dB")
        >>> print(f"SSIM: {metrics['ssim']:.4f}")
        >>> print(f"LPIPS: {metrics['lpips']:.4f}")
    """
    metrics = {}
    
    try:
        metrics['psnr'] = calculate_psnr(img1, img2)
    except Exception as e:
        print(f"Failed to calculate PSNR: {e}")
        metrics['psnr'] = None
    
    try:
        metrics['ssim'] = calculate_ssim(img1, img2)
    except Exception as e:
        print(f"Failed to calculate SSIM: {e}")
        metrics['ssim'] = None
    
    try:
        metrics['lpips'] = calculate_lpips(img1, img2, device=device)
    except Exception as e:
        print(f"Failed to calculate LPIPS: {e}")
        metrics['lpips'] = None
    
    try:
        metrics['ms_ssim'] = calculate_ms_ssim(img1, img2)
    except Exception as e:
        print(f"Failed to calculate MS-SSIM: {e}")
        metrics['ms_ssim'] = None
    
    return metrics


# Helper functions
def _to_tensor(img: Union[torch.Tensor, np.ndarray, Image.Image],
               normalize: bool = False) -> torch.Tensor:
    """
    이미지를 torch.Tensor로 변환
    
    Args:
        img: 입력 이미지
        normalize: [-1, 1]로 정규화할지 여부 (LPIPS용)
    
    Returns:
        torch.Tensor: shape (B, C, H, W), 값 범위 [0, 1] or [-1, 1]
    """
    if isinstance(img, torch.Tensor):
        tensor = img
    elif isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[-1] == 3:  # (H, W, C)
            tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        else:
            tensor = torch.from_numpy(img).float()
        
        # Normalize to [0, 1] if needed
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
    elif isinstance(img, Image.Image):
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    
    # Add batch dimension if needed
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    
    # Normalize to [-1, 1] for LPIPS
    if normalize:
        tensor = tensor * 2.0 - 1.0
    
    return tensor


def _to_numpy(img: Union[torch.Tensor, np.ndarray, Image.Image]) -> np.ndarray:
    """
    이미지를 numpy array로 변환
    
    Returns:
        np.ndarray: shape (H, W, C), 값 범위 [0, 1]
    """
    if isinstance(img, np.ndarray):
        arr = img
        if arr.max() > 1.0:
            arr = arr / 255.0
    elif isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img[0]  # Remove batch dimension
        arr = img.cpu().permute(1, 2, 0).numpy()
    elif isinstance(img, Image.Image):
        arr = np.array(img).astype(np.float32) / 255.0
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    
    return arr


# TODO: 추가 메트릭 구현
def calculate_mae(img1, img2) -> float:
    """Mean Absolute Error"""
    tensor1 = _to_tensor(img1)
    tensor2 = _to_tensor(img2)
    mae = F.l1_loss(tensor1, tensor2)
    return mae.item()


def calculate_rmse(img1, img2) -> float:
    """Root Mean Squared Error"""
    tensor1 = _to_tensor(img1)
    tensor2 = _to_tensor(img2)
    mse = F.mse_loss(tensor1, tensor2)
    rmse = torch.sqrt(mse)
    return rmse.item()


def calculate_dreamsim(img1, img2, device='cuda') -> float:
    """
    DreamSim: 최신 perceptual similarity metric
    
    DreamSim은 LPIPS보다 인간의 perceptual judgment와 높은 상관관계를 보입니다.
    
    참고: https://github.com/ssundaram21/dreamsim
    """
    # TODO: 구현
    # from dreamsim import dreamsim
    # model, preprocess = dreamsim(pretrained=True, device=device)
    # 
    # img1_preprocessed = preprocess(img1).to(device)
    # img2_preprocessed = preprocess(img2).to(device)
    # 
    # with torch.no_grad():
    #     distance = model(img1_preprocessed, img2_preprocessed)
    # 
    # return distance.item()
    pass


"""
FID (Fréchet Inception Distance) Metric

이 모듈은 생성된 이미지의 품질을 평가하는 FID 메트릭을 제공합니다.

FID 설명:
- Inception-v3 네트워크의 feature space에서 실제 이미지와 생성 이미지의 분포 거리 측정
- 낮을수록 좋음 (0에 가까울수록 실제 이미지와 유사한 분포)
- 수식: ||μ_real - μ_gen||^2 + Tr(Σ_real + Σ_gen - 2(Σ_real * Σ_gen)^0.5)

사용 시나리오:
1. 원본 이미지 vs 재구성 이미지 (tokenizer 품질 평가)
2. 실제 데이터셋 vs 생성 데이터셋 (생성 모델 평가)

사용법:
    from vision_metrics.fid import calculate_fid_batch
    
    fid_score = calculate_fid_batch(real_images, generated_images)
    print(f"FID: {fid:.2f}")

참고:
- clean-fid 라이브러리 사용: https://github.com/GaParmar/clean-fid
- 설치: pip install clean-fid
"""

from typing import List, Union
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import tempfile


def calculate_fid_batch(real_images: List[Union[Image.Image, torch.Tensor]], 
                       gen_images: List[Union[Image.Image, torch.Tensor]],
                       batch_size: int = 50,
                       device: str = 'cuda',
                       mode: str = 'clean') -> float:
    """
    Calculate FID between two sets of images using clean-fid
    
    Args:
        real_images: List of real images (PIL or Tensor)
        gen_images: List of generated/reconstructed images (PIL or Tensor)
        batch_size: Batch size for feature extraction
        device: Device to use
        mode: FID mode - 'clean' (recommended) or 'legacy'
        
    Returns:
        fid: FID score (lower is better)
    
    Example:
        >>> real_imgs = [Image.open(f) for f in real_paths]
        >>> gen_imgs = [Image.open(f) for f in gen_paths]
        >>> fid = calculate_fid_batch(real_imgs, gen_imgs)
        >>> print(f"FID: {fid:.2f}")
    """
    try:
        from cleanfid import fid
    except ImportError:
        print("⚠️ clean-fid not installed. Installing...")
        print("  Run: pip install clean-fid")
        return float('nan')
    
    # Create temporary directories to save images
    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = Path(tmpdir) / "real"
        gen_dir = Path(tmpdir) / "gen"
        real_dir.mkdir()
        gen_dir.mkdir()
        
        # Save real images
        for i, img in enumerate(real_images):
            if isinstance(img, torch.Tensor):
                # Convert tensor to PIL
                if img.ndim == 4:
                    img = img[0]  # Remove batch dimension
                img = img.cpu()
                if img.shape[0] == 3:  # CHW format
                    img = img.permute(1, 2, 0)
                img = (img.clamp(0, 1) * 255).byte().numpy()
                img = Image.fromarray(img)
            elif isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                img = Image.fromarray(img)
            
            img.save(real_dir / f"{i:05d}.png")
        
        # Save generated images
        for i, img in enumerate(gen_images):
            if isinstance(img, torch.Tensor):
                # Convert tensor to PIL
                if img.ndim == 4:
                    img = img[0]  # Remove batch dimension
                img = img.cpu()
                if img.shape[0] == 3:  # CHW format
                    img = img.permute(1, 2, 0)
                img = (img.clamp(0, 1) * 255).byte().numpy()
                img = Image.fromarray(img)
            elif isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                img = Image.fromarray(img)
            
            img.save(gen_dir / f"{i:05d}.png")
        
        # Calculate FID using clean-fid
        try:
            fid_score = fid.compute_fid(
                str(real_dir),
                str(gen_dir),
                mode=mode,
                num_workers=0,
                batch_size=batch_size,
                device=torch.device(device)
            )
            return float(fid_score)
        except Exception as e:
            print(f"⚠️ FID calculation failed: {e}")
            # Try with legacy mode as fallback
            if mode == 'clean':
                print("  Retrying with legacy mode...")
                try:
                    fid_score = fid.compute_fid(
                        str(real_dir),
                        str(gen_dir),
                        mode='legacy',
                        num_workers=0,
                        batch_size=batch_size,
                        device=torch.device(device)
                    )
                    return float(fid_score)
                except Exception as e2:
                    print(f"⚠️ Legacy mode also failed: {e2}")
            return float('nan')


def calculate_fid_from_dirs(real_dir: str,
                            gen_dir: str,
                  batch_size: int = 50,
                  device: str = 'cuda',
                            mode: str = 'clean') -> float:
    """
    Calculate FID between two directories of images
    
    Args:
        real_dir: Directory containing real images
        gen_dir: Directory containing generated images
        batch_size: Batch size for feature extraction
        device: Device to use
        mode: FID mode - 'clean' (recommended) or 'legacy'
    
    Returns:
        fid: FID score (lower is better)
    
    Example:
        >>> fid = calculate_fid_from_dirs('data/real', 'results/generated')
        >>> print(f"FID: {fid:.2f}")
    """
    try:
        from cleanfid import fid
        
        fid_score = fid.compute_fid(
            real_dir,
            gen_dir,
            mode=mode,
            num_workers=0,
            batch_size=batch_size,
            device=torch.device(device)
        )
        return float(fid_score)
    except ImportError:
        print("⚠️ clean-fid not installed. Run: pip install clean-fid")
        return float('nan')
    except Exception as e:
        print(f"⚠️ FID calculation failed: {e}")
        return float('nan')


def calculate_fid(img1: Union[torch.Tensor, np.ndarray, Image.Image],
                  img2: Union[torch.Tensor, np.ndarray, Image.Image],
                             device: str = 'cuda') -> float:
    """
    Calculate FID for a single pair of images (not recommended)
    
    Note: FID is designed for comparing distributions, not individual images.
    For single images, this will return a placeholder value.
    
    Args:
        img1: Original image
        img2: Reconstructed image
        device: Device to use
    
    Returns:
        fid: Always returns 0.0 (FID requires multiple images)
    """
    print("⚠️ FID is designed for image distributions, not single images.")
    print("   Use calculate_fid_batch() with multiple images instead.")
    return 0.0


def precompute_fid_stats(image_dir: str,
                   output_path: str,
                         mode: str = 'clean',
                         batch_size: int = 50,
                   device: str = 'cuda'):
    """
    Precompute FID statistics for a dataset (for faster repeated FID calculations)
    
    Args:
        image_dir: Directory containing images
        output_path: Path to save statistics (.npz file)
        mode: FID mode - 'clean' or 'legacy'
        batch_size: Batch size for feature extraction
        device: Device to use
    
    Example:
        >>> # Precompute statistics once
        >>> precompute_fid_stats('data/coco/val', 'coco_val_stats.npz')
        >>> 
        >>> # Use precomputed stats for fast FID calculation
        >>> fid = calculate_fid_with_stats('results/generated', 'coco_val_stats.npz')
    """
    try:
        from cleanfid import fid
        
        print(f"Computing FID statistics for {image_dir}...")
        fid.make_custom_stats(
            name=Path(output_path).stem,
            fdir=image_dir,
            mode=mode,
            num_workers=0,
            batch_size=batch_size,
            device=torch.device(device)
        )
        print(f"✓ Statistics saved to {output_path}")
    except ImportError:
        print("⚠️ clean-fid not installed. Run: pip install clean-fid")
    except Exception as e:
        print(f"⚠️ Failed to compute statistics: {e}")


def calculate_fid_with_stats(gen_dir: str,
                             stats_name: str,
                             mode: str = 'clean',
                             batch_size: int = 50,
                             device: str = 'cuda') -> float:
    """
    Calculate FID using precomputed statistics
    
    Args:
        gen_dir: Directory containing generated images
        stats_name: Name of precomputed statistics
        mode: FID mode - 'clean' or 'legacy'
        batch_size: Batch size for feature extraction
        device: Device to use
    
    Returns:
        fid: FID score (lower is better)
    """
    try:
        from cleanfid import fid
        
        fid_score = fid.compute_fid(
            gen_dir,
            dataset_name=stats_name,
            mode=mode,
            num_workers=0,
            batch_size=batch_size,
            device=torch.device(device)
        )
        return float(fid_score)
    except ImportError:
        print("⚠️ clean-fid not installed. Run: pip install clean-fid")
        return float('nan')
    except Exception as e:
        print(f"⚠️ FID calculation failed: {e}")
        return float('nan')

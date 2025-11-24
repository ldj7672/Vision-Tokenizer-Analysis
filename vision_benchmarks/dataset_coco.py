"""
COCO Dataset Loader for Vision Tokenizer Benchmarking

이 모듈은 COCO 데이터셋을 로드하고 벤치마크에 사용할 수 있도록 합니다.

지원 기능:
1. COCO val2017 subset 로드
2. 지정된 개수의 샘플만 로드 (예: 1000장)
3. 이미지 전처리 및 변환
4. PyTorch Dataset 인터페이스 구현

사용법:
    from benchmarks.dataset_coco import COCODataset
    
    dataset = COCODataset(
        root='data/coco',
        split='val',
        num_samples=1000
    )
    
    for image, image_id in dataset:
        # Process image
        pass
"""

from typing import Optional, Tuple, Callable
from pathlib import Path
import json
from PIL import Image
import torch
from torch.utils.data import Dataset


class COCODataset(Dataset):
    """
    COCO 데이터셋 로더
    
    COCO val2017에서 지정된 개수의 이미지를 로드합니다.
    """
    
    def __init__(self,
                 root: str,
                 split: str = 'val',
                 year: str = '2017',
                 num_samples: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 return_id: bool = True):
        """
        Args:
            root: COCO 데이터셋 루트 디렉토리
            split: 'train' 또는 'val'
            year: 데이터셋 연도 ('2017', '2014')
            num_samples: 로드할 샘플 개수 (None이면 전체)
            transform: 이미지 변환 함수
            return_id: 이미지 ID 반환 여부
        """
        self.root = Path(root)
        self.split = split
        self.year = year
        self.transform = transform
        self.return_id = return_id
        
        # Paths
        self.image_dir = self.root / f"{split}{year}"
        self.annotation_file = self.root / "annotations" / f"instances_{split}{year}.json"
        
        # Check if dataset exists
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"COCO images not found at {self.image_dir}. "
                f"Please download COCO {split}{year} dataset first. "
                f"Run: python datasets/download_coco_1k.py"
            )
        
        # Load annotations
        self.image_ids, self.image_paths = self._load_image_list(num_samples)
        
        print(f"Loaded {len(self.image_ids)} images from COCO {split}{year}")
    
    def _load_image_list(self, num_samples: Optional[int]) -> Tuple[list, list]:
        """
        이미지 리스트 로드
        
        Args:
            num_samples: 로드할 샘플 개수
        
        Returns:
            Tuple: (image_ids, image_paths)
        """
        if self.annotation_file.exists():
            # Load from annotations
            with open(self.annotation_file, 'r') as f:
                coco_data = json.load(f)
            
            images = coco_data['images']
            
            # Limit number of samples
            if num_samples is not None:
                images = images[:num_samples]
            
            image_ids = [img['id'] for img in images]
            image_paths = [self.image_dir / img['file_name'] for img in images]
        else:
            # Load from directory (if annotations not available)
            image_paths = sorted(self.image_dir.glob('*.jpg'))
            
            if num_samples is not None:
                image_paths = image_paths[:num_samples]
            
            # Use filename as ID
            image_ids = [path.stem for path in image_paths]
        
        return image_ids, image_paths
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int):
        """
        이미지 로드
        
        Args:
            idx: 인덱스
        
        Returns:
            If return_id=True: (image, image_id)
            If return_id=False: image
        """
        image_path = self.image_paths[idx]
        image_id = self.image_ids[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transform
        if self.transform is not None:
            image = self.transform(image)
        
        if self.return_id:
            return image, image_id
        else:
            return image


class COCO1KDataset(COCODataset):
    """
    COCO val2017 1K subset
    
    벤치마크에 자주 사용되는 1000장 subset입니다.
    """
    
    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__(
            root=root,
            split='val',
            year='2017',
            num_samples=1000,
            transform=transform,
            return_id=True
        )


def create_coco_dataloader(root: str,
                          split: str = 'val',
                          num_samples: Optional[int] = None,
                          batch_size: int = 1,
                          num_workers: int = 4,
                          shuffle: bool = False) -> torch.utils.data.DataLoader:
    """
    COCO DataLoader 생성
    
    Args:
        root: COCO 데이터셋 루트 디렉토리
        split: 'train' 또는 'val'
        num_samples: 로드할 샘플 개수
        batch_size: 배치 크기
        num_workers: 워커 프로세스 수
        shuffle: 셔플 여부
    
    Returns:
        DataLoader
    
    Example:
        >>> dataloader = create_coco_dataloader('data/coco', num_samples=100)
        >>> for images, image_ids in dataloader:
        >>>     # Process batch
        >>>     pass
    """
    dataset = COCODataset(
        root=root,
        split=split,
        num_samples=num_samples
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# TODO: 추가 데이터셋 지원
class ImageNetDataset(Dataset):
    """ImageNet validation set"""
    pass


class CustomImageDataset(Dataset):
    """
    커스텀 이미지 디렉토리 로더
    
    임의의 이미지 디렉토리를 로드합니다.
    """
    
    def __init__(self, image_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            image_dir: 이미지 디렉토리 경로
            transform: 이미지 변환 함수
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Load image paths
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            self.image_paths.extend(self.image_dir.glob(ext))
        
        self.image_paths = sorted(self.image_paths)
        
        print(f"Loaded {len(self.image_paths)} images from {image_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, image_path.stem


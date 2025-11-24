"""
COCO Dataset Downloader

COCO val2017 데이터셋을 다운로드하고 1K subset을 준비합니다.

사용법:
    # 전체 COCO val2017 다운로드
    python datasets/download_coco_1k.py --output_dir data/coco
    
    # 1K subset만 다운로드
    python datasets/download_coco_1k.py --output_dir data/coco --subset_only --num_images 1000
"""

import argparse
import os
from pathlib import Path
import urllib.request
import zipfile
import json
from tqdm import tqdm
import shutil


def download_file(url: str, output_path: Path, desc: str = None):
    """
    파일 다운로드 (진행률 표시)
    
    Args:
        url: 다운로드 URL
        output_path: 저장 경로
        desc: 진행률 바 설명
    """
    print(f"Downloading {desc or url}...")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
    
    print(f"Downloaded to {output_path}")


def download_coco_val2017(output_dir: Path):
    """
    COCO val2017 다운로드
    
    Args:
        output_dir: 출력 디렉토리
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs
    images_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    # Download images
    images_zip = output_dir / "val2017.zip"
    if not images_zip.exists():
        download_file(images_url, images_zip, desc="COCO val2017 images")
    else:
        print(f"Images already downloaded: {images_zip}")
    
    # Download annotations
    annotations_zip = output_dir / "annotations_trainval2017.zip"
    if not annotations_zip.exists():
        download_file(annotations_url, annotations_zip, desc="COCO annotations")
    else:
        print(f"Annotations already downloaded: {annotations_zip}")
    
    # Extract images
    val_dir = output_dir / "val2017"
    if not val_dir.exists():
        print("Extracting images...")
        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted to {val_dir}")
    else:
        print(f"Images already extracted: {val_dir}")
    
    # Extract annotations
    annotations_dir = output_dir / "annotations"
    if not annotations_dir.exists():
        print("Extracting annotations...")
        with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted to {annotations_dir}")
    else:
        print(f"Annotations already extracted: {annotations_dir}")
    
    print("\nCOCO val2017 download complete!")
    print(f"Images: {val_dir}")
    print(f"Annotations: {annotations_dir}")


def create_subset(output_dir: Path, num_images: int = 1000):
    """
    COCO val2017에서 subset 생성
    
    Args:
        output_dir: COCO 데이터셋 디렉토리
        num_images: subset 크기
    """
    val_dir = output_dir / "val2017"
    annotations_file = output_dir / "annotations" / "instances_val2017.json"
    
    if not val_dir.exists() or not annotations_file.exists():
        raise FileNotFoundError(
            "COCO val2017 not found. Please download first."
        )
    
    # Load annotations
    print(f"Loading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Select first N images
    selected_images = coco_data['images'][:num_images]
    selected_image_ids = {img['id'] for img in selected_images}
    
    # Filter annotations
    selected_annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] in selected_image_ids
    ]
    
    # Create subset directory
    subset_dir = output_dir / f"val2017_{num_images}"
    subset_dir.mkdir(exist_ok=True)
    
    # Copy images
    print(f"Copying {num_images} images to {subset_dir}...")
    for img in tqdm(selected_images, desc="Copying images"):
        src = val_dir / img['file_name']
        dst = subset_dir / img['file_name']
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    
    # Save subset annotations
    subset_annotations = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'images': selected_images,
        'annotations': selected_annotations,
        'categories': coco_data['categories']
    }
    
    subset_ann_file = output_dir / "annotations" / f"instances_val2017_{num_images}.json"
    print(f"Saving subset annotations to {subset_ann_file}...")
    with open(subset_ann_file, 'w') as f:
        json.dump(subset_annotations, f)
    
    print(f"\nSubset created!")
    print(f"Images: {subset_dir}")
    print(f"Annotations: {subset_ann_file}")
    print(f"Number of images: {len(selected_images)}")
    print(f"Number of annotations: {len(selected_annotations)}")


def verify_dataset(output_dir: Path):
    """
    데이터셋 검증
    
    Args:
        output_dir: COCO 데이터셋 디렉토리
    """
    val_dir = output_dir / "val2017"
    annotations_file = output_dir / "annotations" / "instances_val2017.json"
    
    print("\nVerifying dataset...")
    
    # Check images
    if val_dir.exists():
        num_images = len(list(val_dir.glob("*.jpg")))
        print(f"✓ Images found: {num_images}")
    else:
        print(f"✗ Images not found: {val_dir}")
        return False
    
    # Check annotations
    if annotations_file.exists():
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        print(f"✓ Annotations found: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    else:
        print(f"✗ Annotations not found: {annotations_file}")
        return False
    
    print("\nDataset verification complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Download COCO val2017 dataset')
    parser.add_argument('--output_dir', type=str, default='data/coco',
                       help='Output directory for COCO dataset')
    parser.add_argument('--subset_only', action='store_true',
                       help='Only create subset (assumes full dataset already downloaded)')
    parser.add_argument('--num_images', type=int, default=1000,
                       help='Number of images for subset')
    parser.add_argument('--skip_download', action='store_true',
                       help='Skip download (only verify)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("COCO Dataset Downloader")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Subset size: {args.num_images}")
    print("=" * 80)
    
    if not args.skip_download and not args.subset_only:
        # Download full dataset
        download_coco_val2017(output_dir)
    
    if args.subset_only or not args.skip_download:
        # Create subset
        try:
            create_subset(output_dir, args.num_images)
        except Exception as e:
            print(f"Error creating subset: {e}")
    
    # Verify dataset
    verify_dataset(output_dir)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print("\nYou can now run benchmarks with:")
    print(f"  python benchmarks/run_benchmark.py --config configs/coco_1k_example.yaml")


if __name__ == '__main__':
    main()


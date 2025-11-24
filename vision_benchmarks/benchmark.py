"""
Vision Tokenizer Benchmark Module

BenchmarkRunner 클래스를 제공합니다.
실행은 루트의 run_benchmark.py를 사용하세요.
"""

import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
import time
from tqdm import tqdm
import pandas as pd
import torch
from PIL import Image
import numpy as np
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# 모델 캐시 경로 설정 (가장 먼저 import)
import vision_tokenizers.model_cache

from vision_tokenizers.base import VisionTokenizerBase
from vision_tokenizers.vae_ldm import VAELDMTokenizer
from vision_tokenizers.magvit2 import MAGVIT2Tokenizer
from vision_tokenizers.titok import TiTokTokenizer
from vision_tokenizers.tatok_combined import (
    TATokARDTokTokenizer,
    TATokSANATokenizer,
    TATokLumina2Tokenizer
)

from vision_metrics.reconstruction import calculate_all_metrics
from vision_metrics.token_stats import calculate_all_token_stats
from vision_metrics.fid import calculate_fid_batch

from vision_benchmarks.dataset_coco import COCODataset


class BenchmarkRunner:
    """
    Vision Tokenizer 벤치마크 실행기
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 벤치마크 설정 (YAML에서 로드)
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(config.get('output_dir', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Multi-GPU setup
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.use_multi_gpu = config.get('use_multi_gpu', True) and self.num_gpus > 1
        
        if self.use_multi_gpu:
            print(f"🚀 Multi-GPU enabled: {self.num_gpus} GPUs detected")
            print(f"   Will spawn {self.num_gpus} processes automatically")
            self.should_spawn = True
        else:
            print(f"Single GPU mode")
            self.should_spawn = False
        
        # Internal use for spawned processes
        self.gpu_id = config.get('_gpu_id', 0)
        self.total_gpus = config.get('_total_gpus', 1)
        
        # Initialize tokenizers
        self.tokenizers = self._init_tokenizers()
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Results storage
        self.results = []
        
    def _init_tokenizers(self) -> Dict[str, Any]:
        """
        설정에서 토크나이저 초기화
        
        Returns:
            Dict: {tokenizer_name: tokenizer_instance}
        """
        print("Initializing tokenizers from config...")
        tokenizers = {}
        
        # Get global optimization settings
        use_bf16 = self.config.get('use_bf16', False)
        use_compile = self.config.get('use_compile', False)
        
        if use_bf16:
            print("⚡ BFloat16 enabled")
        if use_compile:
            print("⚡ torch.compile enabled")
        
        for tok_config in self.config.get('tokenizers', []):
            name = tok_config['name']
            type_ = tok_config['type']
            
            # Skip if disabled
            if not tok_config.get('enabled', True):
                print(f"⚠ Skipping disabled tokenizer: {name}")
                continue
            
            try:
                if type_ == 'vae':
                    tokenizer = VAELDMTokenizer(
                        model_name=tok_config.get('model_name', 'stabilityai/sd-vae-ft-mse'),
                        device=self.device,
                        use_quantization=tok_config.get('use_quantization', False),
                        num_quantization_bins=tok_config.get('num_quantization_bins', 256),
                        use_bf16=use_bf16,
                        use_compile=use_compile
                    )
                elif type_ == 'tatok_ardtok':
                    tokenizer = TATokARDTokTokenizer(
                        ta_tok_checkpoint=tok_config.get('ta_tok_checkpoint'),
                        ta_tok_model=tok_config.get('ta_tok_model', 'ByteDance-Seed/Tar-TA-Tok'),
                        ar_dtok_size=tok_config.get('ar_dtok_size', 1024),
                        device=self.device,
                        use_bf16=use_bf16,
                        use_compile=use_compile
                    )
                elif type_ == 'tatok_sana':
                    tokenizer = TATokSANATokenizer(
                        ta_tok_checkpoint=tok_config.get('ta_tok_checkpoint'),
                        ta_tok_model=tok_config.get('ta_tok_model', 'ByteDance-Seed/Tar-TA-Tok'),
                        sana_model=tok_config.get('sana_model', 'csuhan/Tar-SANA-600M-1024px'),
                        device=self.device,
                        use_bf16=use_bf16,
                        use_compile=use_compile
                    )
                elif type_ == 'tatok_lumina2':
                    tokenizer = TATokLumina2Tokenizer(
                        ta_tok_checkpoint=tok_config.get('ta_tok_checkpoint'),
                        ta_tok_model=tok_config.get('ta_tok_model', 'ByteDance-Seed/Tar-TA-Tok'),
                        lumina2_model=tok_config.get('lumina2_model', 'csuhan/Tar-Lumina2'),
                        device=self.device,
                        use_bf16=use_bf16,
                        use_compile=use_compile
                    )
                elif type_ == 'vqgan':
                    tokenizer = VQGANTokenizer(
                        checkpoint_path=tok_config.get('checkpoint_path'),
                        config_path=tok_config.get('config_path'),
                        model_name=tok_config.get('model_name', 'vqgan_imagenet_f16_16384'),
                        device=self.device,
                        use_bf16=use_bf16,
                        use_compile=use_compile
                    )
                elif type_ == 'magvit2':
                    tokenizer = MAGVIT2Tokenizer(
                        image_size=tok_config.get('image_size', 256),
                        codebook_size=tok_config.get('codebook_size', 262144),
                        model_name=tok_config.get('model_name', 'TencentARC/Open-MAGVIT2'),
                        device=self.device,
                        use_bf16=use_bf16,
                        use_compile=use_compile
                    )
                elif type_ == 'titok':
                    tokenizer = TiTokTokenizer(
                        image_size=tok_config.get('image_size', 256),
                        num_latent_tokens=tok_config.get('num_latent_tokens', 32),
                        codebook_size=tok_config.get('codebook_size', 4096),
                        model_name=tok_config.get('model_name', 'yucornetto/tokenizer_titok_l32_imagenet'),
                        device=self.device,
                        use_bf16=use_bf16,
                        use_compile=use_compile
                    )
                else:
                    print(f"⚠ Tokenizer type '{type_}' not yet implemented, skipping '{name}'")
                    continue
                
                tokenizers[name] = tokenizer
                print(f"✓ Loaded tokenizer: {name}")
            except Exception as e:
                print(f"✗ Failed to load tokenizer '{name}': {e}")
                continue
        
        return tokenizers
    
    def _load_dataset(self):
        """데이터셋 로드"""
        print("Loading dataset from config...")
        dataset_config = self.config.get('dataset', {})
        
        try:
            if dataset_config.get('name') == 'coco':
                dataset = COCODataset(
                    root=dataset_config['root'],
                    split=dataset_config.get('split', 'val'),
                    year=dataset_config.get('year', '2017'),
                    num_samples=dataset_config.get('num_samples', None)
                )
                print(f"✓ Loaded dataset: COCO {dataset_config.get('split', 'val')}{dataset_config.get('year', '2017')}")
                return dataset
            else:
                print(f"⚠ Dataset '{dataset_config.get('name')}' not yet implemented")
                return None
        except Exception as e:
            print(f"✗ Failed to load dataset: {e}")
            return None
    
    def run(self):
        """
        벤치마크 실행
        """
        # Multi-GPU: spawn processes
        if self.should_spawn:
            return self._run_multi_gpu()
        
        # Single GPU or spawned process
        print("=" * 80)
        print("Vision Tokenizer Benchmark")
        print("=" * 80)
        print(f"Device: {self.device}")
        if self.total_gpus > 1:
            print(f"GPU: {self.gpu_id}/{self.total_gpus-1}")
        print(f"Number of tokenizers: {len(self.tokenizers)}")
        print(f"Number of images: {len(self.dataset) if self.dataset else 'N/A'}")
        print("=" * 80)
        
        if not self.tokenizers:
            print("\n⚠ No tokenizers loaded. Please check your configuration.")
            return
        
        if not self.dataset:
            print("\n⚠ No dataset loaded. Please check your configuration.")
            return
        
        for tokenizer_name, tokenizer in self.tokenizers.items():
            print(f"\nBenchmarking: {tokenizer_name}")
            print("-" * 80)
            
            # Get tokenizer info
            info = tokenizer.info()
            print(f"Type: {info['type']}")
            print(f"Codebook size: {info.get('codebook_size', 'N/A')}")
            print(f"Compression ratio: {info.get('compression_ratio', 'N/A')}x")
            
            # Benchmark on dataset
            tokenizer_results = self._benchmark_tokenizer(tokenizer_name, tokenizer)
            self.results.extend(tokenizer_results)
        
        # Save results
        self._save_results()
        
        # Print summary (only if single GPU or need to merge)
        if self.total_gpus == 1:
            self._print_summary()
            self._generate_visualizations()
    
    def _run_multi_gpu(self):
        """Run benchmark on multiple GPUs using subprocess"""
        import subprocess
        import tempfile
        
        print(f"\n🚀 Launching {self.num_gpus} GPU processes...")
        print("=" * 80)
        
        processes = []
        log_files = []
        
        # Get the run_benchmark.py path (in project root)
        project_root = Path(__file__).parent.parent
        run_script = project_root / 'run_benchmark.py'
        
        for gpu_id in range(self.num_gpus):
            # Create config for this GPU
            gpu_config = self.config.copy()
            gpu_config['_gpu_id'] = gpu_id
            gpu_config['_total_gpus'] = self.num_gpus
            gpu_config['use_multi_gpu'] = False
            
            # Save temp config
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(gpu_config, f)
                temp_config = f.name
            
            log_file = f"benchmark_gpu{gpu_id}.log"
            log_files.append(log_file)
            
            # Launch process
            cmd = ['python', '-u', str(run_script), '--config', temp_config]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
            
            print(f"📍 GPU {gpu_id}: Logging to {log_file}")
            
            with open(log_file, 'w') as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            processes.append((proc, temp_config))
        
        print("\n✅ All processes launched!")
        print("\nMonitor:")
        for i, log in enumerate(log_files):
            print(f"  GPU {i}: tail -f {log}")
        print("\n⏳ Waiting for completion...\n")
        
        # Wait for all
        for proc, temp_config in processes:
            proc.wait()
            try:
                os.unlink(temp_config)
            except:
                pass
        
        print("\n✅ All GPUs completed!")
        print("\n📊 Merging results...")
        
        # Merge
        all_dfs = []
        for gpu_id in range(self.num_gpus):
            csv_file = self.output_dir / f'benchmark_results_gpu{gpu_id}.csv'
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                all_dfs.append(df)
                print(f"  ✓ GPU {gpu_id}: {len(df)} results")
        
        if all_dfs:
            merged = pd.concat(all_dfs, ignore_index=True)
            merged = merged.sort_values(['tokenizer', 'image_id']).reset_index(drop=True)
            
            out_csv = self.output_dir / 'benchmark_results.csv'
            merged.to_csv(out_csv, index=False)
            print(f"\n✓ Saved merged results: {out_csv}")
            
            # Load for summary
            self.results = merged.to_dict('records')
            self._print_summary()
            self._generate_visualizations()
        
        # Generate visualizations
        output_config = self.config.get('output', {})
        if output_config.get('generate_visualizations', True):
            self._generate_visualizations()
    
    
    def _benchmark_tokenizer(self, name: str, tokenizer) -> List[Dict[str, Any]]:
        """
        단일 토크나이저 벤치마크 (2-phase: generation + metrics)
        
        Args:
            name: 토크나이저 이름
            tokenizer: 토크나이저 인스턴스
        
        Returns:
            List[Dict]: 각 이미지에 대한 결과
        """
        # Create output directory for this tokenizer
        tokenizer_dir = self.output_dir / name
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        
        # Split dataset for multi-GPU processing
        dataset_indices = list(range(len(self.dataset)))
        if self.total_gpus > 1:
            dataset_indices = [i for i in dataset_indices if i % self.total_gpus == self.gpu_id]
            print(f"📊 GPU {self.gpu_id}: Processing {len(dataset_indices)}/{len(self.dataset)} images")
        
        # ========================================
        # PHASE 1: Generate and save images
        # ========================================
        print(f"\n🎨 Phase 1: Generating images for {name}...")
        
        # Load or create timing info file
        timing_file = tokenizer_dir / 'timing_info.json'
        if timing_file.exists():
            import json
            with open(timing_file, 'r') as f:
                generation_times = json.load(f)
        else:
            generation_times = {}
        
        desc = f"Generating {name}" if self.total_gpus == 1 else f"Generating {name} [GPU {self.gpu_id}]"
        skipped = 0
        generated = 0
        
        for idx in tqdm(dataset_indices, desc=desc):
            try:
                image, image_id = self.dataset[idx]
                
                # Check if already generated
                recon_path = tokenizer_dir / f"{image_id}_recon.png"
                
                if recon_path.exists():
                    # Skip if already generated
                    skipped += 1
                    continue
                
                # Measure encoding time
                start_time = time.time()
                encoded = tokenizer.encode(image)
                encode_time = time.time() - start_time
                
                # Measure decoding time
                start_time = time.time()
                reconstructed = tokenizer.decode(encoded)
                decode_time = time.time() - start_time
                
                # Resize reconstructed image to match original size
                if reconstructed.size != image.size:
                    reconstructed = reconstructed.resize(image.size, Image.LANCZOS)
                
                # Save reconstructed image with retry logic
                save_success = False
                for retry in range(3):
                    try:
                        reconstructed.save(recon_path)
                        save_success = True
                        break
                    except OSError as e:
                        if retry < 2:
                            print(f"\n⚠ Save failed (attempt {retry+1}/3): {e}")
                            time.sleep(1)  # Wait 1 second before retry
                        else:
                            print(f"\n❌ Save failed after 3 attempts: {e}")
                            raise
                
                if not save_success:
                    continue
                
                # Store timing info
                generation_times[str(image_id)] = {
                    'encode_time': encode_time,
                    'decode_time': decode_time,
                    'total_time': encode_time + decode_time
                }
                generated += 1
            
            except Exception as e:
                print(f"\n⚠ Error generating image {image_id} with {name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save timing info
        import json
        with open(timing_file, 'w') as f:
            json.dump(generation_times, f, indent=2)
        
        print(f"✓ Phase 1 complete: {generated} generated, {skipped} skipped")
        
        # ========================================
        # PHASE 2: Calculate metrics from saved images
        # ========================================
        print(f"\n📊 Phase 2: Calculating metrics for {name}...")
        results = []
        original_images = []
        reconstructed_images = []
        
        desc = f"Metrics {name}" if self.total_gpus == 1 else f"Metrics {name} [GPU {self.gpu_id}]"
        for idx in tqdm(dataset_indices, desc=desc):
            try:
                original, image_id = self.dataset[idx]
                
                # Load reconstructed image
                recon_path = tokenizer_dir / f"{image_id}_recon.png"
                
                if not recon_path.exists():
                    print(f"\n⚠ Missing reconstructed image for {image_id}, skipping")
                    continue
                
                reconstructed = Image.open(recon_path).convert('RGB')
                
                # Store for FID calculation
                original_images.append(original)
                reconstructed_images.append(reconstructed)
                
                # Calculate reconstruction metrics (without LPIPS for now)
                recon_metrics = calculate_all_metrics(original, reconstructed, device=self.device)
                
                # Get timing info
                timing = generation_times.get(str(image_id), {
                    'encode_time': 0.0,
                    'decode_time': 0.0,
                    'total_time': 0.0
                })
                
                # Store results
                result = {
                    'tokenizer': name,
                    'image_id': str(image_id),
                    **timing,
                    **recon_metrics
                }
                results.append(result)
            
            except Exception as e:
                print(f"\n⚠ Error calculating metrics for {image_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate FID for all images
        if original_images and reconstructed_images and len(original_images) > 1:
            try:
                print(f"\n📈 Calculating FID for {name}...")
                fid_score = calculate_fid_batch(
                    original_images, 
                    reconstructed_images,
                    batch_size=min(50, len(original_images)),
                    device=self.device
                )
                for result in results:
                    result['fid'] = fid_score
                print(f"✓ FID: {fid_score:.2f}")
            except Exception as e:
                print(f"⚠ Could not calculate FID: {e}")
                for result in results:
                    result['fid'] = float('nan')
        
        return results
    
    def _save_results(self):
        """결과 저장"""
        if not self.results:
            print("\n⚠ No results to save")
            return
        
        # Add GPU ID suffix if in multi-GPU mode
        suffix = f"_gpu{self.gpu_id}" if self.total_gpus > 1 else ""
        
        # Save to CSV
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / f'benchmark_results{suffix}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to {csv_path}")
        
        if self.total_gpus > 1:
            print(f"   💡 Run merge script after all GPUs finish: python merge_results.py")
        
        # Save to Parquet (more efficient for large datasets)
        try:
            parquet_path = self.output_dir / f'benchmark_results{suffix}.parquet'
            df.to_parquet(parquet_path, index=False)
            print(f"✓ Results saved to {parquet_path}")
        except Exception as e:
            print(f"⚠ Could not save Parquet: {e}")
    
    def _print_summary(self):
        """결과 요약 출력 및 저장"""
        if not self.results:
            print("\n⚠ No results to summarize")
            return
        
        df = pd.DataFrame(self.results)
        
        # Prepare summary text
        summary_lines = []
        summary_lines.append("=" * 120)
        summary_lines.append("BENCHMARK SUMMARY")
        summary_lines.append("=" * 120)
        summary_lines.append("")
        
        # Group by tokenizer and create formatted table
        grouped = df.groupby('tokenizer')
        
        # Create header
        metrics = ['psnr', 'ssim', 'lpips', 'fid', 'mae', 'rmse', 'encode_time', 'decode_time']
        available_metrics = [m for m in metrics if m in df.columns]
        
        header = f"{'Tokenizer':<25}"
        for metric in available_metrics:
            header += f" | {metric.upper():>18}"
        summary_lines.append(header)
        summary_lines.append("-" * len(header))
        
        # Create rows
        for tokenizer in sorted(df['tokenizer'].unique()):
            row = f"{tokenizer:<25}"
            for metric in available_metrics:
                mean_val = grouped.get_group(tokenizer)[metric].mean()
                std_val = grouped.get_group(tokenizer)[metric].std()
                
                # Format based on metric type
                if metric in ['psnr', 'fid']:
                    row += f" | {mean_val:7.2f} ± {std_val:5.2f}"
                elif metric in ['ssim', 'lpips', 'mae', 'rmse']:
                    row += f" | {mean_val:7.4f} ± {std_val:6.4f}"
                else:  # time metrics
                    row += f" | {mean_val:7.3f} ± {std_val:6.3f}"
            
            summary_lines.append(row)
        
        summary_lines.append("")
        
        # Find best tokenizer for each metric
        summary_lines.append("\n" + "-" * 80)
        summary_lines.append("BEST TOKENIZERS")
        summary_lines.append("-" * 80)
        
        grouped = df.groupby('tokenizer')
        
        if 'psnr' in df.columns:
            best_psnr_tok = grouped['psnr'].mean().idxmax()
            best_psnr_val = grouped['psnr'].mean().max()
            summary_lines.append(f"Best PSNR (↑): {best_psnr_tok} ({best_psnr_val:.2f} dB)")
        
        if 'ssim' in df.columns:
            best_ssim_tok = grouped['ssim'].mean().idxmax()
            best_ssim_val = grouped['ssim'].mean().max()
            summary_lines.append(f"Best SSIM (↑): {best_ssim_tok} ({best_ssim_val:.4f})")
        
        if 'lpips' in df.columns:
            best_lpips_tok = grouped['lpips'].mean().idxmin()
            best_lpips_val = grouped['lpips'].mean().min()
            summary_lines.append(f"Best LPIPS (↓): {best_lpips_tok} ({best_lpips_val:.4f})")
        
        if 'fid' in df.columns:
            best_fid_tok = grouped['fid'].mean().idxmin()
            best_fid_val = grouped['fid'].mean().min()
            summary_lines.append(f"Best FID (↓): {best_fid_tok} ({best_fid_val:.2f})")
        
        if 'decode_time' in df.columns:
            best_time_tok = grouped['decode_time'].mean().idxmin()
            best_time_val = grouped['decode_time'].mean().min()
            summary_lines.append(f"Best Decode Time (↓): {best_time_tok} ({best_time_val:.3f}s)")
        
        summary_lines.append("=" * 80)
        
        # Print to console
        summary_text = "\n".join(summary_lines)
        print("\n" + summary_text)
        
        # Save to file
        summary_path = self.output_dir / 'benchmark_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        print(f"\n✓ Summary saved to {summary_path}")
    
    def _generate_visualizations(self):
        """결과 시각화 생성"""
        if not self.results:
            print("\n⚠ No results to visualize")
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            sns.set_style("whitegrid")
            plt.rcParams['figure.dpi'] = 150
            plt.rcParams['font.size'] = 10
            
            df = pd.DataFrame(self.results)
            
            # Create comprehensive figure with subplots for each metric
            metrics_config = [
                ('psnr', 'PSNR (dB)', '↑ Higher is Better', 'viridis'),
                ('ssim', 'SSIM', '↑ Higher is Better', 'viridis'),
                ('lpips', 'LPIPS', '↓ Lower is Better', 'viridis_r'),
                ('fid', 'FID', '↓ Lower is Better', 'viridis_r'),
                ('decode_time', 'Decode Time (s)', '↓ Lower is Better', 'viridis_r'),
            ]
            
            available_metrics = [(m, label, direction, cmap) for m, label, direction, cmap in metrics_config if m in df.columns]
            
            if available_metrics:
                # Calculate statistics for each tokenizer
                tokenizers = sorted(df['tokenizer'].unique())
                n_metrics = len(available_metrics)
                n_tokenizers = len(tokenizers)
                
                # Create figure with subplots (2 rows x 3 cols)
                fig, axes = plt.subplots(2, 3, figsize=(20, 12))
                axes = axes.flatten()
                
                # Color palette
                colors = plt.cm.Set3(np.linspace(0, 1, n_tokenizers))
                
                for idx, (metric, label, direction, cmap) in enumerate(available_metrics):
                    if idx >= len(axes):
                        break
                    
                    ax = axes[idx]
                    
                    # Get mean and std for each tokenizer
                    metric_stats = df.groupby('tokenizer')[metric].agg(['mean', 'std']).reindex(tokenizers)
                    means = metric_stats['mean'].values
                    stds = metric_stats['std'].values
                    
                    # Create bar chart
                    x_pos = np.arange(n_tokenizers)
                    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                                 color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
                    
                    # Add value labels on bars
                    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                        height = bar.get_height()
                        # Format based on metric
                        if metric == 'decode_time':
                            value_text = f'{mean:.2f}±{std:.2f}'
                        elif metric in ['psnr', 'fid']:
                            value_text = f'{mean:.1f}±{std:.1f}'
                        else:
                            value_text = f'{mean:.3f}±{std:.3f}'
                        
                        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                               value_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
                    # Highlight best performer
                    if '↓' in direction:  # Lower is better
                        best_idx = np.argmin(means)
                    else:  # Higher is better
                        best_idx = np.argmax(means)
                    bars[best_idx].set_edgecolor('red')
                    bars[best_idx].set_linewidth(3)
                    
                    # Customize subplot
                    ax.set_ylabel(label, fontsize=12, fontweight='bold')
                    ax.set_title(f'{label} {direction}', fontsize=13, fontweight='bold', pad=10)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(tokenizers, rotation=45, ha='right', fontsize=10)
                    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                    ax.set_axisbelow(True)
                    
                    # Add y=0 line for reference
                    ax.axhline(y=0, color='black', linewidth=0.8)
                
                # Hide unused subplots
                for idx in range(len(available_metrics), len(axes)):
                    axes[idx].axis('off')
                
                # Add overall title
                fig.suptitle('Vision Tokenizer Benchmark - Comprehensive Comparison\n(Red border = Best performer)', 
                           fontsize=18, fontweight='bold', y=0.98)
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(self.output_dir / 'benchmark_summary.png', dpi=200, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved comprehensive visualization: benchmark_summary.png")
        
        except Exception as e:
            print(f"⚠ Could not generate visualizations: {e}")
            import traceback
            traceback.print_exc()


# This module contains only the BenchmarkRunner class
# Use run_benchmark.py in the root directory to execute benchmarks
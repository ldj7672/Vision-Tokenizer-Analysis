#!/usr/bin/env python3
"""
Vision Tokenizer Benchmark Runner

Usage:
    # Quick test (10 images)
    python run_benchmark.py --config configs/discrete_tokenizers.yaml
    
    # Full evaluation (3000 images)
    python run_benchmark.py --config configs/discrete_tokenizers_full.yaml
    
    # Background execution
    nohup python -u run_benchmark.py --config configs/discrete_tokenizers_full.yaml > benchmark.log 2>&1 &
    
    # Custom output directory
    python run_benchmark.py --config configs/discrete_tokenizers.yaml --output_dir results/my_test
    
    # Force single GPU
    python run_benchmark.py --config configs/discrete_tokenizers.yaml --device cuda:0

Multi-GPU:
    - Automatically detects and uses multiple GPUs
    - Set use_multi_gpu: false in config to disable
    - Each GPU processes a subset of images in parallel
"""

import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import benchmark runner
from vision_benchmarks.benchmark import BenchmarkRunner


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Vision Tokenizer Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU (default)
  python run_benchmark.py --config configs/discrete_tokenizers_full.yaml
  
  # With BFloat16 (faster, less memory)
  python run_benchmark.py --config configs/discrete_tokenizers_full.yaml --bf16
  
  # With torch.compile (faster inference)
  python run_benchmark.py --config configs/discrete_tokenizers_full.yaml --compile
  
  # All optimizations
  python run_benchmark.py --config configs/discrete_tokenizers_full.yaml --bf16 --compile
  
  # Multi-GPU
  python run_benchmark.py --config configs/discrete_tokenizers_full.yaml --multi-gpu
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu/cuda:0). Overrides config.'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results. Overrides config.'
    )
    
    parser.add_argument(
        '--multi-gpu',
        action='store_true',
        help='Enable multi-GPU processing (default: single GPU)'
    )
    
    parser.add_argument(
        '--bf16',
        action='store_true',
        help='Use BFloat16 for faster inference and lower memory (requires Ampere+ GPU)'
    )
    
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Use torch.compile for faster inference (requires PyTorch 2.0+)'
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"📄 Loading config: {args.config}")
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.device:
        config['device'] = args.device
        print(f"   Device override: {args.device}")
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
        print(f"   Output directory override: {args.output_dir}")
    
    # Handle multi-GPU flag (default: single GPU)
    if args.multi_gpu:
        config['use_multi_gpu'] = True
        print(f"   Multi-GPU: Enabled")
    else:
        config['use_multi_gpu'] = False
        print(f"   Single GPU mode")
    
    # Handle optimization flags
    if args.bf16:
        config['use_bf16'] = True
        print(f"   ⚡ BFloat16: Enabled")
    
    if args.compile:
        config['use_compile'] = True
        print(f"   ⚡ torch.compile: Enabled")
    
    print()
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    runner.run()
    
    print("\n" + "=" * 80)
    print("✅ Benchmark completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

# Single GPU (default):
# nohup python -u run_benchmark.py --config configs/discrete_tokenizers_full.yaml > benchmark.log 2>&1 &

# With BF16 + compile (recommended for speed):
# nohup python -u run_benchmark.py --config configs/discrete_tokenizers_full.yaml --bf16 --compile > benchmark.log 2>&1 &
# nohup python -u run_benchmark.py --config configs/discrete_tokenizers_etc.yaml --bf16 --compile > benchmark_etc.log 2>&1 &
# 🔬 Vision Tokenizer Analysis

A comprehensive toolkit for analyzing and benchmarking vision tokenizers with detailed experimental results.

## 📖 Overview

This project provides three main functionalities:

1. **🎮 Interactive Testing** - Test tokenizers on your own images with a simple CLI
2. **📊 Comprehensive Benchmark** - Evaluate multiple tokenizers on COCO dataset with detailed metrics
3. **🌐 Interactive Demo** - Web-based interface for real-time experimentation

### Supported Tokenizers

| Tokenizer | Type | Resolution | Tokens | Codebook |
|-----------|------|------------|--------|----------|
| **TA-Tok + AR-DTok** | Autoregressive | 512px | 729 → 256 | 65K → 16K |
| **TA-Tok + SANA** | Diffusion | 512px | 729 | 65K |
| **TA-Tok + Lumina2** | Diffusion | 512px | 729 | 65K |
| **MAGVIT-v2** | LFQ | 256px | 256 | 262K |
| **TiTok-L-32** | 1D Latent | 256px | 32 | 4K |
| **VAE (SD-MSE)** | VAE | 512px | - | 256 |

## 🚀 Quick Start

### Installation

```bash
# 1. Create conda environment
conda create -n vtp python=3.11
conda activate vtp

# 2. Install all dependencies
pip install -r requirements.txt
```

**Note:** PyTorch CUDA wheels and flash-attention are included in `requirements.txt`. If you encounter issues with flash-attention compilation, install it separately:
```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

### Model Setup

Most models are **automatically downloaded** on first use. For Lumina2:

```bash
# Login to Hugging Face (required for Gemma-2-2b access)
huggingface-cli login

# Download Gemma-2-2b (one-time setup)
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/gemma-2-2b')"
```

All models are cached in `./model_weights/` directory.

## 💻 Usage

### 1. Interactive Testing (CLI)

Test tokenizers on your own images:

```bash
# List available tokenizers
python playground.py --list

# Test single image
python playground.py --model magvit2_256 --input image.jpg --output results/

# Test folder of images
python playground.py --model tatok_ardtok_512 --input images/ --output results/

# Compare multiple tokenizers
python playground.py --model magvit2_256,titok_256,vae_512 --input image.jpg --output results/

# Save token arrays
python playground.py --model magvit2_256 --input image.jpg --output results/ --save-tokens
```

### 2. Benchmark Evaluation

Run comprehensive benchmarks on COCO dataset:

**First, download COCO dataset:**
```bash
python datasets/download_coco_1k.py
```

**Then run benchmarks:**
```bash
# Quick test (10 images)
python run_benchmark.py --config configs/discrete_tokenizers.yaml

# Full evaluation (1000 images) - Run in background
nohup python -u run_benchmark.py --config configs/discrete_tokenizers_full.yaml > benchmark.log 2>&1 &
```

**Output:** (saved in config's `output_dir`, e.g., `results/discrete_tokenizers/`)
- `benchmark_results.csv` - Detailed metrics
- `benchmark_summary.txt` - Text summary
- `benchmark_summary.png` - Visualization
- `{tokenizer_name}/` - Reconstructed images

### 3. Gradio Web Demo

Launch interactive web interface:

```bash
# Default (port 7860)
python run_demo.py

# Custom port
python run_demo.py --port 8080
```

Then open `http://localhost:7860` in your browser.

**Demo Interface:**

![Gradio Demo](docs/images/figures/demo.png)

The demo allows you to:
- Upload any image and see reconstructed results from different tokenizers
- Compare visual quality across various tokenization methods
- All models are preloaded at startup for fast inference

## 📊 Evaluation Metrics

- **PSNR** (↑) - Peak Signal-to-Noise Ratio
- **SSIM** (↑) - Structural Similarity Index
- **LPIPS** (↓) - Learned Perceptual Image Patch Similarity
- **FID** (↓) - Fréchet Inception Distance
- **MAE** (↓) - Mean Absolute Error
- **RMSE** (↓) - Root Mean Square Error
- **Encode/Decode Time** (↓) - Inference speed

## 📁 Project Structure

```
vision_tokenizer_playground/
├── playground.py              # 🎮 Interactive CLI
├── run_benchmark.py           # 📊 Benchmark runner
├── run_demo.py                # 🌐 Gradio demo launcher
├── requirements.txt           # Package dependencies
│
├── scripts/
│   └── demo_gradio.py         # Gradio demo implementation
│
├── vision_tokenizers/         # Tokenizer wrapper classes (unified interface)
│   ├── base.py                # Base tokenizer class (VisionTokenizerBase)
│   ├── tatok.py               # TA-Tok encoder wrapper
│   ├── tatok_combined.py      # TA-Tok + de-tokenizer combinations
│   ├── ar_dtok.py             # AR-DTok wrapper (uses tok.ar_dtok)
│   ├── sana_dtok.py           # SANA wrapper (uses tok.dif_dtok_sana)
│   ├── lumina2_dtok.py        # Lumina2 wrapper (uses tok.dif_dtok_lumina2)
│   ├── magvit2.py             # MAGVIT-v2 tokenizer
│   ├── titok.py               # TiTok tokenizer
│   ├── vae_ldm.py             # VAE baseline
│   ├── model_cache.py         # Model caching utilities
│   ├── magvit2_modules/       # Extracted MAGVIT2 code
│   └── titok_modules/         # Extracted TiTok code
│
│   Note: Wrapper classes provide unified `encode()`/`decode()` interface.
│   TA-Tok variants internally use `tok/` module for actual implementation.
│
├── vision_metrics/            # Metric implementations
│   ├── reconstruction.py      # PSNR, SSIM, LPIPS
│   ├── fid.py                 # FID calculation
│   └── token_stats.py         # Token statistics
│
├── vision_benchmarks/         # Benchmark implementation
│   ├── benchmark.py           # Main benchmark logic
│   └── dataset_coco.py        # COCO dataset loader
│
├── configs/                   # Configuration files
│   ├── discrete_tokenizers.yaml
│   └── discrete_tokenizers_full.yaml
│
├── datasets/                  # Dataset utilities
│   └── download_coco_1k.py   # COCO dataset downloader
│
├── tok/                       # Tar project original code (low-level implementations)
│   ├── ta_tok.py              # TA-Tok encoder (original implementation)
│   ├── ar_dtok/               # AR-DTok implementation
│   ├── dif_dtok_sana.py       # SANA Dif-DTok
│   ├── dif_dtok_lumina2.py    # Lumina2 Dif-DTok
│   ├── lumina2_model.py       # Lumina2 model utilities
│   ├── mm_autoencoder.py      # Multi-modal autoencoder
│   ├── models.py              # Model definitions
│   ├── transport/             # Transport-based diffusion
│   └── utils.py               # Utility functions
│
│   Note: `tok/` contains original Tar project code, while `vision_tokenizers/` 
│   provides unified wrapper classes that use `tok/` internally.
│
├── docs/                      # Documentation assets
│   └── images/                # Figures and samples
│
├── data/                      # Dataset storage (auto-created)
│   └── coco/                  # COCO dataset
│
├── results/                   # Benchmark results (auto-created)
│
└── model_weights/             # Downloaded models (auto-created)
```

## 🔧 Configuration

Edit `configs/discrete_tokenizers.yaml` to customize:

```yaml
dataset:
  name: coco
  root: data/coco
  num_samples: 10  # Number of images

tokenizers:
  - name: magvit2_256
    type: magvit2
    enabled: true  # Set to false to skip

metrics:
  reconstruction:
    - psnr
    - ssim
    - lpips
    - fid
```

## 📚 Python API

Use tokenizers programmatically:

```python
from playground import TokenizerPlayground

# Initialize
playground = TokenizerPlayground(device='cuda')

# Load tokenizer
playground.load('magvit2_256')

# Encode & decode
from PIL import Image
image = Image.open('example.jpg')
tokens = playground.encode(image)
reconstructed = playground.decode(tokens)

# Get info
info = playground.info()
print(f"Tokens: {info['num_tokens']}, Codebook: {info['codebook_size']}")
```

## 🎯 Experimental Results

We evaluated 6 vision tokenizers on 1000 images from COCO val2017 dataset. The results show a comprehensive comparison across multiple metrics.

### Benchmark Summary

![Benchmark Results](docs/images/figures/benchmark_summary.png)

### Quantitative Results

| Tokenizer | PSNR (↑) | SSIM (↑) | LPIPS (↓) | FID (↓) | Encode Time (s) | Decode Time (s) |
|-----------|----------|----------|-----------|---------|-----------------|-----------------|
| **vae_sd_mse_q8_512** | **24.92 ± 4.99** | **0.6953 ± 0.1467** | **0.0830 ± 0.0522** | **12.45 ± 0.00** | 0.039 ± 0.295 | 0.070 ± 0.344 |
| magvit2_256 | 18.65 ± 4.56 | 0.4951 ± 0.1717 | 0.3936 ± 0.1298 | 34.37 ± 0.00 | 0.015 ± 0.181 | **0.011 ± 0.169** |
| titok_l32_256 | 15.02 ± 2.40 | 0.3766 ± 0.1657 | 0.5724 ± 0.1206 | 54.23 ± 0.00 | 0.013 ± 0.197 | 0.028 ± 0.501 |
| tatok_lumina2_512 | 14.01 ± 2.21 | 0.3910 ± 0.1641 | 0.4452 ± 0.0948 | 44.01 ± 0.00 | 0.028 ± 0.007 | 95.720 ± 0.038 |
| tatok_sana_512 | 13.11 ± 2.24 | 0.3702 ± 0.1629 | 0.4998 ± 0.1044 | 48.91 ± 0.00 | 0.021 ± 0.002 | 2.008 ± 0.019 |
| tatok_ardtok_512 | 12.35 ± 1.97 | 0.3609 ± 0.1570 | 0.5357 ± 0.1135 | 57.95 ± 0.00 | 0.016 ± 0.013 | 58.287 ± 41.953 |

**Key Findings:**

- **Best Overall Quality**: VAE (SD-MSE) achieves the best reconstruction quality across all metrics (PSNR: 24.92 dB, SSIM: 0.6953), but uses continuous latent space with quantization rather than discrete tokens.

- **Best Discrete Tokenizer**: Among discrete tokenization models, **MAGVIT2** offers the best quality-speed trade-off with PSNR of 18.65 dB and the fastest decoding time (0.011s), making it ideal for real-time applications.

- **TA-Tok Variants**: Lumina2 provides the best quality among TA-Tok variants but is slow (95.72s). SANA offers a good balance (2.008s decode time), while AR-DTok is slower with lower quality.

- **Extreme Compression**: TiTok-L-32 achieves moderate quality (PSNR: 15.02 dB) with very fast decoding (0.028s) using only 32 tokens for 256×256 images, making it suitable for extreme compression scenarios.

### Sample Reconstructions

| Model | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Original** | ![orig_1](docs/images/originals/original_1000.jpg) | ![orig_2](docs/images/originals/original_100582.jpg) | ![orig_3](docs/images/originals/original_100624.jpg) | ![orig_4](docs/images/originals/original_101068.jpg) | ![orig_5](docs/images/originals/original_101420.jpg) |
| **MAGVIT2**<br>(256px) | ![magvit2_1](docs/images/samples/magvit2_256_1000_recon.png) | ![magvit2_2](docs/images/samples/magvit2_256_100582_recon.png) | ![magvit2_3](docs/images/samples/magvit2_256_100624_recon.png) | ![magvit2_4](docs/images/samples/magvit2_256_101068_recon.png) | ![magvit2_5](docs/images/samples/magvit2_256_101420_recon.png) |
| **TA-Tok + AR-DTok**<br>(512px) | ![ardtok_1](docs/images/samples/tatok_ardtok_512_1000_recon.png) | ![ardtok_2](docs/images/samples/tatok_ardtok_512_100582_recon.png) | ![ardtok_3](docs/images/samples/tatok_ardtok_512_100624_recon.png) | ![ardtok_4](docs/images/samples/tatok_ardtok_512_101068_recon.png) | ![ardtok_5](docs/images/samples/tatok_ardtok_512_101420_recon.png) |
| **TA-Tok + Lumina2**<br>(512px) | ![lumina2_1](docs/images/samples/tatok_lumina2_512_1000_recon.png) | ![lumina2_2](docs/images/samples/tatok_lumina2_512_100582_recon.png) | ![lumina2_3](docs/images/samples/tatok_lumina2_512_100624_recon.png) | ![lumina2_4](docs/images/samples/tatok_lumina2_512_101068_recon.png) | ![lumina2_5](docs/images/samples/tatok_lumina2_512_101420_recon.png) |
| **TA-Tok + SANA**<br>(512px) | ![sana_1](docs/images/samples/tatok_sana_512_1000_recon.png) | ![sana_2](docs/images/samples/tatok_sana_512_100582_recon.png) | ![sana_3](docs/images/samples/tatok_sana_512_100624_recon.png) | ![sana_4](docs/images/samples/tatok_sana_512_101068_recon.png) | ![sana_5](docs/images/samples/tatok_sana_512_101420_recon.png) |
| **TiTok-L-32**<br>(256px) | ![titok_1](docs/images/samples/titok_l32_256_1000_recon.png) | ![titok_2](docs/images/samples/titok_l32_256_100582_recon.png) | ![titok_3](docs/images/samples/titok_l32_256_100624_recon.png) | ![titok_4](docs/images/samples/titok_l32_256_101068_recon.png) | ![titok_5](docs/images/samples/titok_l32_256_101420_recon.png) |
| **VAE SD-MSE**<br>(512px) | ![vae_1](docs/images/samples/vae_sd_mse_q8_512_1000_recon.png) | ![vae_2](docs/images/samples/vae_sd_mse_q8_512_100582_recon.png) | ![vae_3](docs/images/samples/vae_sd_mse_q8_512_100624_recon.png) | ![vae_4](docs/images/samples/vae_sd_mse_q8_512_101068_recon.png) | ![vae_5](docs/images/samples/vae_sd_mse_q8_512_101420_recon.png) |

### Detailed Results

Full benchmark results are available in:
- `docs/images/figures/benchmark_summary.png` - Comprehensive visualization
- `docs/images/figures/benchmark_summary.txt` - Text summary with detailed statistics
- `results/discrete_tokenizers_full/benchmark_results_merged.csv` - Detailed per-image metrics (6000 images × 6 tokenizers)


## 🙏 Acknowledgments

- [Tar](https://github.com/csuhan/Tar) - TA-Tok and de-tokenizers
- [Open-MAGVIT2](https://github.com/TencentARC/Open-MAGVIT2) - MAGVIT-v2 implementation
- [TiTok](https://github.com/fun-research/TiTok) - TiTok implementation
- [Diffusers](https://github.com/huggingface/diffusers) - Stable Diffusion VAE


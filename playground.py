#!/usr/bin/env python
"""
Vision Tokenizer Analysis - Simple Inference Interface

A simple class-based interface to test and analyze different vision tokenizers.

Usage:
    from playground import TokenizerPlayground
    
    # Initialize
    playground = TokenizerPlayground(device='cuda')
    
    # Load tokenizer
    playground.load('tatok_ardtok_512')
    
    # Encode & decode
    from PIL import Image
    image = Image.open('example.jpg')
    tokens = playground.encode(image)
    reconstructed = playground.decode(tokens)
"""

import torch
from PIL import Image
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np


class TokenizerPlayground:
    """
    Simple interface to experiment with vision tokenizers.
    
    Available tokenizers:
        - tatok: TA-Tok encoder only
        - tatok_ardtok_512: TA-Tok + AR-DTok (512px)
        - tatok_sana_512: TA-Tok + SANA (512px)
        - tatok_lumina2_512: TA-Tok + Lumina2 (512px)
        - magvit2_256: MAGVIT-v2 (256px)
        - titok_256: TiTok-L-32 (256px)
        - vae_512: Stable Diffusion VAE (512px)
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize playground.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        self.tokenizer_name = None
        
        print(f"🔬 Vision Tokenizer Analysis initialized (device: {self.device})")
    
    def list_available(self) -> Dict[str, str]:
        """List all available tokenizers with descriptions."""
        tokenizers = {
            'tatok': 'TA-Tok encoder only (no reconstruction)',
            'tatok_ardtok_512': 'TA-Tok + AR-DTok (Autoregressive, 512px)',
            'tatok_sana_512': 'TA-Tok + SANA (Diffusion, 512px)',
            'tatok_lumina2_512': 'TA-Tok + Lumina2 (Diffusion, 512px, best quality)',
            'magvit2_256': 'MAGVIT-v2 (LFQ, 256px)',
            'titok_256': 'TiTok-L-32 (32 tokens, 256px)',
            'vae_512': 'Stable Diffusion VAE (512px, baseline)',
        }
        return tokenizers
    
    def load(self, tokenizer_name: str, **kwargs):
        """
        Load a tokenizer.
        
        Args:
            tokenizer_name: Name of the tokenizer to load
            **kwargs: Additional arguments for tokenizer initialization
        """
        print(f"\n📦 Loading {tokenizer_name}...")
        
        try:
            if tokenizer_name == 'tatok':
                from vision_tokenizers import TATokTokenizer
                self.tokenizer = TATokTokenizer(
                    checkpoint_path=kwargs.get('checkpoint_path', 'model_weights/ta_tok.pth'),
                    device=self.device
                )
            
            elif tokenizer_name == 'tatok_ardtok_512':
                from vision_tokenizers import TATokARDTokTokenizer
                self.tokenizer = TATokARDTokTokenizer(
                    ta_tok_checkpoint=kwargs.get('ta_tok_checkpoint', 'model_weights/ta_tok.pth'),
                    ar_dtok_checkpoint=kwargs.get('ar_dtok_checkpoint', 'csuhan/TA-Tok'),
                    output_size=kwargs.get('output_size', 512),
                    device=self.device
                )
            
            elif tokenizer_name == 'tatok_sana_512':
                from vision_tokenizers import TATokSANATokenizer
                self.tokenizer = TATokSANATokenizer(
                    ta_tok_checkpoint=kwargs.get('ta_tok_checkpoint', 'model_weights/ta_tok.pth'),
                    sana_model=kwargs.get('sana_model', 'csuhan/Tar-SANA-600M-512px'),
                    output_size=kwargs.get('output_size', 512),
                    device=self.device
                )
            
            elif tokenizer_name == 'tatok_lumina2_512':
                from vision_tokenizers import Lumina2Detokenizer
                self.tokenizer = Lumina2Detokenizer(
                    ta_tok_path=kwargs.get('ta_tok_path', 'model_weights/ta_tok.pth'),
                    lumina2_model=kwargs.get('lumina2_model', 'csuhan/Tar-Lumina2'),
                    output_size=kwargs.get('output_size', 512),
                    device=self.device
                )
            
            elif tokenizer_name == 'magvit2_256':
                from vision_tokenizers import MAGVIT2Tokenizer
                self.tokenizer = MAGVIT2Tokenizer(
                    image_size=kwargs.get('image_size', 256),
                    model_name=kwargs.get('model_name', 'TencentARC/Open-MAGVIT2'),
                    device=self.device
                )
            
            elif tokenizer_name == 'titok_256':
                from vision_tokenizers import TiTokTokenizer
                self.tokenizer = TiTokTokenizer(
                    image_size=kwargs.get('image_size', 256),
                    model_name=kwargs.get('model_name', 'yucornetto/tokenizer_titok_l32_imagenet'),
                    device=self.device
                )
            
            elif tokenizer_name == 'vae_512':
                from vision_tokenizers import VAELDMTokenizer
                self.tokenizer = VAELDMTokenizer(
                    model_name=kwargs.get('model_name', 'stabilityai/sd-vae-ft-mse'),
                    image_size=kwargs.get('image_size', 512),
                    quantize_bits=kwargs.get('quantize_bits', 8),
                    device=self.device
                )
            
            else:
                raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
            
            self.tokenizer_name = tokenizer_name
            print(f"✅ Loaded {tokenizer_name}")
            
            # Print info
            info = self.info()
            print(f"   Tokens: {info.get('num_tokens', 'N/A')}")
            print(f"   Codebook: {info.get('codebook_size', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Failed to load {tokenizer_name}: {e}")
            import traceback
            traceback.print_exc()
            self.tokenizer = None
            self.tokenizer_name = None
    
    def encode(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Encode image to tokens.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
        
        Returns:
            Token indices as numpy array (or dict for some tokenizers)
        """
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer loaded. Call load() first.")
        
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        result = self.tokenizer.encode(image)
        
        # Extract indices from dict if needed
        if isinstance(result, dict):
            # Return the dict itself, decode() will handle it
            return result
        else:
            return result
    
    def decode(self, tokens: np.ndarray) -> Image.Image:
        """
        Decode tokens to image.
        
        Args:
            tokens: Token indices
        
        Returns:
            Reconstructed image as PIL Image
        """
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer loaded. Call load() first.")
        
        return self.tokenizer.decode(tokens)
    
    def reconstruct(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Encode and decode image (full reconstruction).
        
        Args:
            image: Input image
        
        Returns:
            Reconstructed image
        """
        tokens = self.encode(image)
        return self.decode(tokens)
    
    def info(self) -> Dict[str, Any]:
        """Get tokenizer information."""
        if self.tokenizer is None:
            return {}
        return self.tokenizer.info()
    
    def compare(self, image: Union[str, Path, Image.Image], tokenizers: list):
        """
        Compare multiple tokenizers on the same image.
        
        Args:
            image: Input image
            tokenizers: List of tokenizer names to compare
        
        Returns:
            Dictionary of {tokenizer_name: reconstructed_image}
        """
        results = {}
        
        # Load image once
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        for tok_name in tokenizers:
            print(f"\n🔄 Testing {tok_name}...")
            self.load(tok_name)
            
            try:
                reconstructed = self.reconstruct(image)
                results[tok_name] = reconstructed
                print(f"✅ {tok_name} completed")
            except Exception as e:
                print(f"❌ {tok_name} failed: {e}")
                results[tok_name] = None
        
        return results


def main():
    """CLI interface for TokenizerPlayground."""
    import argparse
    import glob
    import os
    
    parser = argparse.ArgumentParser(
        description='Vision Tokenizer Playground - Test tokenizers on images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available tokenizers
  python playground.py --list
  
  # Test single image
  python playground.py --model magvit2_256 --input image.jpg --output results/
  
  # Test folder of images
  python playground.py --model tatok_ardtok_512 --input images/ --output results/
  
  # Compare multiple tokenizers
  python playground.py --model magvit2_256,titok_256 --input image.jpg --output results/
  
  # Use CPU
  python playground.py --model vae_512 --input image.jpg --output results/ --device cpu
        """
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available tokenizers'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Tokenizer model(s) to use (comma-separated for multiple)'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input image file or folder'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='playground_results',
        help='Output directory (default: playground_results)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    
    parser.add_argument(
        '--save-tokens',
        action='store_true',
        help='Save token arrays as .npy files'
    )
    
    args = parser.parse_args()
    
    # Initialize playground
    playground = TokenizerPlayground(device=args.device)
    
    # List tokenizers
    if args.list:
        print("\n" + "=" * 80)
        print("📋 Available Tokenizers")
        print("=" * 80)
        for name, desc in playground.list_available().items():
            print(f"  • {name:<25} - {desc}")
        print("=" * 80 + "\n")
        return
    
    # Validate arguments
    if not args.model or not args.input:
        parser.print_help()
        return
    
    # Parse models
    models = [m.strip() for m in args.model.split(',')]
    
    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        input_files = [input_path]
    elif input_path.is_dir():
        # Find all image files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        input_files = []
        for ext in extensions:
            input_files.extend(input_path.glob(ext))
            input_files.extend(input_path.glob(ext.upper()))
        input_files = sorted(input_files)
    else:
        print(f"❌ Input path not found: {input_path}")
        return
    
    if not input_files:
        print(f"❌ No image files found in: {input_path}")
        return
    
    print("\n" + "=" * 80)
    print("🎨 Vision Tokenizer Playground")
    print("=" * 80)
    print(f"Models: {', '.join(models)}")
    print(f"Input: {len(input_files)} image(s)")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print("=" * 80 + "\n")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for img_idx, img_path in enumerate(input_files, 1):
        print(f"\n📷 [{img_idx}/{len(input_files)}] Processing: {img_path.name}")
        print("-" * 80)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            print(f"   Loaded: {image.size}")
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            continue
        
        # Process with each model
        for model_name in models:
            print(f"\n   🔄 {model_name}...")
            
            try:
                # Load model
                playground.load(model_name)
                
                # Encode
                tokens = playground.encode(image)
                
                # Handle different return types
                if isinstance(tokens, dict):
                    # Some tokenizers return dict with 'indices' key
                    token_shape = tokens.get('indices', tokens.get('tokens', None))
                    if token_shape is not None:
                        if hasattr(token_shape, 'shape'):
                            print(f"      Encoded: {token_shape.shape}")
                        else:
                            print(f"      Encoded: dict with keys {list(tokens.keys())}")
                    else:
                        print(f"      Encoded: dict with keys {list(tokens.keys())}")
                elif hasattr(tokens, 'shape'):
                    print(f"      Encoded: {tokens.shape}")
                else:
                    print(f"      Encoded: {type(tokens)}")
                
                # Decode
                reconstructed = playground.decode(tokens)
                print(f"      Decoded: {reconstructed.size}")
                
                # Save reconstructed image
                output_name = f"{img_path.stem}_{model_name}.png"
                output_path = output_dir / output_name
                reconstructed.save(output_path)
                print(f"      ✅ Saved: {output_path}")
                
                # Save tokens if requested
                if args.save_tokens:
                    tokens_path = output_dir / f"{img_path.stem}_{model_name}_tokens.npy"
                    np.save(tokens_path, tokens)
                    print(f"      💾 Tokens: {tokens_path}")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✨ Processing completed!")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()


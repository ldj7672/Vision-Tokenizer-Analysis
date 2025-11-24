"""
Vision Tokenizer Analysis - Interactive Gradio Demo

Upload an image and compare different vision tokenizers in real-time.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image

# Import tokenizers
from vision_tokenizers import (
    TATokTokenizer,
    TATokARDTokTokenizer,
    TATokSANATokenizer,
    TATokLumina2Tokenizer,
    MAGVIT2Tokenizer,
    TiTokTokenizer,
    VAELDMTokenizer,
)

# No metrics needed for demo


class TokenizerDemo:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizers = {}
        self.loaded_tokenizers = set()  # Track successfully loaded tokenizers
        
    def load_tokenizer(self, tokenizer_name):
        """Load tokenizer - models should be preloaded"""
        if tokenizer_name in self.tokenizers:
            return self.tokenizers[tokenizer_name]
        return None
    
    def load_all_tokenizers(self, tokenizer_configs):
        """Preload all tokenizers at startup"""
        loaded = []
        failed = []
        
        for tokenizer_name, config in tokenizer_configs.items():
            try:
                print(f"Loading {tokenizer_name}...")
                tokenizer_type = config['type']
                
                if tokenizer_type == "tatok":
                    tokenizer = TATokTokenizer(
                        checkpoint_path=config.get('checkpoint', "model_weights/ta_tok.pth"),
                        device=self.device
                    )
                elif tokenizer_type == "tatok_ardtok":
                    tokenizer = TATokARDTokTokenizer(
                        ta_tok_checkpoint=config.get('ta_tok_checkpoint', "model_weights/ta_tok.pth"),
                        ar_dtok_checkpoint=config.get('ar_dtok_checkpoint', "csuhan/TA-Tok"),
                        output_size=config.get('output_size', 512),
                        device=self.device
                    )
                elif tokenizer_type == "tatok_sana":
                    tokenizer = TATokSANATokenizer(
                        ta_tok_checkpoint=config.get('ta_tok_checkpoint', "model_weights/ta_tok.pth"),
                        sana_model=config.get('sana_model', "csuhan/Tar-SANA-600M-512px"),
                        output_size=config.get('output_size', 512),
                        device=self.device
                    )
                elif tokenizer_type == "tatok_lumina2":
                    tokenizer = TATokLumina2Tokenizer(
                        ta_tok_checkpoint=config.get('ta_tok_checkpoint', "model_weights/ta_tok.pth"),
                        lumina2_model=config.get('lumina2_model', "csuhan/Tar-Lumina2"),
                        output_size=config.get('output_size', 512),
                        device=self.device
                    )
                elif tokenizer_type == "magvit2":
                    tokenizer = MAGVIT2Tokenizer(
                        image_size=config.get('image_size', 256),
                        model_name=config.get('model_name', "TencentARC/Open-MAGVIT2"),
                        device=self.device
                    )
                elif tokenizer_type == "titok":
                    tokenizer = TiTokTokenizer(
                        image_size=config.get('image_size', 256),
                        model_name=config.get('model_name', "yucornetto/tokenizer_titok_l32_imagenet"),
                        device=self.device
                    )
                elif tokenizer_type == "vae":
                    tokenizer = VAELDMTokenizer(
                        model_name=config.get('model_name', "stabilityai/sd-vae-ft-mse"),
                        image_size=config.get('image_size', 512),
                        quantize_bits=config.get('quantize_bits', 8),
                        device=self.device
                    )
                else:
                    failed.append((tokenizer_name, "Unknown tokenizer type"))
                    continue
                
                self.tokenizers[tokenizer_name] = tokenizer
                self.loaded_tokenizers.add(tokenizer_name)
                loaded.append(tokenizer_name)
                print(f"✓ Loaded {tokenizer_name}")
                
            except Exception as e:
                failed.append((tokenizer_name, str(e)))
                print(f"✗ Failed to load {tokenizer_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return loaded, failed
    
    def process_image(self, image, tokenizer_name, target_size=512):
        """Process image through selected tokenizer - simple reconstruction only"""
        if image is None:
            return None
        
        if not tokenizer_name:
            return None
        
        # Get preloaded tokenizer
        tokenizer = self.load_tokenizer(tokenizer_name)
        if tokenizer is None:
            return None
        
        try:
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Resize to target size
            image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Encode and decode
            tokens = tokenizer.encode(image)
            reconstructed = tokenizer.decode(tokens)
            
            # Resize reconstructed to match original input size
            if reconstructed.size != image.size:
                reconstructed = reconstructed.resize(image.size, Image.Resampling.LANCZOS)
            
            return reconstructed
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None


def create_demo():
    """Create Gradio interface"""
    demo_handler = TokenizerDemo()
    
    # Tokenizer configurations
    tokenizer_configs = {
        "TA-Tok + AR-DTok (512px)": {
            'type': 'tatok_ardtok',
            'ta_tok_checkpoint': 'model_weights/ta_tok.pth',
            'ar_dtok_checkpoint': 'csuhan/TA-Tok',
            'output_size': 512
        },
        "TA-Tok + SANA (512px)": {
            'type': 'tatok_sana',
            'ta_tok_checkpoint': 'model_weights/ta_tok.pth',
            'sana_model': 'csuhan/Tar-SANA-600M-512px',
            'output_size': 512
        },
        "TA-Tok + Lumina2 (512px)": {
            'type': 'tatok_lumina2',
            'ta_tok_checkpoint': 'model_weights/ta_tok.pth',
            'lumina2_model': 'csuhan/Tar-Lumina2'
        },
        "MAGVIT-v2 (256px)": {
            'type': 'magvit2',
            'image_size': 256,
            'model_name': 'TencentARC/Open-MAGVIT2'
        },
        "TiTok-L-32 (256px)": {
            'type': 'titok',
            'image_size': 256,
            'model_name': 'yucornetto/tokenizer_titok_l32_imagenet'
        },
        "VAE (SD-MSE, 512px)": {
            'type': 'vae',
            'model_name': 'stabilityai/sd-vae-ft-mse',
            'image_size': 512,
            'quantize_bits': 8
        },
    }
    
    # Preload all tokenizers
    print("\n" + "="*80)
    print("Loading all tokenizers...")
    print("="*80)
    loaded, failed = demo_handler.load_all_tokenizers(tokenizer_configs)
    
    print("\n" + "="*80)
    print(f"✓ Successfully loaded {len(loaded)} tokenizers")
    if failed:
        print(f"✗ Failed to load {len(failed)} tokenizers:")
        for name, error in failed:
            print(f"  - {name}: {error}")
    print("="*80 + "\n")
    
    # Only show successfully loaded tokenizers
    tokenizer_choices = loaded if loaded else list(tokenizer_configs.keys())
    
    # Gradio 6.0.0+ doesn't support theme parameter in Blocks constructor
    # Use default theme
    with gr.Blocks(title="Vision Tokenizer Analysis") as demo:
        gr.Markdown(f"""
        # 🔬 Vision Tokenizer Analysis
        
        Upload an image and explore different vision tokenizers in real-time!
        
        **Status:** {len(loaded)}/{len(tokenizer_configs)} tokenizers loaded and ready to use.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=400
                )
                
                tokenizer_dropdown = gr.Dropdown(
                    choices=tokenizer_choices,
                    value=tokenizer_choices[0] if tokenizer_choices else None,
                    label="Select Tokenizer",
                    info=f"Choose a vision tokenizer to test ({len(tokenizer_choices)} available)",
                    interactive=len(tokenizer_choices) > 0
                )
                
                target_size = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    step=256,
                    value=512,
                    label="Target Size (px)",
                    info="Image will be resized to this resolution"
                )
                
                process_btn = gr.Button(
                    "🚀 Encode & Decode", 
                    variant="primary", 
                    size="lg",
                    interactive=len(tokenizer_choices) > 0
                )
                
                gr.Markdown("""
                ### 📝 Tokenizer Info
                
                - **TA-Tok**: Text-aligned tokenizer (encoder only, no reconstruction)
                - **AR-DTok**: Autoregressive de-tokenizer (fast, good quality)
                - **SANA**: Diffusion de-tokenizer (slower, high quality)
                - **Lumina2**: Large diffusion model (slowest, best quality)
                - **MAGVIT-v2**: Lookup-free quantization (efficient)
                - **TiTok**: 1D latent tokenization (32 tokens only!)
                - **VAE**: Stable Diffusion VAE baseline
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Reconstruction")
                output_image = gr.Image(
                    label="Reconstructed Image",
                    type="pil",
                    height=400
                )
        
        # Event handlers
        process_btn.click(
            fn=demo_handler.process_image,
            inputs=[input_image, tokenizer_dropdown, target_size],
            outputs=[output_image]
        )
        
        gr.Markdown("""
        ---
        ### 🔬 About This Demo
        
        Upload an image and see how different vision tokenizers compress and reconstruct it.
        
        **Note:** All tokenizers are preloaded at startup. If a tokenizer failed to load, it will not appear in the dropdown menu.
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


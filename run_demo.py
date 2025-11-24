#!/usr/bin/env python
"""
Launch Gradio Interactive Demo

Usage:
    python run_demo.py [--port PORT] [--share]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.demo_gradio import create_demo


def main():
    parser = argparse.ArgumentParser(description='Launch Vision Tokenizer Analysis Demo')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the demo on')
    parser.add_argument('--share', action='store_true', help='Create a public link')
    parser.add_argument('--server-name', type=str, default='0.0.0.0', help='Server name')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔬 Vision Tokenizer Analysis - Interactive Demo")
    print("=" * 80)
    print(f"\n🚀 Starting Gradio demo on http://localhost:{args.port}")
    if args.share:
        print("📡 Creating public share link...")
    print("\n" + "=" * 80 + "\n")
    
    demo = create_demo()
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()


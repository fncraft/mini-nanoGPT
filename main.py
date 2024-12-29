"""
# Mini-NanoGPT

Based on karpathy/nanoGPT with a GUI and extended features that make GPT model training intuitive and accessible. Currently implements native GPT training, with GPT-2 fine-tuning in development.

- 🚀 One-click data processing, training and inference
- 🎨 Real-time training visualization and logging
- 🔧 Character-level and GPT-2 tokenizer support
- 💾 Checkpoint resume and model evaluation
- 🌏 Multi-language interface (English/Chinese)
- 📊 Rich learning rate scheduling options
- 🎛️ Visual configuration for all parameters, no code editing needed

Compared to the original nanoGPT, this project adds more practical features and flexible configuration options, giving you complete control over the training process. Whether for learning or experimentation, it helps you explore GPT models more easily.

Built with PyTorch and Gradio, featuring clean code structure, perfect for deep learning beginners and researchers.
"""

from app.interface import build_app_interface

def main():
    demo = build_app_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()

# MMDiT-PyTorch

**MMDiT-PyTorch** is a lightweight and standalone PyTorch implementation of a single block from the **Multimodal Diffusion Transformer (MMDiT)**, originally proposed in [*Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*](https://arxiv.org/abs/2403.03206).

<div align="center"><img src="https://raw.githubusercontent.com/KennyStryker/mmdit-pytorch/refs/heads/main/assets/mmdit.png" alt="MMDiT Architecture" width="400"/></div>

This project focuses on simplicity and minimal dependencies to allow easy understanding and extensibility for research and experimentation.

---

## 🔍 Overview

MMDiT introduces a scalable and efficient Transformer-based architecture tailored for high-resolution image synthesis through rectified flows. This repository implements a **single MMDiT block** for educational and experimental purposes.

- 📦 Single-block MMDiT in PyTorch
- 🧠 Minimal and readable implementation
- 🛠️ No training framework dependency

---

## 📦 Installation

Make sure you have Python 3.12+

### Using pip
```bash
pip install mmdit-pytorch
```

### From the source
```bash
git clone https://github.com/KennyStryker/mmdit-pytorch.git
cd mmdit-pytorch
poetry install
```

---

## 🚀 Usage

Make sure you have Python 3.12+ and [Poetry](https://python-poetry.org/) installed.

```python
import torch
from mmdit import MMDiTBlock

# Set embedding dimensions for each modality
dim_txt = 768         # Dimension of text embeddings
dim_img = 512         # Dimension of image embeddings
dim_timestep = 256    # Dimension of timestep embeddings (e.g., for conditioning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the multimodal transformer block
mmdit_block = MMDiTBlock(
    dim_txt=dim_txt,
    dim_img=dim_img,
    dim_timestep=dim_timestep,
    qk_rmsnorm=True  # Use RMSNorm on query/key in attention (optional setting)
).to(device)

# Generate random embeddings for demonstration
txt_emb = torch.randn(1, 512, dim_txt).to(device)
img_emb = torch.randn(1, 1024, dim_img).to(device)
time_emb = torch.randn(1, dim_timestep).to(device)

# Forward pass through the multimodal transformer block
txt_out, img_out = mmdit_block(txt_emb, img_emb, time_emb)

print(f"Text output shape: {txt_out.shape}")
print(f"Image output shape: {img_out.shape}")
```

---

## Citations

```bibtex
@article{arXiv,
    title   = {Scaling Rectified Flow Transformers for High-Resolution Image Synthesis},
    author  = {Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, Robin Rombach},
    url     = {https://arxiv.org/abs/2403.03206}
}
```

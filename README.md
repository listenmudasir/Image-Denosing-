# SCUNet with Frequency-Aware Attention (FAA)

Official PyTorch implementation of **SCUNetWithFAA**, an extension of SCUNet that injects a lightweight **frequency-aware attention (FAA)** module into the bottleneck to enhance texture preservation and reduce haloing/ringing artifacts.

> TL;DR: We keep the strong Swin-style SCUNet backbone, and add a tiny FFT-based attention gate in the bottleneck to let the network "look" at frequency content before decoding, giving sharper details and fewer artifacts.

---

## 🔧 Features

- 🧱 **Backbone:** Original SCUNet architecture with Swin-inspired ConvTrans blocks.   
- 🎚 **Frequency-Aware Attention (FAA):** FFT-based channel attention module applied in the bottleneck, using `rFFT2` magnitude as a global frequency descriptor.   
- 🧮 **Stable mixed precision:** FAA always computes in `float32` internally to avoid numerical issues with AMP.
- 🧪 Easy to plug into other models: FAA is implemented as a standalone `nn.Module`.
- 📊 Evaluation helpers for PSNR/SSIM and visual comparison.

---

## 🏗 Architecture Overview

The code provides two main models:

- `SCUNet` (baseline): defined in `models/network_scunet.py`
- `SCUNetWithFAA`:

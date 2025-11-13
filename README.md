# SCUNetWithFAA: Frequency-Aware Attention for Image Restoration

This repository provides an **unofficial PyTorch implementation** of **SCUNet** and our extension **SCUNetWithFAA**, which integrates a lightweight **Frequency-Aware Attention (FAA)** module into the bottleneck stage.

The goal is to improve **texture preservation** and **reduce halo/ringing artifacts** for tasks such as image denoising and under-display camera (UDC) restoration.

---

## 🔍 Overview

- 🧱 **Backbone:** SCUNet-style encoder–decoder with ConvTrans/Swin-like blocks.
- 🎚 **FAA module:** Uses 2D FFT magnitude as a global frequency descriptor and applies channel-wise attention in the bottleneck.
- 🧮 **PyTorch implementation:** Clean, research-friendly code for easy modification and reproduction.
- 📊 **Evaluation helpers:** PSNR/SSIM and qualitative visualizations.

---

## 📷 Denoising Performance Example

Below is a qualitative comparison between the baseline SCUNet and our SCUNetWithFAA on a UDC-like noisy face image.

<p align="center">
  <img src="docs/figures/Figure_1s.png" width="800" alt="Denoising Performance Comparison">
</p>

- **Baseline SCUNet**: PSNR = 20.41 dB, SSIM = 0.7260  
- **SCUNetWithFAA**: PSNR = 21.73 dB, SSIM = 0.7637  

Our FAA module sharpens facial details and reduces the circular flare around the face, while improving both PSNR and SSIM.

---

## 🏗 Architecture

We provide two main models:

- `SCUNet` (baseline) – defined in `scunet_faa/models/network_scunet.py`
- `SCUNetWithFAA` – defined in `scunet_faa/models/network_scunet_faa.py`

`SCUNetWithFAA` keeps the same encoder–decoder as SCUNet, but applies a **FrequencyAwareAttention** block to the bottleneck features before passing them to the decoder.

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/scunet-faa.git
cd scunet-faa

# Option 1: pip
pip install -r requirements.txt

# Option 2: conda
conda env create -f environment.yml
conda activate scunet-faa

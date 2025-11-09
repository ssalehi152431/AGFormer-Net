# 🧠 AGFormer-Net: A Transformer-Guided Attention-Gated Framework for 3D Brain Tumor Segmentation

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)
---

## 📘 Overview
**AGFormer-Net** is a two-stage deep learning architecture for **3D multimodal brain tumor segmentation** from MRI volumes.  
It unifies **transformers**, **attention gates**, and **variational autoencoders (VAE)** to capture both global and local contextual information for precise delineation of glioma subregions—**Whole Tumor (WT)**, **Tumor Core (TC)**, and **Enhancing Tumor (ET)**—on the **BraTS-2020** benchmark dataset.

The model achieves **Dice scores of 87.09 % (WT), 80.32 % (TC), and 74.63 % (ET)**, surpassing existing U-Net-based baselines in both volumetric overlap and boundary accuracy.

---

## 💡 Motivation
Manual delineation of glioma regions in multi-modal MRI is time-consuming, prone to inter-observer variability, and unscalable for large datasets.  
Traditional CNN architectures struggle with long-range dependency modeling and irregular tumor morphologies.  
AGFormer-Net bridges this gap by combining:
- **Transformer modules** for capturing *global context*  
- **Attention gates** for *region-specific focus*  
- **VAE regularization** for *robust feature learning* and distributional consistency

This synergy enables more accurate segmentation of heterogeneous tumor structures and better generalization across patients.

---

## 🏗️ Architecture Summary
AGFormer-Net is a **two-stage cascaded encoder-decoder** framework:

### 🔹 Stage I – Transformer-Augmented Asymmetric U-Net + VAE
- **ResNet-based encoder** extracts hierarchical 3D features.  
- **Transformer bottleneck** (embedding dim = 512, depth = 4) models global dependencies.  
- **VAE branch** reconstructs the input volume for feature regularization.  
- Output: coarse segmentation map of tumor subregions.

### 🔹 Stage II – Attention-Gated Asymmetric U-Net
- Inputs: concatenation of MRI modalities + Stage I predictions (7-channel input).  
- **Attention gates** suppress irrelevant activations and refine boundaries, emphasizing enhancing tumor regions.  
- Produces the final high-resolution segmentation.

**Loss Function:**  
\[
L = L_{Dice} + 0.1\,L_{L2} + 0.1\,L_{KL}
\]
combining soft Dice loss, VAE reconstruction loss, and Kullback–Leibler divergence.

---

## 🧪 Dataset
- **Benchmark:** [BraTS 2020 Dataset](https://www.med.upenn.edu/cbica/brats2020/)  
- **Samples:** 369 train + 125 validation + 166 test cases  
- **Modalities:** T1, T1Gd, T2, FLAIR  
- **Resolution:** 240 × 240 × 155 voxels  
- **Labels:** Necrotic/Non-enhancing (1), Edema (2), Enhancing (4)

Preprocessing includes intensity normalization, patch cropping (128³ voxels), and 3D flipping augmentations.

---

## ⚙️ Implementation Details
- **Framework:** PyTorch  
- **Hardware:** Tesla P100 (16 GB GPU, Google Colab Pro)  
- **Optimizer:** Adam (lr = 1e-4, 50 epochs)  
- **Scheduler:** Polynomial decay \((1 – e/ N_e)^{0.9}\)  
- **Augmentations:** Random cropping, intensity scaling [0.9, 1.1], 3D axis flips  

---

## ▶️ Usage
```bash
# Clone the repository
git clone https://github.com/ssalehi152431/AGFormer-Net.git
cd AGFormer-Net

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Train Stage I
python train_stage1.py --dataset brats2020 --epochs 50 --batch_size 2

# Train Stage II
python train_stage2.py --dataset brats2020 --epochs 50 --batch_size 2 --load_stage1 results/stage1.pth

# Evaluate
python evaluate.py --weights results/final_model.pth --dataset brats2020

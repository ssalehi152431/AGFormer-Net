# AGFormer-Net: A Transformer-Guided Attention-Gated Framework for 3D Brain Tumor Segmentation

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)

## Overview
AGFormer-Net is a two-stage deep learning framework designed for 3D brain tumor segmentation from multimodal MRI scans.  
It integrates **transformer modules**, **attention gates**, and **variational autoencoders (VAE)** to achieve high-precision tumor delineation and robust generalization across heterogeneous data.

This model was proposed in the ICCCNT 2025 paper titled  
**“AGFormer-Net: A Transformer-Guided Attention-Gated Framework for 3D Brain Tumor Segmentation.”**

---

## Motivation
Manual tumor annotation is time-consuming and inconsistent across radiologists.  
While U-Net-based architectures have achieved significant success, they often fail to capture long-range dependencies and complex tumor boundaries.  

AGFormer-Net addresses these challenges by introducing:
- Transformer-based global context modeling
- Attention-gated feature refinement
- Variational autoencoder (VAE) regularization for robust latent representations

---

## Architecture
AGFormer-Net follows a **two-stage cascaded encoder-decoder design**:

### Stage I – Transformer-Augmented Asymmetric U-Net + VAE
- ResNet-based encoder extracts 3D hierarchical features.
- Transformer bottleneck captures global dependencies (embedding dimension = 512, depth = 4).
- A parallel VAE branch reconstructs input volumes for regularization.
- Produces an initial coarse segmentation.

### Stage II – Attention-Gated Asymmetric U-Net
- Input: concatenation of the original modalities and Stage I predictions.
- Attention gates selectively emphasize tumor boundaries.
- Outputs refined segmentation for **Whole Tumor (WT)**, **Tumor Core (TC)**, and **Enhancing Tumor (ET)**.

**Overall Loss:**
L = L_Dice + 0.1 * L_L2 + 0.1 * L_KL

---

## Dataset
- **Dataset:** [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/)
- **Training Samples:** 369 cases  
- **Validation Samples:** 125 cases  
- **Testing Samples:** 166 cases  
- **Modalities:** T1, T1Gd, T2, FLAIR  
- **Resolution:** 240 × 240 × 155 voxels  
- **Labels:** Necrotic/Non-enhancing (1), Edema (2), Enhancing (4)

**Preprocessing:**
- N4 bias correction
- Intensity normalization (z-score)
- Patch extraction (128³)
- Random flipping and rotation augmentations

---

## Implementation Details
- **Framework:** PyTorch
- **Hardware:** Tesla P100 (16 GB GPU)
- **Optimizer:** Adam (lr = 1e-4)
- **Scheduler:** Polynomial decay with power = 0.9
- **Batch Size:** 2
- **Epochs:** 50 per stage

---

## Training and Evaluation

### Clone the repository
```bash
git clone https://github.com/ssalehi152431/AGFormer-Net.git
cd AGFormer-Net
```

### Create environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Train Stage I
```bash
python train_stage1.py --dataset brats2020 --epochs 50 --batch_size 2
```

### Train Stage II
```bash
python train_stage2.py --dataset brats2020 --epochs 50 --batch_size 2 --load_stage1 results/stage1.pth
```

### Evaluate the final model
```bash
python evaluate.py --weights results/final_model.pth --dataset brats2020
```

Outputs include segmentation maps, Dice/HD95 metrics, and qualitative overlays stored in `/results/`.

---

## Results

| Model | WT (DSC %) | TC (DSC %) | ET (DSC %) | WT (HD95 mm) | TC (HD95 mm) | ET (HD95 mm) |
|:------|:------------:|:------------:|:------------:|:-------------:|:-------------:|:-------------:|
| 3D U-Net | 84.11 | 79.06 | 68.76 | 13.37 | 14.61 | 50.98 |
| V-Net | 84.63 | 75.26 | 61.79 | 20.41 | 21.18 | 47.70 |
| Residual 3D U-Net | 82.46 | 76.47 | 71.63 | 12.34 | 13.11 | 37.42 |
| **AGFormer-Net (Proposed)** | **87.09** | **80.32** | **74.63** | **6.47** | **10.47** | **16.72** |

AGFormer-Net significantly improves Dice and boundary precision, particularly for small enhancing tumor regions.

---

## Ablation Study

| Transformer Embedding Dim | WT (DSC %) | TC (DSC %) | ET (DSC %) |
|:--------------------------:|:-----------:|:-----------:|:-----------:|
| 384 | 81.35 | 73.02 | 69.95 |
| 512 | **87.09** | **80.32** | **74.63** |
| 768 | 82.55 | 81.56 | 70.50 |

The 512-dimensional embedding yielded the best performance-to-complexity tradeoff.

---

## Key Contributions
1. A hybrid architecture combining **Transformers + Attention Gates + VAE** for 3D segmentation.  
2. Dual-stage refinement that improves boundary accuracy and lesion completeness.  
3. Demonstrated superior performance on **BraTS 2020** against strong CNN baselines.  
4. Provides a flexible PyTorch implementation for brain tumor segmentation research.

---

## Citation
If you use this work, please cite:

> P. Datta, S. Salehin, A. J. Islam, S. Das, and S. Paul,  
> *“AGFormer-Net: A Transformer-Guided Attention-Gated Framework for 3D Brain Tumor Segmentation,”*  
> 16th IEEE International Conference on Computing, Communication and Networking Technologies (ICCCNT 2025), pp. 1–8, 2025.

---

## Authors
**Prasun Datta**, **Sultanus Salehin**, **Akib Jayed Islam**, **Sajeeb Das**, **Srijit Paul**  
*(BUET, IUT Bangladesh, NTNU Norway, NIT India)*  

Contact: [salehi@ncsu.edu](mailto:salehi@ncsu.edu)



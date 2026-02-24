# TAC-Net: Temporal Alignment and Contrastive Network for Cross-Subject Neural Manifold Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Tested-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of **TAC-Net** (Temporal Alignment and Contrastive Network), a novel geometric deep learning framework designed to project dynamic fMRI signals into a subject-invariant, low-dimensional continuous latent space. 

By integrating Shared Response Modeling (SRM), Gromov-Wasserstein (GW) optimal transport, and a temporal contrastive learning objective (InfoNCE), TAC-Net effectively mitigates severe individual anatomical heterogeneities and prevents global manifold collapse across continuous naturalistic stimuli.



## 🌟 Key Contributions

* **Gromov-Wasserstein Topological Anchor:** Aligns discrete brain regions (e.g., Schaefer 400 parcellation) into a generalized continuous 2D grid mapping via optimal transport, bridging the gap between functional topography and spatial convolution.
* **Temporal Contrastive Learning:** Utilizes the InfoNCE loss to establish a repulsive force separating discrete visual stimuli. This effectively avoids the manifold entanglement and local minima commonly observed in traditional Kullback-Leibler divergence approaches.
* **Robust Cross-Subject Generalization:** Demonstrates uniformly high Inter-Subject Correlation (ISC) synchronization and predictive classification accuracy (nearly 40% across 14 video categories) on completely novel individuals (Train vs. Test zero-shot setting).
* **Aesthetic Scientific Visualization:** Includes highly customized scripts for rendering elegant 3D neural trajectories and high-resolution GW coupling matrix heatmaps.

---

## 🛠️ Installation

Ensure you have Python 3.9+ installed. Clone this repository and install the required dependencies.

```bash
git clone [https://github.com/YourUsername/Temporal-Diverse-Club.git](https://github.com/YourUsername/Temporal-Diverse-Club.git)
cd Temporal-Diverse-Club
pip install -r requirements.txt

<div align="center">
  <h1>Sonny: Breaking the Compute Wall in Medium-Range Weather Forecasting</h1>
  <p><strong>Minjong Cheon</strong><sup>1</sup>, <strong>Jung-A Yang</strong><sup>2,*</sup></p>
  <p><em><sup>1</sup>Department of Computer Science, Korea National Open University</em><br>
  <em><sup>2</sup>Division of Civil and Environmental Engineering, College of Engineering, Konkuk University</em><br>
  <sup>*</sup>Corresponding author: ja0903@konkuk.ac.kr</p>
</div>

---

## 🚀 Overview
**Sonny** is an efficient Small Weather Model (SWM) based on the **StepsNet** architecture. While many Large Weather Models (LWMs) require massive TPU/GPU clusters, Sonny is designed to be accessible for academic groups with limited compute budgets.

### Key Highlights
* **Low Compute Barrier**: Trains on a **single NVIDIA A40 GPU** in **3.5 days** (84 GPU hours, 55 epochs).
* **High Performance**: Achieves competitive medium-range forecast skill compared to HRES and major AI baselines on **WeatherBench2**.
* **Enhanced Stability**: Uses an **Exponential Moving Average (EMA)** of the weights and a **five-epoch learning-rate cooldown** to stabilize long-term iterative rollouts without a fine-tuning stage.

---

## 🏗️ Architecture: StepsNet Design
Sonny utilizes a hierarchical transformer design that decouples atmospheric variables into two physically informed categories:

1.  **Step 1 (Slow Path)**: Processes the **Dynamics group** (U, V, Z, P) in a narrow, deep network to capture long-range spatial dependencies.
2.  **Step 2 (Fast Path)**: Fuses the refined dynamic features with the **Thermodynamics group** (T, Q) to model complex non-linear interactions.
---

## 📊 Experimental Results

### 1. Efficiency Comparison
Sonny demonstrates significant advantages in training time and resource requirements compared to state-of-the-art models.

| Model | Parameters | Training Hardware | Training Time |
| :--- | :---: | :---: | :---: |
| GraphCast | 36.7M | 32 x TPUv4 | 4 weeks |
| Pangu-Weather | 256M | 192 x V100 | 64 days |
| Stormer (ViT-S, same 1.5° grid) | 35.8M | 1 x A40 | 5.6 days |
| **Sonny (Ours)** | **23.6M** | **1 x A40** | **3.5 days** |

### 2. Impact of EMA and the Learning-Rate Cooldown
RMSE on the 355 daily starts of 2022, relative to the raw weights at epoch 50, mean over nine variables and ten lead days:
* EMA at epoch 50 (decay 0.9), **Sonny (epoch 50)**: **−3.4%**.
* Five-epoch cooldown with EMA (decay 0.999), **Sonny**: **−4.6%** (−6.6% on day 3, −2.1% on day 10).

### 3. Case Studies
Forecasts tracked on the 6 h rollout path, with IFS HRES on the same 1.5° grid.
* **Typhoon Nanmadol (2022)**: Mean track error of 69, 77, 105, 193, and 151 km at 24–120 h against the IBTrACS best track, compared with 82, 137, 169, 285, and 317 km for IFS HRES.
* **Winter Storm Elliott (2022)**: Mean position error of 439, 395, and 137 km for the starts on 19, 20, and 21 December against the ERA5 track, compared with 306, 313, and 90 km for IFS HRES. From the 19 December start, Sonny gives 977.0 hPa at 120 h, when ERA5 gives 971.3 hPa.

---

## ⚙️ Configuration
* **Data**: WeatherBench2 (ERA5) at $1.5^{\circ}$ spatial resolution, 120 × 240 grid (the south-pole row is dropped).
* **Vertical Levels**: 13 pressure levels (50hPa to 1000hPa).
* **Objective**: Randomized dynamics forecasting (predicting $\Delta_{\delta t}$ where $\delta t \in \{6, 12, 24\}$ hours).
* **Loss**: Pressure-weighted and latitude-weighted MSE.

---

## 📦 Weights and Usage
Weights, configuration, and normalization constants are at [huggingface.co/Francismj/sonny](https://huggingface.co/Francismj/sonny), where the model card shows how to load them with this code.

| File | Weights |
| :--- | :--- |
| `sonny_final.safetensors` | **Sonny**: EMA (decay 0.999) after the cooldown, evaluated in the revised paper |
| `model.safetensors` | **Sonny (epoch 50)**: EMA (decay 0.9) at epoch 50, the weights of arXiv:2603.21284v1 |

The code needs `torch`, `timm`, `xformers`, and a GPU.

---

<hr>
<h2 align="left">🙏 Acknowledgements</h2>

<p align="left">
  This project builds upon and leverages the excellent work from the following resources:
</p>

<ul>
  <li>
    <b>Data Processing & Inference:</b> 
    Inspired by the implementation in 
    <a href="https://github.com/tung-nd/stormer">
      <img src="https://img.shields.io/badge/GitHub-Stormer-181717?style=flat-square&logo=github" alt="GitHub Stormer">
    </a> 
  </li>
  <br>
  <li>
    <b>Research Paper:</b> 
    "Scaling transformer neural networks for skillful and reliable medium-range weather forecasting" 
    <a href="https://arxiv.org/abs/2312.03876">
      <img src="https://img.shields.io/badge/arXiv-2312.03876-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv Paper">
    </a>
  </li>
</ul>

<blockquote align="left">
  <i>Special thanks to the authors for their open-source contributions and insightful research.</i>
</blockquote>

## 📝 Citation
If you find this work useful, please cite our paper:

```bibtex
@article{cheon2026sonny,
  title={Sonny: Breaking the Compute Wall in Medium-Range Weather Forecasting},
  author={Cheon, Minjong},
  journal={arXiv preprint arXiv:2603.21284},
  year={2026}
}

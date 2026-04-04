<div align="center">
  <h1>Sonny: Breaking the Compute Wall in Medium-Range Weather Forecasting</h1>
  <p><strong>Minjong Cheon</strong></p>
  <p><em>Department of Computer Science and Engineering, Sejong University</em></p>
</div>

---

## 🚀 Overview
**Sonny** is an efficient Small Weather Model (SWM) based on the **StepsNet** architecture. While many Large Weather Models (LWMs) require massive TPU/GPU clusters, Sonny is designed to be accessible for academic groups with limited compute budgets.

### Key Highlights
* **Low Compute Barrier**: Can be trained to convergence on a **single NVIDIA A40 GPU** in approximately **5.5 days**.
* **High Performance**: Achieves competitive medium-range forecast skill compared to HRES and major AI baselines on **WeatherBench2**.
* **Enhanced Stability**: Uses **Exponential Moving Average (EMA)** during training to stabilize long-term iterative rollouts without expensive fine-tuning stages.

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
| **Sonny (Ours)** | **20.5M** | **1 x A40** | **5.5 days** |

### 2. Impact of EMA
Implementing EMA with a decay rate of 0.9 resulted in:
* An average error reduction of **3.34%** across all forecast periods.
* A peak error reduction of **4.75%** on Day 3.
* Consistently lower RMSE across nine meteorological variables.

### 3. Case Studies
* **Typhoon Nanmadol (2022)**: Track error of ~137.5 km at a 120-hour lead time, comparable to Pangu-Weather and GraphCast.
* **Winter Storm Elliott (2022)**: Captured rapid intensification despite $1.5^{\circ}$ resolution, with a central pressure error of only 4.9 hPa.

---

## ⚙️ Configuration
* **Data**: WeatherBench2 (ERA5) at $1.5^{\circ}$ spatial resolution.
* **Vertical Levels**: 13 pressure levels (50hPa to 1000hPa).
* **Objective**: Randomized dynamics forecasting (predicting $\Delta_{\delta t}$ where $\delta t \in \{6, 12, 24\}$ hours).
* **Loss**: Pressure-weighted and latitude-weighted MSE.

---

## 📝 Citation
If you find this work useful, please cite our paper:

```bibtex
@article{cheon2026sonny,
  title={Sonny: Breaking the Compute Wall in Medium-Range Weather Forecasting},
  author={Cheon, Minjong},
  journal={arXiv preprint arXiv:2603.21284},
  year={2026}
}

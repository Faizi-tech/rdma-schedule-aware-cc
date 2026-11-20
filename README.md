# Schedule-Aware Learned Rate Control for RDMA AI Fabrics

**Authors:** Abdul Samad (24I-0095) & Faizan Ahmed (24I-0059)  
**Affiliation:** FAST-NUCES Islamabad, BS Artificial Intelligence  
**Course:** Computer Networks (Phase 4 — Research Paper)

---

## Overview

This repository contains the full implementation of the paper:

> **"Schedule-Aware Learned Rate Control for RDMA Datacenter Fabrics Carrying Distributed AI Training Traffic"**

A lightweight 121-parameter learned rate controller for RDMA collective communication that:
- Anchors each sender at its **expected fair share** using the collective schedule
- Applies a **learned correction** (8→12→1 MLP, trained with Evolution Strategies)
- Reduces mean CCT slowdown by **34.8% vs DCQCN** and **19.2% vs Barre-style**
- Eliminates **PFC pauses** (1.32% → 0.00%)

---

## Files

| File | Description |
|------|-------------|
| `cnsim.py` | Flow-level discrete-event simulator (100 Gbps bottleneck, ECN, PFC) + all controllers |
| `scen.py` | Scenario generator and burst forecaster |
| `train.py` | Evolution Strategies training loop |
| `experiments.py` | All experiments → `results.json` + figures |
| `mkreport.py` | Generates LaTeX tables from `results.json` |
| `theta_*.npz` | Trained policy weights for each ablation variant |
| `mlp_pred.pkl` | Trained burst forecaster (scikit-learn MLP) |
| `results.json` | All experiment results (used directly in the paper) |
| `figs/` | All paper figures |

---

## Requirements

```
Python 3.12+
numpy
numba
scipy
scikit-learn
matplotlib
```

Install: `pip install numpy numba scipy scikit-learn matplotlib`

---

## Reproduce All Results

```bash
# 1. Train the burst forecaster
python train.py mlp

# 2. Train all ES policies (~1 min each)
python train.py full_mlp sched nopred ma oracle fct_reward nophase

# 3. Run all experiments → results.json + figs/
python experiments.py

# 4. (Optional) Regenerate LaTeX tables
python mkreport.py
```

---

## Random Seeds

| Component | Seed |
|-----------|------|
| Validation scenarios | 999 |
| Random test set | 777 |
| Shifted test set | 888 |
| Named scenarios | 100–109 |
| Forecaster training | 0 |
| Main ES policy | 0 |

---

## Key Results (Random Test Set, 40 scenarios)

| Method | CCT Slowdown | PFC% |
|--------|-------------|------|
| DCQCN | 2.044 | 1.32% |
| HPCC-style | 1.894 | 0.46% |
| Barre-style | 1.649 | 0.67% |
| **Ours (predictive)** | **1.333** | **0.00%** |
| **Ours (schedule-aware)** | **1.278** | **0.00%** |

All 16 paired Wilcoxon comparisons: p < 0.001 (Holm-corrected).

---

## Citation

If you use this code, please cite:

```bibtex
@article{samad2026scheduleaware,
  author    = {Samad, Abdul and Ahmed, Faizan},
  title     = {Schedule-Aware Learned Rate Control for RDMA Datacenter Fabrics Carrying Distributed AI Training Traffic},
  year      = {2026},
  note      = {FAST-NUCES Islamabad, Computer Networks Phase 4}
}
```


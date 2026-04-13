# About

Experimental findings can be found in this directory.

# Link Table

The relevant links for different experiments are compiled in the table below.

|Description|Link  |
|--|--|
| Plot 1: Single-variate Richardson extrapolation of different orders are shown. Depolarizing noise is applied only to the multi-qubit time-evolution gate. |[Jupyter notebook](recent/experiment12[time-depol-time-evo-noisy_p1e-3_tmax_various]/experiment12[time-depol-time-evo-noisy_p1e-3_tmax_various].ipynb), [Raw data (JSON)](recent/experiment12[time-depol-time-evo-noisy_p1e-3_tmax_various]/data) |
|Plot 2: VQE/ZNE performance for different choices of the initial time parameter upper bound $T_{\text{max}}$.| [Jupyter notebook](recent/experiment12[time-depol-time-evo-noisy_p1e-3_tmax_various]/experiment12[time-depol-time-evo-noisy_p1e-3_tmax_various].ipynb), [Raw data (JSON)](recent/experiment12[time-depol-time-evo-noisy_p1e-3_tmax_various]/data) |
|Plot 3: Single-variate ZNE of different orders vs $\Delta p$.|[Jupyter notebook](recent/experiment15[depol-time-evo-noisy_tmax_20_various_p]/experiment15[depol-time-evo-noisy_tmax_20_various_p].ipynb), [Raw data (JSON)](recent/experiment15[depol-time-evo-noisy_tmax_20_various_p]/data/)|
|Plot 4: Multivariate ZNE|[Jupyter notebook](recent/experiment14[depol-time-evo-noisy_variousorderzne_tmax_20]/experiment14[depol-time-evo-noisy_variousorderzne_tmax_20].ipynb), [Raw data (JSON)](recent/experiment14[depol-time-evo-noisy_variousorderzne_tmax_20]/data/)|




# Indirect-Control VQE and ZNE Error Mitigation: How to Use the Code

Download the code from [v2 release](https://github.com/arijit-ship/indirect-zne/releases/tag/v2.0).

## 🛠️ Installation

- **Python Version:** `3.11`  
- To install dependencies, run:  
```bash
  pip install -r requirements.txt
```
- On Windows, install minimum dependencies via:
```bash
  pip install -r requirements.winmin.txt
```
## ⚙️ Usage

The `main.py` is used for VQE optimization, redundant circuit runs, and finally performing ZNE.

To run the program, use:  
```bash
  python3 main.py <config.yml>
```
A faster multi-core script can alternatively be run for VQE optimization (the redundant circuit run and ZNE can be done using `main.py`):
```bash
  python3 vqe-mcore.py <config.yml>
```

## 📋 Sample Configuration File

([Sample experimental config files](../config_samples/))
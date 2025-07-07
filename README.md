<h1 align="center">🌳 Nested Tree Reduction via Optimal Transport</h1>

<p align="center"><em>
A fast, modular, and production-ready implementation of scenario tree reduction based on the nested distance and Wasserstein barycenters.
</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-green"/>
  <img src="https://img.shields.io/badge/build-passing-brightgreen"/>
  <img src="https://img.shields.io/badge/platform-MPI%20%7C%20CPU-lightgrey"/>
  <img src="https://img.shields.io/github/last-commit/dan-mim/Nested_tree_reduction"/>
  <img src="https://img.shields.io/github/repo-size/dan-mim/Nested_tree_reduction"/>
</p>

---

## 📘 Overview

This repository implements a new, efficient method for scenario tree reduction based on the **nested distance**, inspired by the article:

> **"A scalable method for stochastic process approximation using Wasserstein barycenters"**  
> Daniel Mimouni, 2024 — [📄 PDF](https://dan-mim.github.io/files/reduction_tree.pdf)

The method is designed for practical use in stochastic decision-making under uncertainty, and is already used by **IFP Energies Nouvelles (IFPEN)** for **energy decision management** applications.

---

## 🚀 Key Contributions

- ✅ Efficient and modular implementation of **Kovacevic–Pichler reduction**:
  > Kovacevic, R.M., & Pichler, A. (2015). Tree approximation for discrete time stochastic processes: a process distance approach. *Annals of Operations Research*, 235(1), 395–421.

- ⚡ Fast computation of nested barycenters using **optimal transport solvers**:
  - [IBP (Sinkhorn)] — entropic method
  - [MAM (Method of Averaged Marginals)] — exact projection method

- 🧠 Insight: the **reduction of a scenario tree** can be expressed as a **succession of Wasserstein barycenter problems**.  
  At each stage, an optimal subtree corresponds to the **barycenter of original subtrees**, weighted by their probability.

<p align="center">
  <img src="figs/nested_barycenter_explained.png" width="650"/>
</p>

---

## 📦 Package Structure

The core implementation is organized in the `pyreductree` package:

```
pyreductree/
├── __init__.py                 # Package interface
├── core.py                     # Main reduction logic (recursive & iterative)
├── barycenters.py              # MAM and IBP solvers for barycenter computation
├── cost_functions.py           # Wasserstein metrics and scenario cost matrices
├── io.py                       # Load/save tree data, formats, and datasets
├── metrics.py                  # Nested distance computation
├── utils.py                    # Tree manipulation, flattening, logging
```

Additionally:

- `mpi_reduction/`: Parallelized version using **MPI** (`mpi4py`) for large-scale use
- `examples/`: Demonstrations and real-world instances
- `figs/`: Visual outputs and benchmark figures

---

## 🔧 Installation

```bash
git clone https://github.com/dan-mim/Nested_tree_reduction.git
cd Nested_tree_reduction
conda create -n treereduct python=3.10
conda activate treereduct
pip install -r requirements.txt
conda install -c conda-forge mpi4py  # if you want to run the MPI version
```

---

## 📈 Applications

This method has been successfully applied in:
- ⚡ **Energy system design** (used by IFPEN)
- 📉 **Stochastic optimization problems**
- 🏭 **Industry-grade scenario compression**
- 🎯 Any process requiring fast and controlled tree reduction under **probability distribution distance metrics**

<p align="center">
  <img src="figs/numerical_app1.png" width="400"/>
  <img src="figs/numerical_app2.png" width="400"/>
</p>

---

## 🧠 Reference

If you use this project in your research or applications, please cite:

```
@article{mimouni2024nested,
  title={A scalable method for stochastic process approximation using Wasserstein barycenters},
  author={Mimouni, Daniel},
  year={2024},
  journal={Preprint},
  url={https://dan-mim.github.io/files/reduction_tree.pdf}
}
```

And the original method:

```
@article{kovacevic2015tree,
  title={Tree approximation for discrete time stochastic processes: a process distance approach},
  author={Kovacevic, R.M. and Pichler, A.},
  journal={Annals of Operations Research},
  volume={235},
  number={1},
  pages={395--421},
  year={2015}
}
```

---

## 📜 License

MIT © 2024 — Daniel Mimouni

---
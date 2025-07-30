# An em algorithm for quantum Boltzmann machines

[![arXiv](https://img.shields.io/badge/quant--ph-arXiv:2507.21569-b31b1b.svg?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2507.21569)

This repository contains the code for the paper:

> "An em algorithm for quantum Boltzmann machines"  
> *Takeshi Kimura, Kohtaro Kato, Masahito Hayashi*  
> [arXiv:2507.21569](http://arxiv.org/abs/2507.21569) (2025)

## Program Description

### models

- [sqRBM_em](./src/models/sqRBM_em.py): Contains the `sqRBM_em` class, which implements a semi-quantum RBM (sqRBM) with em algorithm.
- [RBM_em](./src/models/RBM_em.py): Contains the `RBM_em` class, which implements a classical RBM with em algorithm.

### notebooks

- [train.ipynb](./example/notebooks/train.ipynb): A Jupyter notebook for training sqRBM and classical RBM using em algorithm and gradient descent.
- [eval.ipynb](./example/notebooks/eval.ipynb): A Jupyter notebook for evaluating the results of sqRBM and classical RBM using em algorithm and gradient descent.

## Acknowledgements

This project is based on the following open-source software:

- **qbm** by cameronperot
  - Licensed under the MIT License.
  - Link: https://github.com/cameronperot/qbm

The data distribution code is based on the following open-source software:

- **semi-quantum-RBM** by mariademidik
  - Licensed under the Apache License 2.0.
  - Link: https://github.com/mariademidik/semi-quantum-RBM
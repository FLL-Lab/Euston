# Euston: Efficient and User-Friendly Secure Transformer Inference

## Overview
Euston is an efficient and user-friendly framework for secure Transformer inference, offering dual implementations optimized for CPU and GPU platforms. This repository contains the reference implementation accompanying our paper ["Euston: Efficient and User-Friendly Secure Transformer Inference"](d). Our framework achieves efficient homomorphic evaluation while maintaining user-friendly deployment across heterogeneous computing environments.

## Table of Contents
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Data Preparation](#data-preparation)
- [Platform-Specific Implementations](#platform-specific-implementations)
- [Directory Structure](#directory-structure)
- [References](#references)

## Key Features
- Dual-platform support with CPU/GPU optimizations
- Secure inference through integration with FHE libraries
- Efficient and user-friendly matrix multiplications
- Efficient and user-friendly nonlinear functions

## Getting Started

### Prerequisites
- Python 3.8+ for data generation
- CMake 3.20+
- Platform-specific dependencies (see subdirectory READMEs)

### Data Preparation
Generate prerequisite datasets:
```bash
cd Data
python generate***.py
```

## Platform-Specific Implementations
- **CPU Implementation**: See [Euston_CPU/README.md](Euston-CPU/Readme.md) for Eigen/SEAL dependencies and build instructions
- **GPU Implementation**: See [Euston_GPU/README.md](Euston-GPU/Readme.md) for CUDA requirements and compilation guide

## Directory Structure
```
├── Data/               # Dataset generation scripts
├── Euston_CPU/         # CPU-Comparison implementation
├── Euston_GPU/         # GPU-Comparison implementation
└── ...
```

## References

- Built upon [NEXUS](https://github.com/zju-abclab/NEXUS.git)  secure compression  techniques

- Incorporates [SEAL](https://github.com/microsoft/SEAL.git)  for CPU acceleration

- Incorporates [PhantomFHE](https://github.com/encryptorion-lab/phantom-fhe.git)  for GPU acceleration

# KinaseNet

![label1](https://img.shields.io/badge/version-v1.0.0-yellow)	![label2](https://img.shields.io/badge/license-MIT-green)

**| [Overview](#overview) | [Installation](#installation) | [Usage](#usage) | [Data](#data) | [Output](#output) | [Citation](#citation) |**

<!---**![Figure1]()** --->

## Overview

## Installation
We recommend using a dedicated conda environment.
```bash
conda create -n kinasenet python=3.10
conda activate kinasenet
```

### Install PyTorch
Install PyTorch according to your system and CUDA version from the [official website](https://pytorch.org/get-started/locally/).

### Install KinaseNet
```bash
git clone https://github.com/Yizhi-Zhang/KinaseNet.git
cd KinaseNet
pip install .
```

### Verify Installation
Open Python in any directory and try:
```python
import kinasenet
```
If no error is raised, the installation was successful ✅

## Usage
We provide a usage example in the Jupyter notebook [tutorial.ipynb](./tutorial.ipynb), which demonstrates how to:

- Load preprocessed phosphoproteomic data

- Run KinaseNet for inference

- Extract kinase-substrate relationships

- Estimate sample-specific kinase activity

## Data
The [`example_data/`](./example_data/) directory contains input files used to demonstrate KinaseNet's pipeline:
```text
example_data/
├── example.feather              # Input phosphoproteomic data
├── prior_example.tsv            # Input prior kinase-substrate relationships
├── preprocessed_data/
│   ├── data.parquet             # Preprocessed phosphoproteomic data
│   └── prior.parquet            # Preprocessed prior kinase-substrate relationships
```

### 📌 Input Files
Two files are essential for running KinaseNet:

- `example.feather`  
  A phosphoproteomic expression matrix (e.g., log2 intensity values from [CPTAC](https://pdc.cancer.gov/pdc/browse/filters/program_name:Clinical%20Proteomic%20Tumor%20Analysis%20Consortium)), where each **row represents a phosphosite** and each **column represents a sample**.  
  This matrix has been preprocessed by:
    - Filtering: Only retaining phosphosites expressed in ≥30% of samples
    - Missing value imputation: Performed using K-nearest neighbors (KNN)

- `prior_example.tsv`  
  A prior knowledge matrix of kinase-substrate relationships (e.g., from [PhosphoSitePlus](https://www.phosphosite.org/homeAction) or computational predictions), where each **row represents a phosphosite** and each **column represents a kinase**.  
  A value >0 indicates that the kinase is known or predicted to regulate the phosphosite.

### 🧪 Preprocessed Files
After running KinaseNet’s preprocessing module, the following files will be generated and prepared for model input:

- `data.parquet`  
  Normalized phosphoproteomic expression matrix.

- `prior.parquet`  
  Formatted prior matrix.

## Output
After running KinaseNet, results will be saved under a directory named according to the dropout configuration (e.g., [`example_results/dp1_0_dp2_0`](./example_results/dp1_0_dp2_0/)).

### 📂 Typical Output Structure
```text
example_results/
└── dp1_0_dp2_0/
    ├── model/                     # Trained model checkpoints
    ├── performance.csv            # Final evaluation metrics
    ├── performance_tmp.csv        # Intermediate metrics during training
    ├── run_log/                   # Training log files
```

### 📄 Output Descriptions
- `model/`  
  Contains trained PyTorch model weights and checkpoints for future reuse or evaluation.

- `performance.csv` / `performance_tmp.csv`  
  Evaluation metrics (e.g., R<sup>2</sup>, AUPRC) computed during training.  
  Both files contain the same content upon training completion.

- `run_log/`  
  Training logs and loss curves for reproducibility and diagnostic purposes.

## Citation
If you use KinaseNet in your research, please cite:
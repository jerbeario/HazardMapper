# HazardMapper
HazardMapper is an open-source tool designed to analyze, process, and model hazards based on geospatial conditioning factors datasets. It includes components for data preprocessing, partitioning, model training and evaluation, hyperparameter sweeps, map generation and SHAP explanations making it easier to generate hazard maps for various regions.

- **HazardMapper/**
  - Contains the main source code including modules for analysis ([analysis.py](HazardMapper/analysis.py)), architecture ([architecture.py](HazardMapper/architecture.py)), dataset management ([dataset.py](HazardMapper/dataset.py)), modeling ([model.py](HazardMapper/model.py)), partitioning ([partition.py](HazardMapper/partition.py)), preprocessing ([preprocess.py](HazardMapper/preprocess.py)), and various utility functions ([utils.py](HazardMapper/utils.py)).
  

- **Experiments/**
  - Includes shell scripts for running experiments (e.g. [run.sh](Experiments/run.sh) and [preprocess.sh](Experiments/preprocess.sh)).

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/your_username/HazardMapper.git
    cd HazardMapper
    ```

2. **Set up the environment:**

    If using conda, run:

    ```sh
    conda env create -f environment.yml
    conda activate hazardmapper
    ```

3. **Install the package locally:**

    ```sh
    pip install -e .
    ```

## Requirements
- Data: 
    To use this package, a data directory needs to be provided in the dataset.py script, with the npy arrays for modelling. By default it will look in `Input/Europe/`.
- Partition Map
    For the partition.sh script to run, it needs a `partition_map/sub_countries_rasterized.npy` in the data folder. 

## Basic Usage

- **Data Preprocessing:**  
    Run the preprocessing script to prepare your datasets:

    ```sh
    bash Experiments/preprocess.sh
    ```

- **Run Experiments:**  
    Start an experiment run with:

    ```bash
    bash Experiments/run.sh
    ```
   
- **Downscalling:**
    For easier testing and development, downscaling the data is suggested. The `utils.py` script does this downscaling and should be run after preprocessing and partitioning. 

- **Snellius:**
    For usage on snellius, clone the git or transfer the `HazardMapper/` and `Experiments/` directories in the `Suceptibility/` directory. Create the environment using the instructions above and run the experiments with `sbatch Experiments/example.sh`.

## Model Module Overview

The `model.py` module is the core of HazardMapper’s modeling functionality. It provides the classes and functions needed to build, train, evaluate, and interpret both traditional and deep learning hazard susceptibility models.

### Key Components

- **Argument Parsing:**  
  The module uses Python’s `argparse` to define command-line arguments that let you customize the model and training configuration. For example:  
  `-n`/`--name` sets the experiment name, `-z`/`--hazard` selects the hazard type, `-b`/`--batch_size` specifies the training batch size, etc.

- **Model Classes:**
  - **Baseline:**  
    Implements traditional machine learning models (Logistic Regression, Random Forest, or MLP) for pixel-wise classification. Note that for baseline models, only a patch size of 1 is supported.
    
  - **HazardModel:**  
    Implements deep learning models using PyTorch. It supports several architectures:
    - **MLP**
    - **CNN**
    - **SimpleCNN**
    - **SpatialAttentionCNN**

- **Training and Evaluation:**  
  The module implements a complete training pipeline:
  - Loading and partitioning datasets using associated data loader classes.
  - Defining the model architecture and training loops.
  - Monitoring training with early stopping and logging metrics.
  - Saving the best model and exporting to ONNX format.
  - Evaluating model performance with metrics such as accuracy, precision, recall, F1 score, AUROC, average precision, and MAE.

- **Advanced Features:**  
  - **Hyperparameter Optimization (Sweep):**  
    You can enable a hyperparameter sweep (using Weights & Biases) with the `--sweep` flag. This is supported only for PyTorch-based architectures.
    
  - **Hazard Map Generation:**  
    By specifying the `--map` flag, the module creates a hazard susceptibility map for the region using model predictions.
    
  - **Model Explanation:**  
    For deep learning models, the `--explain` flag computes SHAP values to provide model explainability.

- **Model Manager:**  
  The `ModelMgr` class is responsible for:
  - Configuring and coordinating the different model types.
  - Managing output directories, logging, and folder structure for saving results.
  - Integrating evaluation, model saving, and logging results (both locally and to Weights & Biases).

### Command-Line Example

An example command to run a training instance with the desired configuration looks like:

```sh
python HazardMapper/model.py -n "MyExperiment" -z "landslide" -b 1024 -p 5 -a "SimpleCNN" -e 5 --explain
```


## Partition Module Overview

The `partition.py` module handles the creation and management of partition maps for hazard data in Europe. It enables you to:
- **Filter Hazard Occurrences:** Only include regions with hazard data.
- **Erode Partition Borders:** Use binary erosion (with a configurable kernel size) to remove border cells and reduce data leakage during patch sampling.
- **Balance Partitions:** Downsample non-hazard cells to balance the dataset within each split (train, validation, test).
- **Sample the Partition Map:** Randomly select a subset of partition samples to match a desired sample size.

### Argument Parsing in partition.py

The module uses Python’s `argparse` to define command-line arguments that let you customize the partition mapping process. For example:
- `-z`/`--hazard` specifies the hazard type (e.g., flood, wildfire, landslide) for which the partition map is generated.
- `-n`/`--n_samples` sets the number of samples to downsample the partition map, with a default of 1,000,000.


## License

HazardMapper is open source and available under the [The GNU General Public License v3.0](LICENSE).
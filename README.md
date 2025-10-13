# HazardMapper
HazardMapper is an open-source tool designed to analyze, process, and model hazards based on geospatial conditioning factors datasets. It includes components for data preprocessing, partitioning, model training and evaluation, hyperparameter sweeps, and map generation making it easier to generate hazard maps for various regions.

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

## Usage

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



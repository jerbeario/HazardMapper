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

## Weights & Biases (wandb) Login

To enable experiment tracking and logging with Weights & Biases, you need to log in with your API key. Follow these steps:

1. **Obtain Your API Key:**
   - Sign up or log in at [Weights & Biases](https://wandb.ai).
   - Navigate to your account settings and copy your API key.

2. **Login Via Command-Line:**
   - Run the following command in your terminal:
     ```sh
     wandb login YOUR_API_KEY_HERE
     ```
   - Replace `YOUR_API_KEY_HERE` with your actual API key.

3. **Verify Login:**
   - Once logged in, your experiments will automatically sync with your wandb account.

Logging in ensures that metrics, model checkpoints, and other experiment details are stored and visualized online.

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

## Module Overviews

### Dataset

The `dataset.py` module defines a custom dataset and helper functions to load hazard-specific features and labels as image patches for model training and evaluation. Key features include:

- **Path Configuration:**  
  All file paths for raw inputs, preprocessed variables, hazard maps, and partition maps are defined in structured dictionaries (e.g., `raw_paths`, `var_paths`, `hazard_map_paths`, `label_paths`, and their downscaled versions). This ensures consistency across the pipeline and makes it easy to update data locations.

- **Custom Dataset Class (`HazardDataset`):**  
  This class extends PyTorch's `Dataset` to:
  - Load multiple continuous and categorical features as channels.
  - Handle patch extraction from large geospatial arrays by applying appropriate padding.
  - Binarize hazard labels (except for multi-hazard cases).
  - Validate input by ensuring that the specified hazard and variable names exist in the defined paths.

- **Balanced Batch Sampling:**  
  The module includes the `BalancedBatchSampler` class, which ensures that each training batch has a balanced number of positive (hazard occurrence) and negative samples. This is crucial for training on imbalanced hazard data, where the number of non-hazard instances usually far exceeds the occurrences.

- **Index Conversion Helpers:**  
  For efficient handling of spatial indices, helper functions `index2d_to_1d` and `index1d_to_2d` convert between 2D spatial coordinates and their flattened 1D indices. This is especially useful when dealing with large rasterized datasets.

By encapsulating data loading and preprocessing in a dedicated module, HazardMapper ensures seamless integration with PyTorch's data pipelines, making it easier to experiment with different input configurations and model architectures.

### Preprocess

The `preprocess.py` module provides a pipeline for cleaning and transforming raw geospatial data stored in `.npy` files. The main tasks performed by this module include:

- **Cleaning Maps:**  
  The `clean_map` function masks water bodies and out-of-bounds areas by setting their values to NaN based on landcover and elevation data.

- **Normalization:**  
  The `normalize` function uses `MinMaxScaler` to scale data values to the range [0, 1]. This is applied to most input variables to prepare them for subsequent modeling stages.

- **Label Encoding:**  
  The `label_encode` function converts categorical data (e.g., landcover types) into numerical labels while preserving NaN values for missing data.  

#### Processing Workflow

When executed, the module:
1. Loads raw data files for various environmental variables from predefined paths.
2. Applies a log transformation to selected variables (e.g., elevation, slope) to handle skewed distributions.
3. Normalizes all variables and, if needed, applies label encoding for categorical variables.
4. Cleans each variable by masking water bodies and out-of-bounds areas based on the landcover and elevation data.
5. Saves the processed data back as `.npy` files to specified output paths.

#### Command-Line Usage

To run the entire preprocessing pipeline, simply execute:

```sh
python HazardMapper/preprocess.py
```

This script will iterate through the predefined list of variables, apply the necessary transformations, and save the processed files for later use in model training and evaluation.


### Partition

The `partition.py` module handles the creation and management of partition maps for hazard data in Europe. It enables you to:
- **Filter Hazard Occurrences:** Only include regions with hazard data.
- **Erode Partition Borders:** Use binary erosion (with a configurable kernel size) to remove border cells and reduce data leakage during patch sampling.
- **Balance Partitions:** Downsample non-hazard cells to balance the dataset within each split (train, validation, test).
- **Sample the Partition Map:** Randomly select a subset of partition samples to match a desired sample size.

#### Argument Parsing 

The module uses Python’s `argparse` to define command-line arguments that let you customize the partition mapping process. For example:
- `-z`/`--hazard` specifies the hazard type (e.g., flood, wildfire, landslide) for which the partition map is generated.
- `-n`/`--n_samples` sets the number of samples to downsample the partition map, with a default of 1,000,000.


### Model

The `model.py` module is the core of HazardMapper’s modeling functionality. It provides the classes and functions needed to build, train, evaluate, and interpret both traditional and deep learning hazard susceptibility models.

#### Key Components

- **Argument Parsing:**  
The module uses Python’s `argparse` to define command-line arguments that let you customize the model configuration and training process. For example:

- `-n`/`--name`:  
  Sets the experiment name.  
  *Default:* `"HazardMapper"`

- `-z`/`--hazard`:  
  Specifies the hazard type. This could be set to things like `"landslide"`, `"wildfire"`, or `"flood"`.  
  *Default:* `"landslide"`  

- `-b`/`--batch_size`:  
  Defines the batch size for training, i.e., the number of samples processed simultaneously.  
  *Default:* `1024`

- `-p`/`--patch_size`:  
  Determines the patch size for the model input. This refers to the dimensions of the input data patches provided to the model.  
  *Default:* `5`

- `-a`/`--architecture`:  
  Selects the model architecture. The options include:  
  - Baseline models: `"LR"`, `"RF"`, `"MLP"`  
  - Deep learning architectures: `"CNN"`, `"SimpleCNN"`, `"SpatialAttentionCNN"`,`"CNN_GAP"`, `"CNN_GAPatt"`
  *Default:* `"CNN"`

- `-e`/`--epochs`:  
  Specifies the number of training epochs, which is the number of full passes through the training dataset.  
  *Default:* `5`

- `--sweep`:  
  A flag (boolean) to enable hyperparameter optimization (sweep) using tools like Weights & Biases.  
  *Default:* `False`  
  *Usage:* Include this flag if you wish to perform a hyperparameter sweep.

- `--map`:  
  A flag to trigger the creation of a hazard map after training. This automates the map generation process with the trained model.  
  *Default:* `False`

- `--explain`:  
  A flag to compute SHAP (SHapley Additive exPlanations) values for model explainability, helping to interpret model predictions.  
  *Default:* `False`

- **Model Classes:**
  - **Baseline:**  
    Implements traditional machine learning models (Logistic Regression, Random Forest, or MLP) for pixel-wise classification. Note that for baseline models, only a patch size of 1 is supported.
    
  - **HazardModel:**  
    Implements deep learning models using PyTorch. It supports several architectures:
    - **MLP**
    - **CNN**
    - **SimpleCNN**
    - **SpatialAttentionCNN**
    - **CNN_GAP**
    - **CNN_GAPatt**


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

#### Command-Line Example

An example command to run a training instance with the desired configuration looks like:

```sh
python HazardMapper/model.py -n "MyExperiment" -z "landslide" -b 1024 -p 5 -a "SimpleCNN" -e 5 --explain
```

### Architecture

The `architecture.py` module provides various architectures for hazard susceptibility modeling. These include both deep learning models built with PyTorch and traditional baseline models implemented with scikit-learn. The available architectures are:

- **Baseline Models (scikit-learn):**
  - **Logistic Regression (LR):**  
    Implements a logistic regression model for pixel-wise classification. It is simple and interpretable, but only supports a patch size of 1.
  - **Random Forest (RF):**  
    Uses an ensemble of decision trees for robust classification. Like LR, it only supports pixel-wise classification with a patch size of 1.

- **Deep Learning Models (PyTorch):**
  - **MLP:**  
    A fully connected neural network designed to process 1D feature vectors. Useful as a baseline for non-spatial inputs.
  - **CNN:**  
    A basic convolutional neural network that applies convolutional filters to capture spatial patterns in input patches.
  - **SimpleCNN:**  
    A lightweight CNN that balances complexity and performance, designed for patch-based spatial data.
  - **SpatialAttentionCNN:**  
    Incorporates a spatial attention mechanism to focus on important areas in the input data.
  - **CNN_GAP:**  
    A CNN with Global Average Pooling (GAP) to create robust, patch-size agnostic representations.  
    **Note:** This architecture is one of the most stable models in the system.
  - **CNN_GAPatt:**  
    Combines convolutional feature extraction with an attention mechanism and GAP.  
    **Note:** This architecture is also among the most stable options.
  - **CNNatt:**  
    A variant that applies spatial attention after the convolutional layers for improved feature emphasis.

Depending on your experimental requirements and the nature of your input data, you can select the appropriate architecture. For robust and stable deep learning performance, **CNN_GAP** and **CNN_GAPatt** are recommended, while **LR** and **RF** serve as quick and interpretable baselines using scikit-learn.


### Utils 

The `utils.py` module provides a collection of utility functions for preprocessing and visualizing geospatial data. These functions help with tasks such as downscaling maps, converting raster files to NumPy arrays, creating water masks, and plotting data on maps. Key functions include:

- **Downscaling Maps:**  
  The `downscale_map(path)` function takes a raster map (stored as a NumPy array) and downsamples it by a defined factor (default factor is 10). This function prints the original and downscaled shapes, and saves the downscaled array as a new `.npy` file.

- **Converting TIF Files to Numpy Arrays:**  
  The `tif_to_npy(tif_file, npy_file)` function reads a `.tif` file (using the `rasterio` library) and converts it into a NumPy array. The resulting array is then saved to disk in `.npy` format.

- **Creating Water Masks:**  
  The `make_water_mask(downsample_factor=1)` function processes landcover data to create a binary water mask. Pixels with a specific landcover value (e.g., 210 for water) are marked, and the mask is saved for reuse.

- **Plotting Numpy Arrays on Maps:**  
  The `plot_npy_arrays(...)` function provides a robust mechanism for visualizing NumPy arrays over a map. It supports:
  - Multiple plot types (e.g., continuous, partition, bins, categorical).
  - Downsampling for faster plotting.
  - Logarithmic transformations and debugging options.
  - Overlaying additional features such as water masks.
  
  This function is useful for visualizing hazard maps, partition maps, and other geospatial data.

- **Plotting a Grid of Maps:**  
  The `plot_maps_grid(...)` function generates a multi-panel grid (e.g., 4×4) showcasing different conditioning factors. It covers:
  - Downsampling of input arrays.
  - Customizable color maps and grid layouts.
  - Shared axis labels and a common colorbar.
  
- **Data Normalization:**  
  The `normalize_label(hazard_map, threshold=0.99)` function normalizes a given hazard map to the range [0, 1] based on a specified percentile threshold. This is particularly useful when preprocessing hazard data for visualization or further analysis.

#### Command-Line Usage

The `utils.py` module also supports command-line execution for common tasks. For example:

- **Downscaling Raster Maps:**  
  Run the module with the `--downscale` flag to downscale all maps defined in your variable and label paths:
  ```sh
  python HazardMapper/utils.py --downscale
  ```

- **Plotting a Grid of Maps:**  
  Run the module with the `--plot_grid` flag to automatically generate and save a grid of plots:
  ```sh
  python HazardMapper/utils.py --plot_grid
  ```

These commands leverage the defined paths in the module (such as `var_paths`, `label_paths`, and `partition_paths`) to locate and process the relevant data files.

This module streamlines the preprocessing and visualization steps, making it easier to work with large geospatial datasets and to quickly generate visual insights.

## License

HazardMapper is open source and available under the [The GNU General Public License v3.0](LICENSE).


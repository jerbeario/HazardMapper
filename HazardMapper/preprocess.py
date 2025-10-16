"""HazardMapper - Preprocessing Pipeline.
========================
This module provides functions for normalizing and transforming .npy files.
"""
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from HazardMapper.dataset import var_paths, raw_paths, label_paths

def clean_map(data:np.ndarray, landcover:np.ndarray, elevation:np.ndarray) -> np.ndarray:
    """
    Masks water bodies in the input numpy array by setting their values to NaN and sets out-of-bounds areas to NaN.
    Args:
        npy_file_path (str): Path to the input .npy file.
        output_file_path (str): Path to save the cleaned .npy file.
    Returns:
        np.ndarray: The cleaned numpy array with water bodies masked as NaN.
    """
    # Fill nans with -1
    data = np.where(np.isnan(data), -1, data)

    # Define water types based on landcover classification
    water_types = [210]  

    # Mask water areas and out-of-bounds areas
    water_mask = np.isin(landcover, water_types)
    data[water_mask] = np.nan  # Set water areas to NaN (masked)
    out_of_bounds = np.isnan(elevation)
    data[out_of_bounds] = np.nan  # Set out-of-bounds areas to NaN (masked)

    return data

def normalize(data:np.ndarray) -> np.ndarray:
    """
    Normalizes the values in a .npy file to the range [0, 1] using MinMaxScaler.
    Saves the normalized array to the specified output path.

    Args:
        npy_file_path (str): Path to the input .npy file.
        output_file_path (str): Path to save the normalized .npy file.
    """

    # Reshape for MinMaxScaler
    reshaped_data = data.reshape(-1, 1)

    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(reshaped_data)

    # Reshape back to original shape
    normalized_data = normalized_data.reshape(data.shape)

    # print(f"Data normalized: min={np.nanmin(normalized_data)}, max={np.nanmax(normalized_data)}")
    # sns.displot(normalized_data.flatten()[::100], bins=50)
    # plt.show()
    return normalized_data

def label_encode(data: np.ndarray) -> np.ndarray:
    """
    Label encodes the values in a .npy file and saves the encoded array to the specified output path.

    Args:
        npy_file_path (str): Path to the input .npy file.
        output_file_path (str): Path to save the label encoded .npy file.
    """

    flat = data.flatten()
    nan_mask = np.isnan(flat)

    le = LabelEncoder()
    non_nan_vals = flat[~nan_mask]

    # Fit encoder on existing non-NaN values
    le.fit(non_nan_vals)
    encoded_flat = np.empty(flat.shape, dtype=float)
    encoded_flat[nan_mask] = np.nan
    encoded_flat[~nan_mask] = le.transform(non_nan_vals)

    encoded = encoded_flat.reshape(data.shape)
    return encoded


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Preprocess .npy files by normalizing and transforming them.")
    argparser.add_argument('-v','--variables', action='store_true', default=False, help="Process all variables.")
    argparser.add_argument('-l','--labels', action='store_true', default=False, help="Process all hazard inventories.")

    args = argparser.parse_args()
    if not args.variables and not args.labels:
        print("No action specified. Use -v to process variables or -l to process hazard inventories.")
        exit(0)
    

    # List of all variable names to process
    encode_vars = ['landcover', 'HWSD', 'GLIM'] # Categorical variables to label encode
    log_vars = ['elevation', 'slope'] # Variables to log transform
    norm_vars = ['soil_moisture_root', 'soil_moisture_surface', 'NDVI', 
                 'wind_direction_daily', 'wind_speed_daily', 'temperature_daily', 
                 'precipitation_daily', 'fire_weather', 'pga', 'accuflux', 'coastlines',
                   'rivers', 'GEM', 'aspect', 'curvature', 'landcover', 'HWSD', 'GLIM', 
                   'elevation', 'slope'] # Variable to normalize (All variables)
    hazard_vars = ['flood', 'wildfire', 'landslide'] # Hazard variables to clean
 

    elevation = np.load(raw_paths['elevation'])
    landcover = np.load(raw_paths['landcover'])

    if args.variables:
    # Process each variable
        for var in var_paths.keys():
            print(f"Processing {var}...")
            try:
                data = np.load(raw_paths[var])
            except FileNotFoundError:
                print(f"File for {var} not found at {raw_paths[var]}. Skipping...")
                continue
            

            # Apply transformations
            if var in encode_vars:
                print(f"Label encoding {var}...")
                data = label_encode(data)
            elif var in log_vars:
                print(f"Log transforming {var}...")
                data = np.log1p(data)  # log(1 + x) to handle zero values 
            
            # Normalize all variables
            print(f"Normalizing {var}...")
            data = normalize(data)
            
            # Clean map for specific variables
            data = clean_map(data, landcover, elevation)
            
            # Save processed data
            np.save(var_paths[var], data)
            print(f"Saved processed {var} to {var_paths[var]}")

    if args.labels:
        # Process each hazard inventory
        for var in hazard_vars:
            print(f"Cleaning hazard map {var}...")
            try:
                data = np.load(label_paths[var]).astype(np.float32)
            except FileNotFoundError:
                print(f"File for {var} not found at {label_paths[var]}. Skipping...")
                continue

            data = clean_map(data, landcover, elevation)
            np.save(label_paths[var], data)
            print(f"Saved cleaned hazard map {var} to {label_paths[var]}")
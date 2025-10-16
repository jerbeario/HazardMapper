""" HazardMapper - Dataset Module 
========================
This module defines a custom dataset for loading hazard-specific features and labels as patches.
The paths to the variables and labels are defined in dictionaries, and the dataset can be used with PyTorch's DataLoader for training models.

"""

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler 
from HazardMapper import ROOT


##################################################################################
# Configure paths                                                                #
#                                                                                #     
# temp_path = f"/Users/jeremypalmerio/Repos/MasterThesis/Input/Europe" # Temporary files will be stored here       #
temp_path = f"{ROOT}/Input/Europe" # Temporary files will be stored here       #
input_path = f"{ROOT}/Input/Europe" # User needs to provide the input data here #
output_path = f"{ROOT}/Output/Europe" # Output data will be saved here          #
#                                                                                #
##################################################################################


# Raw unprocessed rasterized file paths
# These need to be provided by the user and must be consistent in shape and resolution
raw_paths = {
    # Continuous CIs
    "soil_moisture_root" : f"{temp_path}/npy_arrays/masked_soil_moisture_root_Europe.npy",
    "soil_moisture_surface" : f"{temp_path}/npy_arrays/masked_soil_moisture_surface_Europe.npy",
    "NDVI" : f"{temp_path}/npy_arrays/masked_NDVI_Europe_flat.npy",
    "wind_direction_daily" : f"{temp_path}/npy_arrays/masked_wind_direction_daily_Europe.npy",
    "wind_speed_daily" : f"{temp_path}/npy_arrays/masked_wind_speed_daily_Europe.npy",
    "temperature_daily" : f"{temp_path}/npy_arrays/masked_temperature_daily_Europe.npy",
    "precipitation_daily" : f"{temp_path}/npy_arrays/masked_precipitation_daily_Europe.npy",
    "fire_weather" : f"{temp_path}/npy_arrays/masked_fire_weather_Europe.npy",
    "pga" : f"{temp_path}/npy_arrays/masked_pga_Europe.npy",
    "accuflux" : f"{temp_path}/npy_arrays/masked_accuflux_Europe.npy",
    "coastlines" : f"{temp_path}/npy_arrays/masked_coastlines_Europe.npy",
    "rivers" : f"{temp_path}/npy_arrays/masked_rivers_Europe.npy",
    "slope" : f"{temp_path}/npy_arrays/masked_slope_Europe.npy",
    "strahler" : f"{temp_path}/npy_arrays/masked_strahler_Europe.npy",
    "GEM" : f"{temp_path}/npy_arrays/masked_GEM_Europe.npy",
    "aspect" : f"{temp_path}/npy_arrays/masked_aspect_Europe.npy",
    "elevation" : f"{temp_path}/npy_arrays/masked_elevation_Europe.npy",
    "curvature" : f"{temp_path}/npy_arrays/masked_curvature_Europe.npy",
    "test": f"{temp_path}/test_var.npy",

    # Categorical CIs
    "HWSD" : f"{temp_path}/npy_arrays/masked_HWSD_Europe.npy",
    "GLIM" : f"{temp_path}/npy_arrays/masked_GLIM_Europe.npy",
    "landcover" : f"{temp_path}/npy_arrays/masked_landcover_Europe_flat.npy",
   }

# Preprocessed rasterized file paths
var_paths = {

    # Continuous CIs
    "soil_moisture_root" : f"{input_path}/npy_arrays/masked_soil_moisture_root_Europe_preprocessed.npy",
    "soil_moisture_surface" : f"{input_path}/npy_arrays/masked_soil_moisture_surface_Europe_preprocessed.npy",
    "NDVI" : f"{input_path}/npy_arrays/masked_NDVI_Europe_flat_preprocessed.npy",
    "wind_direction_daily" : f"{input_path}/npy_arrays/masked_wind_direction_daily_Europe_preprocessed.npy",
    "wind_speed_daily" : f"{input_path}/npy_arrays/masked_wind_speed_daily_Europe_preprocessed.npy",
    "temperature_daily" : f"{input_path}/npy_arrays/masked_temperature_daily_Europe_preprocessed.npy",
    "precipitation_daily" : f"{input_path}/npy_arrays/masked_precipitation_daily_Europe_preprocessed.npy",
    "fire_weather" : f"{input_path}/npy_arrays/masked_fire_weather_Europe_preprocessed.npy",
    "pga" : f"{input_path}/npy_arrays/masked_pga_Europe_preprocessed.npy",
    "accuflux" : f"{input_path}/npy_arrays/masked_accuflux_Europe_preprocessed.npy",
    "coastlines" : f"{input_path}/npy_arrays/masked_coastlines_Europe_preprocessed.npy",
    "rivers" : f"{input_path}/npy_arrays/masked_rivers_Europe_preprocessed.npy",
    "slope" : f"{input_path}/npy_arrays/masked_slope_Europe_preprocessed.npy",
    "strahler" : f"{input_path}/npy_arrays/masked_strahler_Europe_preprocessed.npy",
    "GEM" : f"{input_path}/npy_arrays/masked_GEM_Europe_preprocessed.npy",
    "aspect" : f"{input_path}/npy_arrays/masked_aspect_Europe_preprocessed.npy",
    "elevation" : f"{input_path}/npy_arrays/masked_elevation_Europe_preprocessed.npy",
    "curvature" : f"{input_path}/npy_arrays/masked_curvature_Europe_preprocessed.npy",

    # Categorical CIs
    "HWSD" : f"{input_path}/npy_arrays/masked_HWSD_Europe_preprocessed.npy",
    "GLIM" : f"{input_path}/npy_arrays/masked_GLIM_Europe_preprocessed.npy",
    "landcover" : f"{input_path}/npy_arrays/masked_landcover_Europe_flat_preprocessed.npy",
}


# Hazard Maps
hazard_map_paths = {
    # Statistical
    "drought" : f"{input_path}/npy_arrays/masked_drought_Europe.npy",
    "heatwave" : f"{input_path}/npy_arrays/masked_heatwave_Europe.npy",
    "extreme_wind" : f"{input_path}/npy_arrays/masked_extreme_wind_Europe.npy",
    "volcano" : f"{input_path}/npy_arrays/masked_volcano_Europe.npy",
    "earthquake" : f"{input_path}/npy_arrays/masked_earthquake_Europe.npy",
    
    # Deep Learning
    "wildfire" : f"{output_path}/wildfire/hazard_map/SimpleCNN_wildfire_hazard_map.npy",
    "flood" : f"{output_path}/flood/hazard_map/SimpleCNN_flood_hazard_map.npy",
    "landslide" : f"{output_path}/landslide/hazard_map/SimpleCNN_landslide_hazard_map.npy",
}

# Raw Hazard Inventories
label_paths = {
    "wildfire" : f"{input_path}/npy_arrays/masked_wildfire_Europe.npy",
    "flood" : f"{input_path}/npy_arrays/masked_flood_Europe.npy",
    "landslide" : f"{input_path}/npy_arrays/masked_landslide_Europe.npy",
    "multi_hazard" : f"{input_path}/npy_arrays/masked_multi_hazard_Europe.npy",
}

# Hazard Partition Maps
partition_paths = {
    "sub_countries" : f"{input_path}/partition_map/sub_countries_rasterized.npy",
    "wildfire" : f"{input_path}/partition_map/wildfire_partition.npy",
    "flood" : f"{input_path}/partition_map/flood_partition.npy",
    "landslide" : f"{input_path}/partition_map/landslide_partition.npy",
}

# Preprocessed rasterized file paths - downscaled
var_paths_downscaled = {

    # Continuous CIs
    "soil_moisture_root" : f"{input_path}/npy_arrays/masked_soil_moisture_root_Europe_preprocessed_downscaled.npy",
    "soil_moisture_surface" : f"{input_path}/npy_arrays/masked_soil_moisture_surface_Europe_preprocessed_downscaled.npy",
    "NDVI" : f"{input_path}/npy_arrays/masked_NDVI_Europe_flat_preprocessed_downscaled.npy",
    "wind_direction_daily" : f"{input_path}/npy_arrays/masked_wind_direction_daily_Europe_preprocessed_downscaled.npy",
    "wind_speed_daily" : f"{input_path}/npy_arrays/masked_wind_speed_daily_Europe_preprocessed_downscaled.npy",
    "temperature_daily" : f"{input_path}/npy_arrays/masked_temperature_daily_Europe_preprocessed_downscaled.npy",
    "precipitation_daily" : f"{input_path}/npy_arrays/masked_precipitation_daily_Europe_preprocessed_downscaled.npy",
    "fire_weather" : f"{input_path}/npy_arrays/masked_fire_weather_Europe_preprocessed_downscaled.npy",
    "pga" : f"{input_path}/npy_arrays/masked_pga_Europe_preprocessed_downscaled.npy",
    "accuflux" : f"{input_path}/npy_arrays/masked_accuflux_Europe_preprocessed_downscaled.npy",
    "coastlines" : f"{input_path}/npy_arrays/masked_coastlines_Europe_preprocessed_downscaled.npy",
    "rivers" : f"{input_path}/npy_arrays/masked_rivers_Europe_preprocessed_downscaled.npy",
    "slope" : f"{input_path}/npy_arrays/masked_slope_Europe_preprocessed_downscaled.npy",
    "strahler" : f"{input_path}/npy_arrays/masked_strahler_Europe_preprocessed_downscaled.npy",
    "GEM" : f"{input_path}/npy_arrays/masked_GEM_Europe_preprocessed_downscaled.npy",
    "aspect" : f"{input_path}/npy_arrays/masked_aspect_Europe_preprocessed_downscaled.npy",
    "elevation" : f"{input_path}/npy_arrays/masked_elevation_Europe_preprocessed_downscaled.npy",
    "curvature" : f"{input_path}/npy_arrays/masked_curvature_Europe_preprocessed_downscaled.npy",
    "test": f"{input_path}/test_var.npy",

    # Categorical CIs
    "HWSD" : f"{input_path}/npy_arrays/masked_HWSD_Europe_preprocessed_downscaled.npy",
    "GLIM" : f"{input_path}/npy_arrays/masked_GLIM_Europe_preprocessed_downscaled.npy",
    "landcover" : f"{input_path}/npy_arrays/masked_landcover_Europe_flat_preprocessed_downscaled.npy",
}

# Raw Hazard Inventories - downscaled
label_paths_downscaled = {
    "test": f"{input_path}/test_hazard_downscaled.npy",
    "wildfire" : f"{input_path}/npy_arrays/masked_wildfire_Europe_downscaled.npy",
    "flood" : f"{input_path}/npy_arrays/masked_flood_Europe_downscaled.npy",
    "landslide" : f"{input_path}/npy_arrays/masked_landslide_Europe_downscaled.npy",
    "multi_hazard" : f"{input_path}/npy_arrays/masked_multi_hazard_Europe_downscaled.npy",
}

# Hazard Partition Maps - downscaled
partition_paths_downscaled = {
    "sub_countries" : f"{input_path}/partition_map/sub_countries_rasterized_downscaled.npy",
    "wildfire" : f"{input_path}/partition_map/wildfire_partition_downscaled.npy",
    "flood" : f"{input_path}/partition_map/flood_partition_downscaled.npy",
    "landslide" : f"{input_path}/partition_map/landslide_partition_downscaled.npy",
}

class HazardDataset(Dataset):
    def __init__(self, hazard, variables, patch_size=5, downscale=False):
        """
        Custom Dataset for loading hazard-specific features and labels as patches.

        Parameters:
        - hazard (str): The hazard type (e.g., "wildfire").
        - patch_size (int): The size of the patch (n x n) around the center cell.
        """
        if downscale:
            used_label_paths = label_paths_downscaled
            used_var_paths = var_paths_downscaled
        else:
            used_label_paths = label_paths
            used_var_paths = var_paths

        self.patch_size = patch_size
        self.num_vars = len(variables)
     
        # Check if the hazard is valid
        if hazard not in used_label_paths.keys():
            raise ValueError(f"Hazard '{hazard}' is not defined in the dataset.")
        
        # Check if the variables are valid
        for variable in variables:
            if variable not in used_var_paths.keys():
                raise ValueError(f"Variable '{variable}' is not defined in the dataset.")
    
        # get features and labels for the hazard
        feature_paths = [used_var_paths[variable] for variable in variables]
        label_path = used_label_paths[hazard]

        # Load features (stacked along the first axis for channels)
        self.features = np.stack([np.load(path) for path in feature_paths], axis=0)
        self.features = np.nan_to_num(self.features, nan=0.0)  # Handle NaN values

        # Load labels
        self.labels = np.load(label_path)
        if hazard != "multi_hazard":
            self.labels = (self.labels > 0).astype(int)  # Binarize labels except for multi_hazard

        # Ensure the spatial dimensions match between features and labels
        assert self.features.shape[1:] == self.labels.shape, "Mismatch between features and labels!"

        # Padding to handle edge cases for patches
        self.pad_size = patch_size // 2
        self.features = np.pad(
            self.features,
            pad_width=((0, 0), (self.pad_size, self.pad_size), (self.pad_size, self.pad_size)),  #  padding for 3D array
            mode='constant',
            constant_values=0
        )
       
    def __len__(self):
        """
        Returns the number of samples in the dataset (total number of cells).
        """
        return self.labels.size

    def __getitem__(self, idx):
        """
        Returns a single sample (patch and label) at the given index.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - patch (torch.Tensor): The n x n patch of features.
        - label (torch.Tensor): The label for the center cell of the patch.
        """

        # Convert 1D index to 2D spatial index
        h, w = self.labels.shape
        row, col = divmod(idx, w)

        # Extract the patch centered at (row, col)
        patch = self.features[:, row:row + self.patch_size, col:col + self.patch_size]

        # Get the label for the center cell
        label = self.labels[row, col]

        # Convert to PyTorch tensors
        patch = torch.tensor(patch, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
         
        # Reshape patch to (num_vars, patch_size, patch_size)
        patch = patch.view(self.num_vars, self.patch_size, self.patch_size)
        
        return patch, label

# Custom balanced batch sampler
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, neg_ratio=1, replacement=False):
        """
        Args:
            labels (array-like): 1D array of labels (0 or 1).
            batch_size (int): Total size of each batch.
            neg_ratio (int or float): Number of negative samples per positive (default 1:1).
            replacement (bool): Whether to sample with replacement.
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.replacement = replacement
        
        self.pos_indices = np.where(self.labels == 1)[0]
        self.neg_indices = np.where(self.labels == 0)[0]
        
        # Compute number of positive and negative samples per batch
        # Total batch size = pos_per_batch + neg_per_batch
        self.pos_per_batch = int(self.batch_size / (1 + self.neg_ratio))
        self.neg_per_batch = self.batch_size - self.pos_per_batch
        
        # Check if calculated numbers make sense
        if self.pos_per_batch == 0:
            raise ValueError("Batch size too small for the given neg_ratio.")

    def __iter__(self):
        pos_indices = np.random.choice(
            self.pos_indices, size=len(self.pos_indices), replace=self.replacement
        )
        neg_indices = np.random.choice(
            self.neg_indices, size=len(self.neg_indices), replace=self.replacement
        )
        
        n_pos_batches = len(pos_indices) // self.pos_per_batch
        n_neg_batches = len(neg_indices) // self.neg_per_batch
        n_batches = min(n_pos_batches, n_neg_batches)
        
        for i in range(n_batches):
            pos_start = i * self.pos_per_batch
            neg_start = i * self.neg_per_batch
            
            pos_end = pos_start + self.pos_per_batch
            neg_end = neg_start + self.neg_per_batch
            
            pos_batch = pos_indices[pos_start:pos_end]
            neg_batch = neg_indices[neg_start:neg_end]
            
            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        n_pos_batches = len(self.pos_indices) // self.pos_per_batch
        n_neg_batches = len(self.neg_indices) // self.neg_per_batch
        return min(n_pos_batches, n_neg_batches)

# Helper functions to convert between 1D and 2D indices
def index2d_to_1d(idx):
    """
    Convert a 2D index (or array of indices) to 1D index.

    Parameters:
      idx: A tuple/list of two ints (row, col) or a numpy array of shape (n, 2).
      shape: Tuple of (n_rows, n_cols).

    Returns:
      A single integer if idx is a pair, or a numpy array of shape (n,) if idx is an array.
    """
    shape = (16560, 25560)
    
    arr = np.atleast_2d(idx)
    one_d = arr[:, 0] * shape[1] + arr[:, 1]
    return one_d[0] if one_d.size == 1 else one_d

def index1d_to_2d(idx):
    """
    Convert a 1D index (or an array of indices) to a 2D index for an array with the given shape.
    
    Parameters:
        idx: An integer or an array-like of integers representing indices in the flattened array.

    
    Returns:
        A tuple (row, col) if a single index is provided, or
        a numpy array of shape (n, 2) for multiple indices.
    """
    shape = (16560, 25560)
    
    # Ensure idx is a numpy array
    idx_arr = np.atleast_1d(idx)
    
    rows = idx_arr // shape[1]
    cols = idx_arr % shape[1]
    
    result = np.column_stack((rows, cols))
    
    # If only a single index was provided, return as tuple
    if result.shape[0] == 1:
        return tuple(result[0])
    return result
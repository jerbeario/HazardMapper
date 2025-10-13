""" HazardMapper - Utilities module.
==========================
This module provides utility functions for handling raster data, plotting numpy arrays on maps,
and normalizing numpy arrays. It handles some of the feature preprocessing such as normalization.

"""
import os

import rasterio

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


import cartopy.crs as ccrs
import cartopy.feature as cfeature

from PIL import Image

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from HazardMapper.dataset import var_paths, raw_paths, label_paths, partition_paths


def downscale_map(path):
    """
    Downscale a raster map by a given factor and save it as a .npy file.

    Parameters:
    - path: str, path to the input raster map file.

    """
    map = np.load(path)
    downscaled_map = map[::10, ::10]
    print(f"Original shape: {map.shape}, Downscaled shape: {downscaled_map.shape}")
    base, ext = os.path.splitext(path)
    downscaled_path = f"{base}_downscaled.npy"
    np.save(downscaled_path, downscaled_map)
    print(f"Downscaled map saved to: {downscaled_path}")


def tif_to_npy(tif_file, npy_file):
    """
    Convert a .tif file to a .npy array.

    Parameters:
    - tif_file: str, path to the input .tif file.
    - npy_file: str, path to save the output .npy file.
    """
    # Open the .tif file
    with rasterio.open(tif_file) as src:
        # read how many bands are in the raster
        num_bands = src.count
        print(f"Number of bands in the raster: {num_bands}")
        # Read the data (assuming single-band raster)
        data = src.read(1)  # Read the first band
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")

    # Save the data as a .npy file
    np.save(npy_file, data)
    print(f"Saved .npy file to: {npy_file}")



def plot_npy_arrays(npy_file, name, type, title = "", debug_nans=False, log=False, water=False, downsample_factor=10, save_path=None, cmap='viridis', labels=None, extent=None):
    """
    Plots the data from npy files on a map with the correct coordinates.

    Parameters:
    npy_file: str or np.ndarray
        Path to the .npy file or a numpy array.
    name: str
        Name of the data being plotted (e.g., 'Susceptibility', 'Hazard').
    type: str       
        Type of data being plotted ('continuous', 'partition', 'bins').
    title: str, optional    
        Title of the plot. If empty, defaults to 'European {name} Map'.     
    debug_nans: bool, optional
        If True, sets all non-NaN values to 0 and NaN values to 1 for debugging purposes.
    log: bool, optional
        If True, applies a logarithmic transformation to the data.
    downsample_factor: int, optional
        Factor by which to downsample the data. Default is 10 (10x downsampling).
    save_path: str, optional
        Path to save the plot. If None, the plot will not be saved.

    Returns:
    None
    """
    print(f"Plotting {name}...")

    # Define the extent for the map (longitude and latitude bounds)
    # This extent corresponds to the geographical bounds of Europe
    if extent is None:
        # Default extent for Europe
        # Longitude: -25 to 45, Latitude: 27 to 73
        # This can be adjusted based on the specific area of interest
        extent = (-25.0001389, 45.9998611,  27.0001389, 73.0001389)

    # Create a subplot with PlateCarree projection
    fig, axs = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    if isinstance(npy_file, str):
        npy_data = np.load(npy_file)
        print("File loaded")
    elif isinstance(npy_file, np.ndarray):
        npy_data = npy_file
        print("Array loaded")

    if downsample_factor > 1:
        npy_data = npy_data[::downsample_factor, ::downsample_factor]
        print(f"Downsampled data shape: {npy_data.shape}")
    
    if log:
        npy_data = np.log1p(npy_data)

    if debug_nans:
        # set everything to 0 except for NaNs
        npy_data[~np.isnan(npy_data)] = 0
        npy_data[np.isnan(npy_data)] = 1


    # Plot the data on the subplot grid
    im = axs.imshow(npy_data, cmap=cmap, extent=extent)
    print("image created")
    if water:
        # Load the landcover data
        landcover = np.load(raw_paths['landcover'])
        if downsample_factor > 1:
            landcover = landcover[::downsample_factor, ::downsample_factor]

        # Create a mask for landcover where the value is 210 (water)
        water_mask = np.where(landcover == 210, 1, 0)
        custom_blue_cmap = ListedColormap(['none', '#A6CAE0'])
        axs.imshow(water_mask, cmap=custom_blue_cmap, extent=extent, transform=ccrs.PlateCarree(), origin='upper')

    # Set title for each subplot
    axs.set_title(title, fontsize=16)

    # Set longitude tick labels
    axs.set_xticks(np.arange(extent[0], extent[1] + 1, 5), crs=ccrs.PlateCarree())
    axs.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}°'))

    # Set latitude tick labels
    axs.set_yticks(np.arange(extent[2], extent[3] + 1, 5), crs=ccrs.PlateCarree())
    axs.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}°'))

    axs.set_xlabel('Longitude')
    axs.set_ylabel('Latitude')

    # Add coastlines and country borders
    axs.add_feature(cfeature.COASTLINE, linewidth=0.1, edgecolor='black')
    axs.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.1)
    axs.add_feature(cfeature.LAND, facecolor='#FFEB3B', alpha=0.1)
    axs.add_feature(cfeature.OCEAN, facecolor='#A6CAE0')

    # Adjust layout for better spacing
    plt.tight_layout()


    # Add a colorbar for all subplots

    if type == 'continuous':
        cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1)
        cbar.set_label(f'{name}', fontsize=16)
        if log:
            cbar.set_label(f'log {name}', fontsize=16)
        cbar.ax.tick_params(labelsize=12) 
    elif type == 'partition':
        labels = ['Ignored', 'Train', 'Validation', 'Test', 'Border']
        cmap = plt.get_cmap(cmap, 5)  # Use a colormap with 5 distinct colors
        colors = [cmap(i) for i in range(cmap.N)]
        patches =[mpatches.Patch(color=color,label=labels[i]) for i, color in enumerate(colors)]
        fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.53, 0.15),
        fancybox=True, ncol=5)
    elif type == 'bins':
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        cmap = plt.get_cmap(cmap, 5)
        colors = [cmap(i) for i in range(cmap.N)]
        patches = [mpatches.Patch(color=color, label=labels[i]) for i, color in enumerate(colors)]
        fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.53, 0.15),
                    fancybox=True, ncol=5)
    elif type == 'categorical':
        if labels is None:
            categories = np.unique(npy_data)
            cmap = plt.get_cmap(cmap, len(categories))  # Use a colormap with the number of unique categories
            colors = [cmap(i) for i in range(cmap.N)]
            patches = [mpatches.Patch(color=color, label=str(category)) for category, color in zip(categories, colors)]
            
        else: 
            cmap = plt.get_cmap(cmap, len(labels))  # Use a colormap with the number of labels
            colors = [cmap(i) for i in range(cmap.N)]
            patches = [mpatches.Patch(color=color, label=labels[i]) for i, color in enumerate(colors)]
            fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.53, 0.15),
                        fancybox=True, ncol=len(labels))
    
    else:
        raise ValueError("Invalid type. Choose from 'continuous', 'partition', 'bins', or 'categorical'.")

    # Save the plot
    if save_path is not None:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory

    # plt.show()  # Show the plot


def plot_maps_grid(
    data_list,
    names,
    cmap='viridis',
    downsample=10,
    figsize=(12,8),
    nrows=4,
    ncols=4,
    save_path=None
):
    """
    4×4 PlateCarree grid with:
      - only bottom‐center xlabel (“Longitude”)
      - only left‐center ylabel (“Latitude”)
      - optional super title
      - no per‐axes colorbars
      - minimal whitespace via manual row shifts
    """
    if len(data_list) > nrows*ncols or len(names) > nrows*ncols:
        raise ValueError("data_list and titles must each have max {nrows} x {ncols} elements")

    cmaps = [cmap]*nrows*ncols if isinstance(cmap, str) else cmap
    if len(cmaps) > nrows*ncols:
        raise ValueError("cmap must be a string or list of  max {nrows} x {ncols} strings")

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree()},
        gridspec_kw={'hspace': 0.1, 'wspace': 0.1}
    )

    fig.suptitle("European Conditioning Factors",fontweight='bold' ,  fontsize=18, x= 0.5,  y=0.975)
    extent = (-25.0001389, 45.9998611,  27.0001389, 73.0001389)
    axs = axs.flatten()
    lon_ticks = np.arange(extent[0], extent[1]+1, 15)
    lat_ticks = np.arange(extent[2], extent[3]+1, 10)

    for idx, (datum, ax, name, cm) in enumerate(zip(data_list, axs, names, cmaps)):
        # Load the numpy array
        arr = np.load(datum) if isinstance(datum, str) else datum
        # arr = np.zeros((1625, 2556))


        if name == "Elevation":
            arr[arr > 4000] = 4000  # cap elevation at 4000m
        if name == "Precipitation Daily":
            arr[arr > 30] = 30  # cap precipitation at 30mm/day
        if name == "Rivers":
            arr[arr > 5000] = 4000  # cap river distance at 4000m
            # arr = np.log1p(arr)  # apply log transformation to river distance
        if name == "Curvature":
            arr = np.log1p(arr)  # apply log transformation to curvature
        if name == "Accuflux":
            arr = np.log1p(arr)  # apply log transformation to accuflux
        if name == "Slope":
            arr = np.log1p(arr)  # apply log transformation to slope
        
        if name in ["Drought", "Earthquake", "Extreme Wind", "Heatwave", "Volcano",]:
            # Normalize the hazard maps to [0, 1] based on a threshold
            arr = normalize_label(arr, threshold=0.99)
            
        if downsample > 1:
            arr = arr[::downsample, ::downsample]


        scaler = MinMaxScaler()
        arr = scaler.fit_transform(arr.reshape(-1, 1)).reshape(arr.shape)

        ax.imshow(
            arr,
            cmap=cm,
            extent=extent,
            transform=ccrs.PlateCarree(),
            origin='upper'
        )

        if name in ["Drought", "Earthquake", "Extreme Wind", "Heatwave", "Volcano", "Wildfire", "Flood", "Landslide"]:
            landcover = np.load("Input/Europe/npy_arrays/masked_landcover_Europe_flat.npy")
            if downsample > 1:
                landcover = landcover[::downsample, ::downsample]

            # Create a mask for landcover where the value is 210
            water_mask = np.where(landcover == 210, 1, 0)
            custom_blue_cmap = ListedColormap(['none', '#A6CAE0'])
            ax.imshow(water_mask, cmap=custom_blue_cmap, extent=extent, transform=ccrs.PlateCarree(), origin='upper')


        ax.set_title(name, fontweight='normal',fontsize=11, pad=2)

        # map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.2)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.2)
        ax.add_feature(cfeature.LAND, facecolor='#FFEB3B', alpha=0.1)
        ax.add_feature(cfeature.OCEAN, facecolor='#A6CAE0')

        row, col = divmod(idx, ncols)

        # bottom row: ticks only
        if row == nrows - 1:
            ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}°'))
        else:
            ax.set_xticks([])

        # left col: ticks only
        if col == 0:
            ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}°'))
        else:
            ax.set_yticks([])

    for idx, ax in enumerate(axs):
        pos = ax.get_position()
        
        print(f"before: {pos.y0}")
        print(row)

        
        ax.set_position([pos.x0, 
                         pos.y0, 
                         pos.width, 
                         pos.height
                         ])

        print(f"After ; {ax.get_position().y0}")
        

    # shared axis labels
    fig.text(0.5,   0.05, 'Longitude', ha='center', va='bottom', fontsize=14)
    fig.text(0.075,  0.53, 'Latitude',  ha='left',   va='center',
             rotation='vertical', fontsize=14)
    
    # shared colorbar
    cbar_ax = fig.add_axes([0.9, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(axs[0].images[0], cax=cbar_ax, orientation='vertical')
    cbar.set_label('Normalized Value', fontsize=14)
    cbar.ax.tick_params(labelsize=10)



    if save_path:
        plt.savefig(save_path, dpi=500)




def save_full_resolution_plot(npy_file, npy_name):
    """
    Work in progress, not yet functional.
    Saves the full resolution plot of a numpy array as a JPEG image.
    """


    if isinstance(npy_file, str):
        npy_data = np.load(npy_file)
        print("File loaded")
    elif isinstance(npy_file, np.ndarray):
        npy_data = npy_file
        print("Array loaded")
    

    image = Image.fromarray(npy_data)
    image.save(f"{npy_name}_full_resolution.jpeg")

def normalize_label(hazard_map, threshold=0.99):
    """
    Normalize a hazard map to the range [0, 1] based on a specified percentile threshold.
    
    Args:
        hazard_map (np.ndarray): The input hazard map to normalize.
        threshold (float): The percentile threshold for normalization (default is 0.99).
        
    Returns:
        np.ndarray: The normalized hazard map.
    """

    # Ensure the hazard map is a numpy array
    if isinstance(hazard_map, str):
        hazard_map = np.load(hazard_map)
    elif not isinstance(hazard_map, np.ndarray):
        raise ValueError("Input hazard_map must be a numpy array.")

    # Get the threshold value at the specified percentile
    threshold_value = np.nanpercentile(hazard_map[hazard_map > 0], threshold * 100)
    print(f"{threshold*100}th percentile threshold: {threshold_value}")

    # Cutoff the hazard map at the threshold
    hazard_map[hazard_map > threshold_value] = threshold_value

    # Normalize the hazard map to [0, 1]
    scaler = MinMaxScaler()
    normalized_map = scaler.fit_transform(hazard_map.reshape(-1, 1)).reshape(hazard_map.shape)

    return normalized_map

if __name__ == "__main__":
    for var in var_paths.values():
        try:
            downscale_map(var)
        except Exception as e:
            print(f"Error processing {var}: {e}")    

    for label in label_paths.values():
        try:
            downscale_map(label)
        except Exception as e:
            print(f"Error processing {label}: {e}")
    
    for partition in partition_paths.values():
        try:
            downscale_map(partition)
        except Exception as e:
            print(f"Error processing {partition}: {e}")
    

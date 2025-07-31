import numpy as np
import scipy.ndimage as ndimage
import open3d as o3d
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os

def apply_filter(point_cloud, filter_type='median', filter_size=3, iterations=1):
    """
    Apply a single filter (median, gaussian, average) to a point cloud.
    
    Parameters:
    - point_cloud: np.ndarray of shape (N, 3), where N is the number of points.
    - filter_type: str, type of filter ('median', 'gaussian', 'average').
    - filter_size: int, size of the filter kernel (3, 5, 7, 9, ...).
    - iterations: int, number of times the filter is applied.
    
    Returns:
    - Filtered point cloud as np.ndarray.
    """
    if filter_size % 2 == 0:
        raise ValueError("Filter size must be an odd number (3, 5, 7, 9, ...)")

    filtered_points = np.copy(point_cloud).astype(np.float64)
    z_vals = filtered_points[:, 2]
    
    # Apply the chosen filter
    for _ in range(iterations):
        if filter_type == 'median':
            z_vals = ndimage.median_filter(z_vals, size=filter_size).astype(np.float64)
        elif filter_type == 'gaussian':
            z_vals = ndimage.gaussian_filter(z_vals, sigma=filter_size / 3.0).astype(np.float64)
        elif filter_type == 'average':
            z_vals = ndimage.uniform_filter(z_vals, size=filter_size).astype(np.float64)
        else:
            raise ValueError("Invalid filter type. Choose from 'median', 'gaussian', 'average'.")
    
    filtered_points[:, 2] = z_vals
    return filtered_points

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder Containing Point Cloud Files")
    return folder_path if folder_path else None

def load_point_clouds_from_folder(folder_path):
    """
    Load all CSV point clouds from the selected folder.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    point_clouds = {}
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            point_clouds[file] = pd.read_csv(file_path, header=0).values.astype(np.float64)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    return point_clouds

def select_save_folder():
    """
    Ask user to select a folder to save the filtered point clouds.
    """
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder to Save Filtered Files")
    return folder_path if folder_path else None

def save_point_clouds(point_clouds, save_folder):
    """
    Save all processed point clouds to the selected save folder.
    """
    for file_name, point_cloud in point_clouds.items():
        save_path = os.path.join(save_folder, f"filtered_{file_name}")
        pd.DataFrame(point_cloud, columns=['x', 'y', 'z']).to_csv(save_path, index=False, header=True)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    # Load files from a selected folder
    input_folder = select_folder()
    if not input_folder:
        print("No input folder selected.")
        exit()
    
    point_clouds = load_point_clouds_from_folder(input_folder)
    if not point_clouds:
        print("No valid point cloud files found.")
        exit()
    
    # Apply multiple filters sequentially
    filtered_clouds = {}
    for name, pc in point_clouds.items():
        Filter = apply_filter(pc, filter_type='average', filter_size=21, iterations=3)
        # Filter = apply_filter(Filter, filter_type='gaussian', filter_size=21, iterations=3)
        # Filter = apply_filter(Filter, filter_type='gaussian', filter_size=17, iterations=3)
        filtered_clouds[name] = Filter

    # Select save folder
    save_folder = select_save_folder()
    if not save_folder:
        print("No save folder selected.")
        exit()

    # Save processed point clouds
    save_point_clouds(filtered_clouds, save_folder)
    print("All files processed and saved successfully.")

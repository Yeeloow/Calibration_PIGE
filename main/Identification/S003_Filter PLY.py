import numpy as np
import scipy.ndimage as ndimage
import open3d as o3d
import tkinter as tk
from tkinter import filedialog
import os


def apply_filter(point_cloud: np.ndarray, filter_type: str = 'median', filter_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Apply a filter to the Z-values of a point cloud.
    Parameters:
    - point_cloud: np.ndarray of shape (N, 3)
    - filter_type: 'median', 'gaussian', 'average'
    - filter_size: odd int kernel size
    - iterations: number of times to apply the filter
    Returns:
    - filtered point cloud (np.ndarray)
    """
    if filter_size % 2 == 0:
        raise ValueError("Filter size must be an odd number")

    pts = point_cloud.copy().astype(np.float64)
    z = pts[:, 2]
    for _ in range(iterations):
        if filter_type == 'median':
            z = ndimage.median_filter(z, size=filter_size)
        elif filter_type == 'gaussian':
            z = ndimage.gaussian_filter(z, sigma=filter_size / 3.0)
        elif filter_type == 'average':
            z = ndimage.uniform_filter(z, size=filter_size)
        else:
            raise ValueError("Invalid filter type: choose 'median', 'gaussian', or 'average'")
    pts[:, 2] = z
    return pts


def select_files(title: str, filetypes: list) -> list:
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    return list(files)


def load_point_cloud(path: str) -> np.ndarray:
    """
    Load a single PLY or CSV point cloud file.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.ply':
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points, dtype=np.float64)
    elif ext == '.csv':
        import pandas as pd
        df = pd.read_csv(path, header=0)
        return df[['x', 'y', 'z']].values.astype(np.float64)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_point_cloud_as_ply(pts: np.ndarray, out_path: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(out_path, pcd)


if __name__ == '__main__':
    # Select multiple PLY (or CSV) files
    file_paths = select_files("Select PLY/CSV Files to Filter", [("Point Clouds", "*.ply *.csv")])
    if not file_paths:
        print("No files selected.")
        exit(1)

    # Configure filter parameters
    filter_type = 'gaussian'
    filter_size = 2
    iterations = 1

    # Process each file
    results = []  # list of tuples (original_path, filtered_points)
    for path in file_paths:
        try:
            pts = load_point_cloud(path)
            filtered_pts = apply_filter(pts, filter_type, filter_size, iterations)
            results.append((path, filtered_pts))
        except Exception as e:
            print(f"Failed to process {path}: {e}")

    # Select folder to save
    save_folder = filedialog.askdirectory(title="Select Folder to Save Filtered PLY Files")
    if not save_folder:
        print("No save folder selected.")
        exit(1)

    # Save results
    for original_path, pts in results:
        name = os.path.splitext(os.path.basename(original_path))[0]
        out_path = os.path.join(save_folder, f"filtered_{name}.ply")
        save_point_cloud_as_ply(pts, out_path)
        print(f"Saved: {out_path}")

    print("Processing complete.")
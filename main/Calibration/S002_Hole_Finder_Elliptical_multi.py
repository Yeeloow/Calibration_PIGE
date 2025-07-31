import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.cluster import DBSCAN
from skimage.measure import EllipseModel
import tkinter as tk
from tkinter import filedialog
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_point_cloud(file_path):
    df = pd.read_csv(file_path)
    points = df[['x', 'y', 'z']].to_numpy()
    return points

def detect_ellipses(points, min_axis_length=0.68):
    lower_percentile = 93
    upper_percentile = 99
    z_lower = np.percentile(points[:, 2], lower_percentile)
    z_upper = np.percentile(points[:, 2], upper_percentile)
    edge_points = points[(points[:, 2] >= z_lower) & (points[:, 2] <= z_upper)]

    clustering = DBSCAN(eps=0.03, min_samples=3).fit(edge_points[:, :2])
    labels = clustering.labels_
    unique_labels = set(labels)
    ellipses = []

    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = edge_points[labels == label, :2]
        cluster_points = np.unique(cluster_points, axis=0)

        if len(cluster_points) < 200:
            continue

        try:
            model = EllipseModel()
            if not model.estimate(cluster_points):
                continue
            xc, yc, a, b, theta = model.params
            if min(a, b) < min_axis_length:
                continue
            ellipses.append((xc, yc, a, b, theta))
        except:
            continue

    return ellipses

def compute_normal_vector(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca.components_[-1]

def angle_in_xz_plane(vector):
    x, _, z = vector
    vector_xz = np.array([x, z])
    z_axis_xz = np.array([0, 1])
    norm = np.linalg.norm(vector_xz)
    return np.degrees(np.arccos(np.clip(np.dot(vector_xz, z_axis_xz) / norm, -1.0, 1.0))) if norm > 0 else np.nan

def angle_in_yz_plane(vector):
    _, y, z = vector
    vector_yz = np.array([y, z])
    z_axis_yz = np.array([0, 1])
    norm = np.linalg.norm(vector_yz)
    return np.degrees(np.arccos(np.clip(np.dot(vector_yz, z_axis_yz) / norm, -1.0, 1.0))) if norm > 0 else np.nan

def load_point_clouds_from_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    point_clouds = {}
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            point_clouds[file] = load_point_cloud(file_path)
        except:
            continue
    return point_clouds

def select_folder(title="Select Folder"):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title=title)

def process_all_files():
    input_folder = select_folder("Select Folder Containing CSV Files")
    if not input_folder:
        print("No input folder selected. Exiting.")
        return

    point_clouds = load_point_clouds_from_folder(input_folder)
    if not point_clouds:
        print("No valid point cloud files found.")
        return

    save_folder = select_folder("Select Folder to Save Results")
    if not save_folder:
        print("No save folder selected. Exiting.")
        return

    all_ellipse_data = []

    for file_name, points in point_clouds.items():
        ellipses = detect_ellipses(points)
        print(f"{file_name}: Detected {len(ellipses)} ellipses.")

        for (xc, yc, a, b, theta) in ellipses:
            mask = np.linalg.norm(points[:, :2] - np.array([xc, yc]), axis=1) <= max(a, b)
            ellipse_points = points[mask]
            centroid_z = np.mean(ellipse_points[:, 2])
            if len(ellipse_points) >= 3:
                normal_vector = compute_normal_vector(ellipse_points)
            else:
                normal_vector = (np.nan, np.nan, np.nan)
            angle_b = angle_in_xz_plane(normal_vector)
            angle_a = angle_in_yz_plane(normal_vector)
            all_ellipse_data.append([
                file_name,
                round(yc, 3), round(xc, 3), round(2 * max(a, b), 3),
                round(centroid_z, 3),
                round(normal_vector[0], 3), round(normal_vector[1], 3), round(normal_vector[2], 3),
                round(angle_b, 3), round(angle_a, 3)
            ])

    save_path = os.path.join(save_folder, "all_ellipses.csv")
    df = pd.DataFrame(all_ellipse_data, columns=[
        "File_Name", "Center_X", "Center_Y", "Diameter",
        "Center_Z", "Normal_X", "Normal_Y", "Normal_Z", "Degree_B", "Degree_A"
    ])
    df.to_csv(save_path, index=False)
    print(f"All detected ellipses saved to {save_path}")

if __name__ == "__main__":
    process_all_files()
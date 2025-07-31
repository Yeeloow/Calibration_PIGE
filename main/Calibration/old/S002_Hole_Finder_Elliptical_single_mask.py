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
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select the CSV file", filetypes=[["CSV files", "*.csv"]])
    df = pd.read_csv(file_path)
    points = df[['x', 'y', 'z']].to_numpy()
    return points

def detect_ellipses(points, min_axis_length=0.5):
    lower_percentile = 86
    upper_percentile = 98
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

        if len(cluster_points) < 20:
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

    return ellipses, edge_points

def plot_point_cloud_3d(points):
    fig = plt.figure(figsize=(8, 6))
    sc = plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='cividis', s=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Point Cloud with Height-based Color')
    cbar = plt.colorbar(sc)
    cbar.set_label('Z value (height)')
    plt.axis('equal')
    plt.show()

def plot_detected_ellipses(points, edge_points, ellipses):
    plt.figure(figsize=(8,8))
    plt.scatter(points[:,0], points[:,1], s=1, c='blue', label="All Points")
    plt.scatter(edge_points[:,0], edge_points[:,1], s=1, c='orange', label="Filtered Z Points")
    ax = plt.gca()
    for (xc, yc, a, b, theta) in ellipses:
        ellipse_patch = patches.Ellipse((xc, yc), width=2*a, height=2*b,
                                        angle=np.degrees(theta), edgecolor='red', facecolor='none', lw=5, label="Detected Ellipse")
        mask_patch = patches.Ellipse((xc, yc), width=2*a, height=2*b,
                                     angle=np.degrees(theta), edgecolor='gray', facecolor='gray', lw=1, alpha=0.9, label="Masked Region")
        ax.add_patch(ellipse_patch)
        ax.add_patch(mask_patch)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Detected Ellipses on XY Plane")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.axis("equal")
    plt.show()

def main():
    points = load_point_cloud(None)
    ellipses, edge_points = detect_ellipses(points, min_axis_length=0.68)

    print(f"Detected {len(ellipses)} ellipses.")
    for i, (xc, yc, a, b, theta) in enumerate(ellipses):
        print(f"Ellipse {i+1}: Center=({xc:.3f}, {yc:.3f}), a={a:.3f}, b={b:.3f}, angle={np.degrees(theta):.2f}°")

    plot_point_cloud_3d(points)
    plot_detected_ellipses(points, edge_points, ellipses)

if __name__ == "__main__":
    main()
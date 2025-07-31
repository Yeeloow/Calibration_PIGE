import os
import numpy as np
import pandas as pd
import open3d as o3d
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

# ─── Parameters ─────────────────────────────────────────────────────────
PLANE_DIST_THRESH        = 0.005    # RANSAC plane distance threshold (m)
MIN_HOLE_DIST            = 0.01    # Min depth below plane for hole points (m)
MAX_HOLE_DIST            = 0.5    # Max depth below plane for hole points (m)
HOLE_CLUSTER_EPS         = 0.01    # DBSCAN eps for hole-point filtering (m)
HOLE_CLUSTER_MIN_SAMPLES = 10      # DBSCAN min_samples for hole-point filtering
MIN_HOLE_CLUSTER_SIZE    = 20000   # Cluster size threshold (below = noise)

# ─── Functions ───────────────────────────────────────────────────────────
def load_point_cloud(csv_path):
    df = pd.read_csv(csv_path)
    pts = df[['x','y','z']].to_numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd, pts


def detect_hole_clusters(pts):
    # Plane segmentation
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=PLANE_DIST_THRESH,
        ransac_n=3,
        num_iterations=1000
    )
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    norm_n = np.linalg.norm(normal)
    signed_dist = (pts @ normal + d) / norm_n

    # Filter points below plane within depth range
    mask_below = signed_dist <= 0
    depth_below = -signed_dist[mask_below]
    hole_mask = (depth_below >= MIN_HOLE_DIST) & (depth_below <= MAX_HOLE_DIST)
    hole_pts = pts[mask_below][hole_mask]

    # DBSCAN filtering
    db = DBSCAN(eps=HOLE_CLUSTER_EPS, min_samples=HOLE_CLUSTER_MIN_SAMPLES)
    labels = db.fit_predict(hole_pts[:, :2])
    valid_labels = [lab for lab in set(labels) if lab != -1 and np.sum(labels == lab) >= MIN_HOLE_CLUSTER_SIZE]
    if valid_labels:
        final_pts = np.vstack([hole_pts[labels == lab] for lab in valid_labels])
    else:
        final_pts = np.empty((0,3))
    return final_pts


def save_hole_csv(hole_pts, output_dir, base_name):
    """
    Save detected hole points to CSV with columns x,y,z.
    """
    if hole_pts.size == 0:
        print(f"[{base_name}] No hole points to save.")
        return
    df = pd.DataFrame(hole_pts, columns=['x', 'y', 'z'])
    filename = os.path.join(output_dir, f"{base_name}_holes.csv")
    df.to_csv(filename, index=False)
    print(f"[{base_name}] Saved hole points to CSV: {filename}")


def plot_holes_2d(hole_pts, base_name=None):
    plt.figure()
    plt.scatter(hole_pts[:,0], hole_pts[:,1], s=1)
    plt.xlabel('X'); plt.ylabel('Y')
    title = 'Hole Points (XY View)' if base_name is None else f'Hole Points: {base_name}'
    plt.title(title)
    plt.axis('equal'); plt.show()


def plot_holes_3d(hole_pts, base_name=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(hole_pts[:,0], hole_pts[:,1], hole_pts[:,2], s=1)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    title = 'Detected Hole Point Cloud' if base_name is None else f'Hole Cloud: {base_name}'
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# ─── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Select input and output directories
    root = tk.Tk(); root.withdraw()
    input_dir = filedialog.askdirectory(title="Select input directory with CSV files")
    if not input_dir:
        print("No input directory selected. Exiting.")
        exit(0)
    output_dir = filedialog.askdirectory(title="Select output directory for hole CSVs")
    if not output_dir:
        print("No output directory selected. Exiting.")
        exit(0)

    # Process each CSV in input folder
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.csv'):
            continue
        csv_path = os.path.join(input_dir, fname)
        base_name = os.path.splitext(fname)[0]
        print(f"Processing {fname}...")

        # Load points
        try:
            _, pts = load_point_cloud(csv_path)
        except Exception as e:
            print(f"[{base_name}] Error loading file: {e}")
            continue

        # Detect holes
        hole_pts = detect_hole_clusters(pts)
        print(f"[{base_name}] Detected {hole_pts.shape[0]} hole points.")

        # Save CSV
        save_hole_csv(hole_pts, output_dir, base_name)

        # Optional plotting per file
        # plot_holes_2d(hole_pts, base_name)
        # plot_holes_3d(hole_pts, base_name)

    print("Batch processing complete.")

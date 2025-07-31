import os
import numpy as np
import pandas as pd
import open3d as o3d
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import warnings

# ─── Parameters ─────────────────────────────────────────────────────────
PLANE_DIST_THRESH        = 0.03      # RANSAC plane distance threshold (m)
MIN_HOLE_DIST            = 0.04      # Min depth below plane for hole points (m)
MAX_HOLE_DIST            = 0.07      # Max depth below plane for hole points (m)

# Boundary detection thresholds
MIN_BOUNDARY_DIST_THRESH = 0.005      # Min XY distance to hole for boundary detection (m)
MAX_BOUNDARY_DIST_THRESH = 0.05      # Max XY distance to hole for boundary detection (m)

# Hole‑point filtering parameters
HOLE_CLUSTER_EPS         = 0.01      # DBSCAN eps for hole‑point filtering (m)
HOLE_CLUSTER_MIN_SAMPLES = 5        # DBSCAN min_samples for hole‑point filtering
MIN_HOLE_CLUSTER_SIZE    = 20000     # Cluster size threshold (below = noise)


warnings.filterwarnings("ignore", category=UserWarning)


def process_file(file_path, output_dir):
    """
    Process a single CSV or PLY point cloud file to detect hole boundary points and save to CSV.
    """
    ext = os.path.splitext(file_path)[1].lower()
    # Load points
    try:
        if ext == '.csv':
            df = pd.read_csv(file_path)
            pts = df[['x','y','z']].to_numpy()
        elif ext == '.ply':
            pcd = o3d.io.read_point_cloud(file_path)
            pts = np.asarray(pcd.points)
        else:
            print(f"Unsupported file type: {file_path}, skipping.")
            return
    except Exception as e:
        print(f"[{os.path.basename(file_path)}] Failed to load: {e}")
        return

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    # Plane segmentation
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=PLANE_DIST_THRESH,
        ransac_n=3,
        num_iterations=1000
    )
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    norm_n = np.linalg.norm(normal)

    plane_pts = pts[inliers]
    plane_xy  = plane_pts[:, :2]

    # Remove points above plane
    signed_dist = (pts @ normal + d) / norm_n
    mask_below = signed_dist <= 0
    pts_below  = pts[mask_below]
    depth_below = -signed_dist[mask_below]

    # Hole point extraction
    hole_mask = (depth_below >= MIN_HOLE_DIST) & (depth_below <= MAX_HOLE_DIST)
    hole_pts = pts_below[hole_mask]
    hole_xy  = hole_pts[:, :2]

    # DBSCAN filtering of hole clusters
    if hole_xy.size:
        db = DBSCAN(eps=HOLE_CLUSTER_EPS, min_samples=HOLE_CLUSTER_MIN_SAMPLES)
        labels = db.fit_predict(hole_xy)
        good_labels = [lbl for lbl in set(labels)
                       if lbl != -1 and np.sum(labels == lbl) >= MIN_HOLE_CLUSTER_SIZE]
        if good_labels:
            mask_good = np.isin(labels, good_labels)
            filtered_hole_pts = hole_pts[mask_good]
            filtered_hole_xy  = filtered_hole_pts[:, :2]
        else:
            filtered_hole_pts = np.empty((0,3))
            filtered_hole_xy  = np.empty((0,2))
    else:
        filtered_hole_pts = np.empty((0,3))
        filtered_hole_xy  = np.empty((0,2))

    # Boundary detection
    if filtered_hole_xy.size and plane_xy.size:
        nbrs = NearestNeighbors(n_neighbors=1).fit(filtered_hole_xy)
        dists, _ = nbrs.kneighbors(plane_xy)
        d_flat = dists.flatten()
        boundary_mask = (d_flat >= MIN_BOUNDARY_DIST_THRESH) & (d_flat <= MAX_BOUNDARY_DIST_THRESH)
        boundary_pts  = plane_pts[boundary_mask]
    else:
        boundary_pts = np.empty((0,3))

    # Save boundary points to CSV
    base = os.path.splitext(os.path.basename(file_path))[0]
    out_csv = os.path.join(output_dir, f"{base}_boundary.csv")
    pd.DataFrame(boundary_pts, columns=['x','y','z']).to_csv(out_csv, index=False)
    print(f"[{base}] Saved {len(boundary_pts)} boundary points → {out_csv}")


if __name__ == "__main__":
    # 1) 다중 파일 선택
    root = tk.Tk(); root.withdraw()
    files = filedialog.askopenfilenames(
        title="Select PLY/CSV files to process",
        filetypes=[("PointCloud files", "*.ply *.csv"), ("All files","*.*")]
    )
    if not files:
        print("No files selected. Exiting.")
        exit(0)

    # 2) 출력 폴더 선택
    output_dir = filedialog.askdirectory(title="Select output directory for CSVs")
    if not output_dir:
        print("No output directory selected. Exiting.")
        exit(0)

    # 3) 처리 루프
    for file_path in files:
        print(f"Processing {os.path.basename(file_path)} …")
        process_file(file_path, output_dir)

    print("Batch processing complete.")

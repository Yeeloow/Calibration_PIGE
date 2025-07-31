import os
import numpy as np
import pandas as pd
import open3d as o3d
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN   # DBSCAN for filtering

# ─── Parameters ─────────────────────────────────────────────────────────
PLANE_DIST_THRESH        = 0.01      # RANSAC plane distance threshold (m)
MIN_HOLE_DIST            = 0.01      # Min depth below plane for hole points (m)
MAX_HOLE_DIST            = 0.25      # Max depth below plane for hole points (m)

# Boundary detection thresholds
MIN_BOUNDARY_DIST_THRESH = 0.005      # Min XY distance to hole for boundary detection (m)
MAX_BOUNDARY_DIST_THRESH = 0.05      # Max XY distance to hole for boundary detection (m)

# Hole‑point filtering parameters
HOLE_CLUSTER_EPS         = 0.01      # DBSCAN eps for hole‑point filtering (m)
HOLE_CLUSTER_MIN_SAMPLES = 5        # DBSCAN min_samples for hole‑point filtering
MIN_HOLE_CLUSTER_SIZE    = 20000     # Cluster size threshold (below = noise)

if __name__ == '__main__':
    # ─── 1. File selection ─────────────────────────────────────────────────
    root = tk.Tk(); root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select point cloud file",
        filetypes=[("PointCloud files","*.csv;*.ply"), ("All files","*.*")]
    )
    if not file_path:
        print("No file selected.")
        exit()

    # ─── 2. Load points (CSV or PLY) ───────────────────────────────────────
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(file_path)
        points = df[['x','y','z']].to_numpy()
    elif ext == '.ply':
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # ─── 3. Detect plane via RANSAC ─────────────────────────────────────────
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=PLANE_DIST_THRESH,
        ransac_n=3,
        num_iterations=1000
    )
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    norm_n = np.linalg.norm(normal)
    plane_pts = points[inliers]
    plane_xy = plane_pts[:, :2]

    # ─── 4. Remove points above the plane ───────────────────────────────────
    signed_dist_all = (points @ normal + d) / norm_n
    mask_below = signed_dist_all <= 0
    pts_below = points[mask_below]

    # ─── 5. Extract hole points by depth range ─────────────────────────────
    depth_below = -signed_dist_all[mask_below]
    hole_mask = (depth_below >= MIN_HOLE_DIST) & (depth_below <= MAX_HOLE_DIST)
    hole_pts = pts_below[hole_mask]
    hole_xy = hole_pts[:, :2]

    # ─── 5.1 Filter sparse hole points via DBSCAN ──────────────────────────
    db_hole = DBSCAN(eps=HOLE_CLUSTER_EPS, min_samples=HOLE_CLUSTER_MIN_SAMPLES)
    labels_hole = db_hole.fit_predict(hole_xy)
    good_labels = [lbl for lbl in set(labels_hole)
                   if lbl != -1 and np.sum(labels_hole == lbl) >= MIN_HOLE_CLUSTER_SIZE]
    mask_good = np.isin(labels_hole, good_labels)
    hole_pts = hole_pts[mask_good]
    hole_xy = hole_pts[:, :2]

    # ─── 6. Detect boundary: plane points near hole region ────────────────
    nbrs_hole = NearestNeighbors(n_neighbors=1).fit(hole_xy)
    dists_to_hole, _ = nbrs_hole.kneighbors(plane_xy)
    d_flat = dists_to_hole.flatten()
    boundary_mask = (d_flat >= MIN_BOUNDARY_DIST_THRESH) & (d_flat <= MAX_BOUNDARY_DIST_THRESH)
    boundary_pts = plane_pts[boundary_mask]

    # ─── 7. Create Open3D clouds ───────────────────────────────────────────
    plane_cloud = o3d.geometry.PointCloud()
    plane_cloud.points = o3d.utility.Vector3dVector(plane_pts)
    plane_cloud.paint_uniform_color([1,1,0])  # yellow
    hole_cloud = o3d.geometry.PointCloud()
    hole_cloud.points = o3d.utility.Vector3dVector(hole_pts)
    hole_cloud.paint_uniform_color([1,0,0])   # red
    boundary_cloud = o3d.geometry.PointCloud()
    boundary_cloud.points = o3d.utility.Vector3dVector(boundary_pts)
    boundary_cloud.paint_uniform_color([0,0,1])  # blue

    # ─── 8. Save boundary points ───────────────────────────────────────────
    save_path = filedialog.asksaveasfilename(
        title="Save boundary points as",
        defaultextension=".csv",
        initialfile=os.path.splitext(os.path.basename(file_path))[0] + '_boundary.csv',
        filetypes=[("CSV files","*.csv")]
    )
    if save_path:
        pd.DataFrame(boundary_pts, columns=['x','y','z']).to_csv(save_path, index=False)
        print(f"Saved boundary points to {save_path}")
    else:
        print("Save cancelled.")
        exit()

    # ─── 9. Visualize in Open3D ──────────────────────────────────────────
    o3d.visualization.draw_geometries(
        [plane_cloud, hole_cloud, boundary_cloud],
        window_name='Plane (Y), Hole (R), Boundary (B)', width=800, height=600
    )

    # ─── 10. 2D Plot ───────────────────────────────────────────────────────
    plt.figure(figsize=(8,6))
    plt.scatter(plane_xy[:,0], plane_xy[:,1], c='yellow', s=2, label='Plane')
    plt.scatter(hole_xy[:,0],  hole_xy[:,1],  c='red',    s=2, label='Hole (filtered)')
    plt.scatter(boundary_pts[:,0], boundary_pts[:,1], c='blue',  s=4, label='Boundary')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.title('Plane, Hole, Boundary')
    plt.legend(); plt.axis('equal'); plt.show()

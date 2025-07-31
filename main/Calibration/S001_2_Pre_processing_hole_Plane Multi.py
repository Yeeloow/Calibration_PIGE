import os
import pandas as pd
import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import filedialog

# ─── Parameters ─────────────────────────────────────────────────────────
RANSAC_DIST_THRESH = 0.01  # meters
MIN_DEPTH = 0.018          # meters (min depth below plane)
MAX_DEPTH = 0.15           # meters (max depth below plane)

if __name__ == '__main__':
    # Initialize GUI
    root = tk.Tk(); root.withdraw()

    # Select input directory (folder) containing CSV point clouds
    input_dir = filedialog.askdirectory(
        title="Select input directory with point cloud CSVs"
    )
    if not input_dir:
        print("No input directory selected. Exiting.")
        exit(0)

    # Select output directory (folder) for CSV hole outputs
    output_dir = filedialog.askdirectory(
        title="Select output directory for hole CSVs"
    )
    if not output_dir:
        print("No output directory selected. Exiting.")
        exit(0)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each CSV in the input folder
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.csv'):
            continue
        base = os.path.splitext(fname)[0]
        csv_path = os.path.join(input_dir, fname)
        print(f"Processing '{fname}'...")

        # Load CSV data
        try:
            df = pd.read_csv(csv_path)
            pts = df[['x','y','z']].to_numpy()
        except Exception as e:
            print(f"[{base}] Failed to load CSV: {e}")
            continue

        # Build point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # Plane segmentation
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=RANSAC_DIST_THRESH,
            ransac_n=3,
            num_iterations=1000
        )
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        norm_n = np.linalg.norm(normal)

        # Compute signed distances and filter by depth range
        signed_dist = (pts @ normal + d) / norm_n
        depth = -signed_dist
        mask = (depth >= MIN_DEPTH) & (depth <= MAX_DEPTH)
        hole_pts = pts[mask]

        if hole_pts.size == 0:
            print(f"[{base}] No hole points detected.")
            continue

        # Save hole points to CSV
        out_file = os.path.join(output_dir, f"{base}_hole.csv")
        pd.DataFrame(hole_pts, columns=['x','y','z']).to_csv(out_file, index=False)
        print(f"[{base}] Saved hole points to: {out_file}")

    print("Batch processing complete.")

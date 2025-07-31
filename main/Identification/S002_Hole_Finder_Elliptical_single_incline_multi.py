import os
import numpy as np
import open3d as o3d
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import DBSCAN
from skimage.measure import EllipseModel
import warnings

# ─── Parameters ─────────────────────────────────────────────────────────
CLUSTER_EPS = 0.5            # DBSCAN eps for boundary clustering
CLUSTER_MIN_SAMPLES = 10     # DBSCAN min_samples for boundary clustering
MIN_ELLIPSE_POINTS = 500     # Minimum points to attempt ellipse fitting
INNER_ELLIPSE_RATIO  = 0.80  # z_mean calculation inner region (80%)
SCALE                = np.sqrt(INNER_ELLIPSE_RATIO)  # scaling factor for ellipse axes

warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == '__main__':
    # Initialize GUI
    root = tk.Tk(); root.withdraw()

    # Select directories
    pc_dir = filedialog.askdirectory(title="Select directory with hole point CSVs (.csv/.ply)")
    if not pc_dir:
        print("No hole point directory selected. Exiting.")
        exit(0)

    boundary_dir = filedialog.askdirectory(title="Select directory with boundary CSVs (.csv/.ply)")
    if not boundary_dir:
        print("No boundary directory selected. Exiting.")
        exit(0)

    output_dir = filedialog.askdirectory(title="Select output directory for ellipse results")
    if not output_dir:
        print("No output directory selected. Exiting.")
        exit(0)

    # Build map of base names to hole files (strip _hole suffix)
    pc_files = {}
    for f in os.listdir(pc_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext not in ['.csv', '.ply']:
            continue
        base_name = os.path.splitext(f)[0]
        key = base_name.lower().removesuffix('_hole')
        pc_files[key] = os.path.join(pc_dir, f)

    # Build map of base names to boundary files (strip _boundary suffix)
    boundary_files = {}
    for f in os.listdir(boundary_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext not in ['.csv', '.ply']:
            continue
        base_name = os.path.splitext(f)[0]
        key = base_name.lower().removesuffix('_boundary')
        boundary_files.setdefault(key, []).append(os.path.join(boundary_dir, f))

    # Match base keys
    common = set(pc_files.keys()) & set(boundary_files.keys())
    if not common:
        print("No matching basenames found between hole and boundary directories.")
        exit(0)

    results = []
    for key in sorted(common):
        pc_path = pc_files[key]
        # Load hole points
        try:
            if pc_path.lower().endswith('.csv'):
                df_pc = pd.read_csv(pc_path)
                coords_pc = df_pc[['x','y']].values
                z_pc      = df_pc['z'].values
            else:
                # .ply
                pcd = o3d.io.read_point_cloud(pc_path)
                pts = np.asarray(pcd.points)
                coords_pc = pts[:, :2]
                z_pc      = pts[:, 2]
        except Exception as e:
            print(f"[{key}] Error loading hole file: {e}")
            continue

        for b_path in boundary_files[key]:
            fname = os.path.basename(b_path)
            print(f"Processing boundary file: {fname} (key: '{key}')")
            try:
                if b_path.lower().endswith('.csv'):
                    df_b = pd.read_csv(b_path)
                    coords = df_b[['x','y']].values
                    z_vals = df_b['z'].values
                else:
                    pcd_b = o3d.io.read_point_cloud(b_path)
                    pts_b = np.asarray(pcd_b.points)
                    coords = pts_b[:, :2]
                    z_vals = pts_b[:, 2]
            except Exception as e:
                print(f"[{key}] Error loading boundary file: {e}")
                continue

            labels = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES).fit_predict(coords)
            for lbl in sorted(set(labels) - {-1}):
                mask = labels == lbl
                pts_cluster = coords[mask]
                if len(pts_cluster) < MIN_ELLIPSE_POINTS:
                    print(f"[{key}] Cluster {lbl}: too few points ({len(pts_cluster)}), skipping.")
                    continue
                em = EllipseModel()
                if not em.estimate(pts_cluster):
                    continue
                xc, yc, a, b, theta = em.params

                # Project hole points into ellipse frame
                dx = coords_pc[:,0] - xc
                dy = coords_pc[:,1] - yc
                c_th, s_th = np.cos(theta), np.sin(theta)
                u = dx * c_th + dy * s_th
                v = -dx * s_th + dy * c_th

                mask_in = (u**2) / (a * SCALE)**2 + (v**2) / (b * SCALE)**2 <= 1
                inner_z = z_pc[mask_in]
                if inner_z.size == 0:
                    z_mean = float(z_vals[mask].mean())
                else:
                    z_mean = float(inner_z.mean())

                results.append({
                    'Base': key,
                    'Cluster': lbl,
                    'Center_X': yc,
                    'Center_Y': xc,
                    'Center_Z': z_mean
                })

    # Save combined results
    if results:
        df_res = pd.DataFrame(results)
        out_csv = os.path.join(output_dir, 'ellipse_results.csv')
        df_res.to_csv(out_csv, index=False)
        print(f"Saved {len(results)} ellipse entries to {out_csv}")
    else:
        print("No ellipse entries detected. Nothing saved.")

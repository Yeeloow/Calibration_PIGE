import os
import numpy as np
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

warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == '__main__':
    # Initialize GUI
    root = tk.Tk(); root.withdraw()

    # Select directories
    pc_dir = filedialog.askdirectory(title="Select directory with full point cloud CSVs")
    if not pc_dir:
        print("No point cloud directory selected. Exiting.")
        exit(0)

    boundary_dir = filedialog.askdirectory(title="Select directory with boundary CSVs")
    if not boundary_dir:
        print("No boundary directory selected. Exiting.")
        exit(0)

    output_dir = filedialog.askdirectory(title="Select output directory for ellipse results")
    if not output_dir:
        print("No output directory selected. Exiting.")
        exit(0)

    # Build maps of basenames to paths
    pc_files = {os.path.splitext(f)[0]: os.path.join(pc_dir, f)
                for f in os.listdir(pc_dir) if f.lower().endswith('.csv')}
    boundary_files = {}
    for f in os.listdir(boundary_dir):
        if not f.lower().endswith('.csv'): continue
        name = os.path.splitext(f)[0]
        base = name[:-9] if name.endswith('_boundary') else name
        boundary_files.setdefault(base, []).append(os.path.join(boundary_dir, f))

    # Match basenames
    common = set(pc_files) & set(boundary_files)
    if not common:
        print("No matching CSV basenames found between point cloud and boundary directories.")
        exit(0)

    # Collect all ellipse results
    results = []

    for base in sorted(common):
        for b_path in boundary_files[base]:
            fname = os.path.basename(b_path)
            print(f"Processing boundary file: {fname} (base: '{base}')")
            try:
                df_b = pd.read_csv(b_path)
            except Exception as e:
                print(f"[{base}] Error loading boundary file: {e}")
                continue
            if not {'x','y','z'}.issubset(df_b.columns):
                print(f"[{base}] Missing required columns. Skipping.")
                continue
            coords = df_b[['x','y']].values
            z_vals = df_b['z'].values

            # Cluster boundary points
            labels = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES).fit_predict(coords)
            for lbl in sorted(set(labels) - {-1}):
                mask = labels == lbl
                pts = coords[mask]
                if len(pts) < MIN_ELLIPSE_POINTS:
                    print(f"[{base}] Cluster {lbl}: too few points ({len(pts)}), skipping.")
                    continue

                em = EllipseModel()
                if not em.estimate(pts):
                    print(f"[{base}] Cluster {lbl}: ellipse fitting failed.")
                    continue
                xc, yc, a, b, theta = em.params
                z_mean = float(z_vals[mask].mean())

                results.append({
                    'Center_X': yc,
                    'Center_Y': -xc,
                    'Center_Z': z_mean,
                })

    # Save combined results
    if results:
        df_res = pd.DataFrame(results)
        out_csv = os.path.join(output_dir, 'ellipse_results.csv')
        df_res.to_csv(out_csv, index=False)
        print(f"Saved {len(results)} ellipse entries to {out_csv}")
    else:
        print("No ellipse entries detected. Nothing saved.")

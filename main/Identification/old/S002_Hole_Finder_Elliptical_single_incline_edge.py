import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from skimage.measure import EllipseModel

# ─── Parameters ─────────────────────────────────────────────────────────
CLUSTER_EPS = 0.5      # DBSCAN eps (in same units as XY)
CLUSTER_MIN_SAMPLES = 10  # DBSCAN min samples for core point

# ─── 1. Select full point cloud CSV ─────────────────────────────────────
root = tk.Tk(); root.withdraw()
pc_path = filedialog.askopenfilename(
    title="Select full point cloud CSV file",
    filetypes=[("CSV files","*.csv")]
)
if not pc_path:
    print("No file selected.")
    exit()

# ─── 2. Load full point cloud ────────────────────────────────────────────
df_pc = pd.read_csv(pc_path)
if not {'x','y'}.issubset(df_pc.columns):
    raise ValueError("Full point cloud CSV must contain 'x' and 'y' columns")
xy_pc = df_pc[['x','y']].values
print(f"Loaded full point cloud: {len(xy_pc)} points")

# ─── 3. Select boundary CSV ─────────────────────────────────────────────
boundary_path = filedialog.askopenfilename(
    title="Select boundary point CSV file",
    filetypes=[("CSV files","*.csv")],
    initialdir=os.path.dirname(pc_path)
)
if not boundary_path:
    print("No boundary file selected.")
    exit()

# ─── 4. Load boundary points ─────────────────────────────────────────────
df_b = pd.read_csv(boundary_path)
if not {'x','y'}.issubset(df_b.columns):
    raise ValueError("Boundary CSV must contain 'x' and 'y' columns")
boundary_xy = df_b[['x','y']].values
print(f"Loaded {len(boundary_xy)} boundary points")

# ─── 5. DBSCAN clustering ─────────────────────────────────────────────────
db = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES)
labels = db.fit_predict(boundary_xy)
unique_labels = sorted(set(labels) - {-1})
print(f"Identified {len(unique_labels)} clusters (excluding noise)")

# ─── 6. Ellipse fitting per cluster ───────────────────────────────────────
ellipses = []  # list of (cluster_label, xc, yc, a, b, theta)
for lbl in unique_labels:
    pts = boundary_xy[labels == lbl]
    if len(pts) < 500:
        print(f"Cluster {lbl}: too few points, skipping")
        continue
    em = EllipseModel()
    success = em.estimate(pts)
    if not success:
        print(f"Cluster {lbl}: ellipse fitting failed")
        continue
    # Correct parameter order: (xc, yc, a, b, theta)
    xc, yc, a, b, theta = em.params
    ellipses.append((lbl, xc, yc, a, b, theta))
    print(f"Cluster {lbl}: center=({xc:.4f},{yc:.4f}), a={a:.4f}, b={b:.4f}, theta={theta:.4f}")

# ─── 7. Save ellipse parameters ─────────────────────────────────────────
base = os.path.splitext(boundary_path)[0]
ellipse_df = pd.DataFrame(ellipses, columns=['cluster','xc','yc','a','b','theta'])
ellipse_csv = base + '_ellipses.csv'
ellipse_df.to_csv(ellipse_csv, index=False)
print(f"Saved ellipse parameters to {ellipse_csv}")

# ─── 8. Visualization ───────────────────────────────────────────────────
plt.figure(figsize=(8,6))
# plot full point cloud in light gray
plt.scatter(xy_pc[:,0], xy_pc[:,1], s=1, c='lightgray', label='Point Cloud')
# plot boundary clusters
colors = plt.cm.tab10(np.linspace(0,1,len(unique_labels)))
for idx, lbl in enumerate(unique_labels):
    pts = boundary_xy[labels == lbl]
    plt.scatter(pts[:,0], pts[:,1], s=4, color=colors[idx], label=f'Cluster {lbl}')
# plot fitted ellipses
t = np.linspace(0,2*np.pi,200)
for lbl, xc, yc, a, b, theta in ellipses:
    x_ell = xc + a * np.cos(t)*np.cos(theta) - b * np.sin(t)*np.sin(theta)
    y_ell = yc + a * np.cos(t)*np.sin(theta) + b * np.sin(t)*np.cos(theta)
    plt.plot(x_ell, y_ell, linewidth=2)

plt.xlabel('X'); plt.ylabel('Y')
plt.title('Point Cloud with Boundary Clusters & Ellipses')
plt.axis('equal'); plt.legend(); plt.show()

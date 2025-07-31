import os
import random
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from skimage.measure import EllipseModel
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import struct
import zlib


def load_point_cloud_pcd(file_path):
    """
    Load point cloud from a PCD file (ASCII or binary).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext != '.pcd':
        raise ValueError(f"Unsupported extension {ext}. Please select a .pcd file.")
    # Try Open3D first
    try:
        print("Attempting Open3D PCD load...")
        pcd = o3d.io.read_point_cloud(file_path)
        pts = np.asarray(pcd.points)
        if pts.size > 0:
            return pts
        print("Open3D loaded no points, falling back to manual parser...")
    except Exception as e:
        print(f"Open3D read failed: {e}, falling back to manual parser...")

    # Manual PCD parsing
    print("Parsing PCD manually...")
    with open(file_path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('ascii', errors='ignore')
            if not line:
                raise ValueError("Reached EOF before DATA header")
            parts = line.strip().split()
            if len(parts) >= 2:
                header[parts[0].lower()] = parts[1:]
            if parts and parts[0].lower() == 'data':
                data_fmt = parts[1].lower()
                break

        fields = header.get('fields', [])
        sizes = list(map(int, header.get('size', [])))
        types = header.get('type', [])
        counts = list(map(int, header.get('count', [])))
        width = int(header.get('width', [1])[0])
        height = int(header.get('height', [1])[0])
        n_pts = width * height

        try:
            ix = fields.index('x')
            iy = fields.index('y')
            iz = fields.index('z')
        except ValueError:
            raise ValueError("PCD missing x/y/z fields")

        if data_fmt == 'ascii':
            data = f.read().decode('ascii', errors='ignore').split()
            arr = np.array(data, dtype=float).reshape(-1, len(fields))
            return arr[:, [ix, iy, iz]]

        body = f.read()
        if data_fmt == 'binary_compressed':
            comp_size, uncomp_size = struct.unpack('ii', body[:8])
            comp = body[8:8+comp_size]
            body = zlib.decompress(comp)

        dtype_list = []
        offset = 0
        for name, sz, tp, ct in zip(fields, sizes, types, counts):
            if tp=='F' and sz==4:
                np_t = np.float32
            elif tp=='F' and sz==8:
                np_t = np.float64
            elif tp=='U' and sz==1:
                np_t = np.uint8
            elif tp=='U' and sz==2:
                np_t = np.uint16
            elif tp=='U' and sz==4:
                np_t = np.uint32
            elif tp=='I' and sz==1:
                np_t = np.int8
            elif tp=='I' and sz==2:
                np_t = np.int16
            elif tp=='I' and sz==4:
                np_t = np.int32
            else:
                raise ValueError(f"Unsupported type {tp}{sz}")
            if ct == 1:
                dtype_list.append((name, np_t))
            else:
                dtype_list.append((name, np_t, (ct,)))
            offset += sz * ct
        dtype_np = np.dtype(dtype_list)
        data = np.frombuffer(body, dtype_np, count=n_pts)
        return np.vstack([data['x'], data['y'], data['z']]).T


def detect_rim_and_plane(points,
                         min_axis_length=0.5,
                         lower_pct=90, upper_pct=100,
                         ransac_dist=0.01,
                         db_eps=0.03, db_min=3,
                         seed=42):
    """
    RANSAC-based plane detection with fixed seed + rim extraction + ellipse fitting.
    Returns: ellipses, rim_pts, plane_pts
    """
    # Fix random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # 1) Fit plane via RANSAC
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=ransac_dist,
        ransac_n=3,
        num_iterations=1000)
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    norm_n = np.linalg.norm(normal)
    print(f"RANSAC plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    # Extract plane points
    plane_pts = points[inliers]
    print(f"Plane inliers: {len(plane_pts)} points")

    # 2) Compute depth relative to plane
    signed_dist = (points @ normal + d) / norm_n
    depth = -signed_dist

    # 3) Percentile thresholds
    lower = np.percentile(depth, lower_pct)
    upper = np.percentile(depth, upper_pct)
    rim_mask = (depth >= lower) & (depth <= upper)
    rim_pts = points[rim_mask]
    print(f"Rim pts: {len(rim_pts)} between {lower_pct}-{upper_pct}%")

    # 4) Cluster rim in XY and fit ellipses
    XY = rim_pts[:, :2]
    labels = DBSCAN(eps=db_eps, min_samples=db_min).fit_predict(XY)
    ellipses = []
    for lbl in set(labels):
        if lbl < 0:
            continue
        cluster = np.unique(XY[labels == lbl], axis=0)
        if len(cluster) < 20:
            continue
        model = EllipseModel()
        if not model.estimate(cluster):
            continue
        xc, yc, a_len, b_len, theta = model.params
        if min(a_len, b_len) < min_axis_length:
            continue
        ellipses.append((xc, yc, a_len, b_len, theta))

    return ellipses, rim_pts, plane_pts


def plot_point_cloud_3d(points):
    plt.figure(figsize=(8,6))
    sc = plt.scatter(points[:,0], points[:,1], c=points[:,2], cmap='cividis', s=2)
    plt.xlabel('X'); plt.ylabel('Y'); plt.title('Point Cloud')
    plt.colorbar(sc, label='Z'); plt.axis('equal'); plt.show()


def plot_results(points, plane_pts, rim_pts, ellipses):
    plt.figure(figsize=(8,8))
    plt.scatter(points[:,0], points[:,1], s=1, c='lightgray', label='All Points')
    plt.scatter(plane_pts[:,0], plane_pts[:,1], s=2, c='yellow', label='Plane Inliers')
    plt.scatter(rim_pts[:,0], rim_pts[:,1], s=2, c='blue', label='Rim Points')
    ax = plt.gca()
    for xc, yc, a_len, b_len, theta in ellipses:
        ellipse = patches.Ellipse((xc, yc), width=2*a_len, height=2*b_len,
                                  angle=np.degrees(theta), edgecolor='red', facecolor='none', lw=2)
        ax.add_patch(ellipse)
    plt.xlabel('X'); plt.ylabel('Y'); plt.title('Detected Rim Ellipses & RANSAC Plane')
    plt.legend(); plt.axis('equal'); plt.show()


def main():
    root = tk.Tk(); root.withdraw()
    file_path = filedialog.askopenfilename(
        title='Select PCD file',
        filetypes=[('PCD Files', '*.pcd')]
    )
    if not file_path:
        print('No file selected.'); return
    pts = load_point_cloud_pcd(file_path)
    ellipses, rim_pts, plane_pts = detect_rim_and_plane(
        pts,
        min_axis_length=0.6,
        lower_pct=0,
        upper_pct=10,
        ransac_dist=0.01,
        db_eps=0.03,
        db_min=3,
        seed=42
    )
    print(f"Detected {len(ellipses)} ellipses")
    for i,(xc,yc,a_len,b_len,theta) in enumerate(ellipses,1):
        print(f"Ellipse {i}: center=({xc:.3f},{yc:.3f}), a={a_len:.3f}, b={b_len:.3f}")
    plot_point_cloud_3d(pts)
    plot_results(pts, plane_pts, rim_pts, ellipses)

if __name__ == '__main__':
    main()
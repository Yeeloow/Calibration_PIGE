import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os
import struct
import zlib


def load_point_cloud(file_path, sample_rate: float = 1.0):
    """Load point‑cloud data from CSV, PCD, PLY, or XYZ file.

    Parameters
    ----------
    file_path : str
        Path to the point‑cloud file.
    sample_rate : float, optional
        Fraction of points to keep (for large CSV files). Default is 1.0 (no sampling).

    Returns
    -------
    np.ndarray
        (N, 3) array of XYZ coordinates.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        print("Loading CSV file …")
        df = pd.read_csv(file_path)
        if sample_rate < 1.0:
            df = df.sample(frac=sample_rate)
        return df[["x", "y", "z"]].to_numpy()

    elif ext in [".ply", ".xyz"]:
        print(f"Loading {ext} via Open3D …")
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points)

    elif ext == ".pcd":
        # Try Open3D first (handles most PCD variants)
        try:
            print("Attempting Open3D PCD load …")
            pcd = o3d.io.read_point_cloud(file_path)
            pts = np.asarray(pcd.points)
            if len(pts):
                return pts
            print("Open3D loaded no points, falling back …")
        except Exception as e:
            print(f"Open3D read failed: {e}, falling back …")

        # Minimal manual PCD parser (ASCII / binary / binary_compressed)
        print("Parsing PCD manually …")
        with open(file_path, "rb") as f:
            header = {}
            while True:
                line = f.readline().decode("ascii", errors="ignore")
                if not line:
                    raise ValueError("Reached EOF before DATA header")
                parts = line.strip().split()
                if parts:
                    header[parts[0].lower()] = parts[1:]
                if parts and parts[0].lower() == "data":
                    data_format = parts[1].lower()
                    break

            fields = header.get("fields", [])
            sizes = list(map(int, header.get("size", [])))
            types = header.get("type", [])
            counts = list(map(int, header.get("count", [])))
            width = int(header.get("width", [1])[0])
            height = int(header.get("height", [1])[0])
            n_pts = width * height

            idx_x, idx_y, idx_z = (fields.index(k) for k in ("x", "y", "z"))

            if data_format == "ascii":
                vals = f.read().decode("ascii", errors="ignore").strip().split()
                arr = np.array(vals, float).reshape(-1, len(fields))
                return arr[:, [idx_x, idx_y, idx_z]]

            if data_format in ("binary", "binary_compressed"):
                body = f.read()
                if data_format == "binary_compressed":
                    comp_sz, uncomp_sz = struct.unpack("ii", body[:8])
                    body = zlib.decompress(body[8 : 8 + comp_sz])
                # Build dtype for structured array
                dtype_fields = []
                for name, sz, tp, ct in zip(fields, sizes, types, counts):
                    if ct == 1:
                        dtype_fields.append((name, np.dtype(f"{tp.lower()}{sz}")))
                    else:
                        dtype_fields.append((name, np.dtype(f"{tp.lower()}{sz}"), (ct,)))
                data = np.frombuffer(body, dtype=np.dtype(dtype_fields), count=n_pts)
                return np.vstack((data["x"], data["y"], data["z"])).T

            raise ValueError(f"Unsupported PCD DATA format: {data_format}")

    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def plot_point_cloud(points: np.ndarray, title: str = "3D Point Cloud", *, width: int = 800, height: int = 600):
    """Interactive 3‑D point‑cloud preview using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd], window_name=title, width=width, height=height)


def plot_xy_grid(points: np.ndarray, title: str = "XY Grid (2D)"):
    """Scatter the X‑Y projection with grid lines for quick overview."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points[:, 0], points[:, 1], s=1, alpha=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4)
    plt.tight_layout()
    plt.show()


def select_file() -> str:
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select a point cloud file",
        filetypes=[
            ("All Files", "*"),
            ("CSV Files", "*.csv"),
            ("PCD Files", "*.pcd"),
            ("PLY Files", "*.ply"),
            ("XYZ Files", "*.xyz"),
        ],
    )


def main():
    file_path = select_file()
    if not file_path:
        print("No file selected.")
        return

    points = load_point_cloud(file_path)

    # 3‑D view
    plot_point_cloud(points, title=os.path.basename(file_path))

    # 2‑D XY scatter with grid lines
    plot_xy_grid(points, title=f"{os.path.basename(file_path)} – XY Grid")


if __name__ == "__main__":
    main()

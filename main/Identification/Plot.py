import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, simpledialog
import os
import struct
import zlib

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
DEFAULT_RATIO = 100  # % (100 = no down‑sample)

# ────────────────────────────────────────────────────────────────────────────────
# Point‑cloud loader with ratio‑based down‑sampling
# ────────────────────────────────────────────────────────────────────────────────

def load_point_cloud(file_path: str, *, sample_rate: float = 1.0) -> np.ndarray:
    """Return (N,3) float32 array of (x, y, z) coordinates.

    `sample_rate` ∈ (0, 1] keeps that fraction of points.
    """
    ext = os.path.splitext(file_path)[1].lower()

    def _apply_rate(arr: np.ndarray) -> np.ndarray:
        if 0 < sample_rate < 1.0 and arr.shape[0]:
            n_keep = max(1, int(arr.shape[0] * sample_rate))
            idx = np.random.choice(arr.shape[0], n_keep, replace=False)
            arr = arr[idx]
        return arr.astype(np.float32, copy=False)

    # ── CSV ────────────────────────────────────────────────────────────────
    if ext == ".csv":
        df = pd.read_csv(file_path, usecols=["x", "y", "z"], dtype=np.float32)
        if sample_rate < 1.0:
            df = df.sample(frac=sample_rate)
        return df.to_numpy()

    # ── PLY / XYZ via Open3D ───────────────────────────────────────────────
    if ext in (".ply", ".xyz"):
        arr = np.asarray(o3d.io.read_point_cloud(file_path).points)
        return _apply_rate(arr)

    # ── PCD ────────────────────────────────────────────────────────────────
    if ext == ".pcd":
        # First try Open3D (handles most PCDs)
        try:
            arr = np.asarray(o3d.io.read_point_cloud(file_path).points)
            if arr.size:
                return _apply_rate(arr)
        except Exception:
            pass  # fall back to manual parser

        # Manual PCD parser for ASCII / binary / binary_compressed
        with open(file_path, "rb") as f:
            header = {}
            while True:
                line = f.readline().decode("ascii", errors="ignore")
                parts = line.strip().split()
                if parts:
                    header[parts[0].lower()] = parts[1:]
                if parts and parts[0].lower() == "data":
                    data_format = parts[1].lower()
                    break

            fields = header.get("fields", [])
            sizes  = list(map(int, header.get("size", [])))
            types  = header.get("type", [])
            counts = list(map(int, header.get("count", [])))
            n_pts  = int(header.get("width", [1])[0]) * int(header.get("height", [1])[0])
            idx_x, idx_y, idx_z = (fields.index(k) for k in ("x", "y", "z"))

            body = f.read()
            # ASCII --------------------------------------------------------
            if data_format == "ascii":
                arr = np.array(body.decode("ascii", errors="ignore").strip().split(), float)
                arr = arr.reshape(-1, len(fields))[:, [idx_x, idx_y, idx_z]]
                return _apply_rate(arr)

            # Binary (possibly compressed) ---------------------------------
            if data_format == "binary_compressed":
                comp_sz, _ = struct.unpack("ii", body[:8])
                body = zlib.decompress(body[8 : 8 + comp_sz])
            # now raw binary buffer in `body`
            dtype_fields = []
            for name, sz, tp, ct in zip(fields, sizes, types, counts):
                np_fmt = {"F": "f", "I": "i", "U": "u"}[tp] + str(sz)
                dtype_fields.append((name, np.dtype(np_fmt), (ct,) if ct > 1 else ()))
            data = np.frombuffer(body, dtype=np.dtype(dtype_fields), count=n_pts)
            arr = np.vstack((data["x"], data["y"], data["z"])).T
            return _apply_rate(arr)

    raise ValueError(f"Unsupported file extension: {ext}")

# ────────────────────────────────────────────────────────────────────────────────
# 2‑D scatter coloured by height (Z)
# ────────────────────────────────────────────────────────────────────────────────

def plot_xy_coloured(points: np.ndarray, title: str = "XY coloured by Z") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap="viridis", s=2, alpha=0.7, linewidths=0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle="--", linewidth=0.3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.88)
    cb.set_label("Z (height)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# ────────────────────────────────────────────────────────────────────────────────
# Dialog helpers
# ────────────────────────────────────────────────────────────────────────────────

def select_file() -> str:
    root = Tk(); root.withdraw()
    return filedialog.askopenfilename(
        title="Select point‑cloud file",
        filetypes=[("CSV", "*.csv"), ("PCD", "*.pcd"), ("PLY", "*.ply"), ("XYZ", "*.xyz"), ("All", "*"),],
    )

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    file_path = select_file()
    if not file_path:
        print("No file selected – exiting.")
        return

    root = Tk(); root.withdraw()
    ratio = simpledialog.askfloat(
        "Down‑sample ratio (%)",
        f"Enter percentage (1–100). Default {DEFAULT_RATIO} means no down‑sample:",
        minvalue=1.0,
        maxvalue=100.0,
        initialvalue=DEFAULT_RATIO,
        parent=root,
    )
    sample_rate = (ratio or DEFAULT_RATIO) / 100.0

    pts = load_point_cloud(file_path, sample_rate=sample_rate)
    pct_txt = f"{int(sample_rate*100)}%" if sample_rate < 1.0 else "100%"
    plot_xy_coloured(pts, title=f"{os.path.basename(file_path)} – {pct_txt} sample ({pts.shape[0]:,} pts)")


if __name__ == "__main__":
    main()

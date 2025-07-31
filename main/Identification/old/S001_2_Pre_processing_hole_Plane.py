import os
import pandas as pd
import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import filedialog

# ─── 1. 파일 열기 대화상자 ─────────────────────────────────────
root = tk.Tk(); root.withdraw()
pc_path = filedialog.askopenfilename(
    title="포인트 클라우드 파일 선택",
    filetypes=[("PointCloud 파일", "*.csv;*.ply"), ("All files", "*.*")]
)
if not pc_path:
    raise SystemExit("포인트 클라우드 파일이 선택되지 않았습니다.")

# ─── 2. 포인트 클라우드 준비 ───────────────────────────────────
ext = os.path.splitext(pc_path)[1].lower()
if ext == '.csv':
    df = pd.read_csv(pc_path)
    pts = df[["x", "y", "z"]].values
elif ext == '.ply':
    pcd = o3d.io.read_point_cloud(pc_path)
    pts = np.asarray(pcd.points)
else:
    raise ValueError(f"Unsupported file type: {pc_path}")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

# ─── 3. 평면 RANSAC ────────────────────────────────────────────
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=1000
)
[a, b, c, d] = plane_model
norm_abc = np.linalg.norm([a, b, c])

# ─── 4. Signed 거리 추출 & 깊이 필터 ───────────────────────────
pts = np.asarray(pcd.points)
signed_dist = (pts @ np.array([a, b, c]) + d) / norm_abc
depth = -signed_dist
min_d, max_d = 0.025, 0.15  # 예: 15mm ~ 150mm
mask = (depth >= min_d) & (depth <= max_d)
hole_pts = pts[mask]

hole_cloud = o3d.geometry.PointCloud()
hole_cloud.points = o3d.utility.Vector3dVector(hole_pts)
hole_cloud.colors = o3d.utility.Vector3dVector(
    np.tile([1, 0, 0], (hole_pts.shape[0], 1))
)

# ─── 5. 저장 대화상자 & 디렉터리 준비 ─────────────────────────
save_path = filedialog.asksaveasfilename(
    title="홀만 추출한 포인트 클라우드 저장",
    defaultextension=".ply",
    filetypes=[("PCD 파일", "*.pcd"), ("PLY 파일", "*.ply"), ("모든 파일", "*.*")]
)
if not save_path:
    raise SystemExit("저장 경로가 선택되지 않았습니다.")

os.makedirs(os.path.dirname(save_path), exist_ok=True)

# ─── 6. Open3D로 저장 ────────────────────────────────────────
success = o3d.io.write_point_cloud(
    save_path,
    hole_cloud,
    write_ascii=save_path.lower().endswith('.pcd')
)
if success:
    print(f"✔ 정상 저장: {save_path}")
else:
    print("⚠️ Open3D 저장 실패")

# ─── 7. (선택) 시각화 ─────────────────────────────────────────
o3d.visualization.draw_geometries([
    hole_cloud
], window_name="Hole Points")

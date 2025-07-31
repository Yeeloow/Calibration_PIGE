#!/usr/bin/env python3
"""
**S003_Line_ID_batch.py ― 반복 2D 플롯 + 결과 폴더 선택**

* Base + Cluster 매핑으로 홀‑라인 거리를 계산.
* 3가지 플롯 모드: `none`, `each`, `loop`.
* **결과 저장 폴더를 실행 시 탐색기에서 선택**하도록 수정.
  선택한 폴더에 `OUTPUT_CSV` 이름으로 결과가 저장됩니다.
"""

# ──────────────── 사용자 설정 ────────────────
ELLIPSE_CSV = None              # None → 파일 선택 대화상자
OUTPUT_CSV  = "line_hole_distances.csv"  # 저장 파일명
SAVE_DIR    = None              # None → 폴더 선택 대화상자
PLOT_MODE   = "loop"            # "none" | "each" | "loop"

# ROI & 알고리즘 파라미터
ROI_X          = None          # (xmin,xmax)
# ROI_Y          = (9.8, 10.2) #T001
# ROI_Y          = (6.5, 7) #T001
ROI_Y          = (8.5, 9) #T002
K_NEIGHBORS    = 7
GRAD_THRESHOLD = 2.5
RANSAC_THRESH  = 0.03
DOWNSAMPLE_PCT = 50            # % (loop 모드에도 적용)
# ────────────────────────────────────────────

import os, sys, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
try:
    import open3d as o3d
except ImportError:
    o3d = None
try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None

plt.rcParams.update({"figure.max_open_warning": 0})

# ── 보조 함수 ─────────────────────────────────

def ask_files(title):
    if tk is None:
        return input(f"{title} (공백 구분): ").strip().split()
    root = tk.Tk(); root.withdraw()
    paths = filedialog.askopenfilenames(title=title,
                                        filetypes=[("Point-cloud", "*.csv *.ply"), ("All", "*.*")])
    root.destroy(); return list(paths)


def ask_file(title):
    if tk is None:
        return input(f"{title}: ").strip()
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(title=title)
    root.destroy(); return path


def ask_dir(title):
    if tk is None:
        return input(f"{title}: ").strip()
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy(); return path


def load_point_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, usecols=[0, 1, 2], dtype=np.float32)
        return df.values
    elif ext == ".ply":
        if o3d is None:
            raise ImportError("open3d required → pip install open3d")
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def edge_points(P, k, thr):
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree", leaf_size=32).fit(P[:, :2])
    idx = nn.kneighbors(return_distance=False)
    nb = P[idx[:, 1:], :3]
    p0 = P[:, None, :3]
    dy = nb[..., 1] - p0[..., 1]
    dz = nb[..., 2] - p0[..., 2]
    denom = (dy * dy).sum(1)
    beta = np.where(denom > 0, (dy * dz).sum(1) / denom, 0)
    return P[np.abs(beta) > thr]


# ── 전역 플롯 레코드 ───────────────────────────
PLOT_RECORDS = []


def process_file(path, hole_xy, save_plot_data=False):
    pts = load_point_cloud(path)
    mask = np.ones(len(pts), bool)
    if ROI_X:
        mask &= (pts[:, 0] >= ROI_X[0]) & (pts[:, 0] <= ROI_X[1])
    if ROI_Y:
        mask &= (pts[:, 1] >= ROI_Y[0]) & (pts[:, 1] <= ROI_Y[1])
    pts_roi = pts[mask]
    if len(pts_roi) < 50:
        raise RuntimeError("ROI 내 점이 부족합니다 — ROI 설정 확인")
    if DOWNSAMPLE_PCT < 100:
        keep = np.random.rand(len(pts_roi)) < (DOWNSAMPLE_PCT / 100)
        pts_roi = pts_roi[keep]
    edges = edge_points(pts_roi, K_NEIGHBORS, GRAD_THRESHOLD)
    if len(edges) < 10:
        raise RuntimeError("경계점 부족 — 파라미터 확인")
    ransac = RANSACRegressor(residual_threshold=RANSAC_THRESH,
                             max_trials=50, min_samples=50, random_state=0)
    ransac.fit(edges[:, 0:1], edges[:, 1])
    m = float(ransac.estimator_.coef_[0])
    c = float(ransac.estimator_.intercept_)
    x0, y0 = hole_xy
    dist = abs(m * x0 - y0 + c) / np.hypot(m, 1)
    if PLOT_MODE == "each":
        _quick_plot(path, pts_roi, edges, m, c, hole_xy)
    elif PLOT_MODE == "loop" and save_plot_data:
        PLOT_RECORDS.append(dict(path=path, pts=pts_roi, edges=edges,
                                 m=m, c=c, hole=hole_xy, dist=dist))
    return m, c, dist


def _quick_plot(path, pts, edges, m, c, hole_xy):
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap="viridis", s=1)
    plt.colorbar(sc, label="z")
    plt.scatter(edges[:, 0], edges[:, 1], color="orange", s=8)
    xs = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 200)
    plt.plot(xs, m * xs + c, "r-", lw=2)
    plt.scatter([hole_xy[0]], [hole_xy[1]], color="red", marker="x", s=60)
    plt.title(os.path.basename(path))
    plt.tight_layout(); plt.show()


# ── 메인 ─────────────────────────────────────
if __name__ == "__main__":
    pc_files = ask_files("Select point-cloud file(s)")
    if not pc_files:
        sys.exit("파일을 선택하지 않았습니다.")

    if ELLIPSE_CSV is None:
        ELLIPSE_CSV = ask_file("Select ellipse_results.csv")
    if not os.path.exists(ELLIPSE_CSV):
        sys.exit("ellipse_results.csv를 찾을 수 없습니다.")

    if SAVE_DIR is None:
        SAVE_DIR = ask_dir("Select folder to save results")
    if not SAVE_DIR:
        print("[WARN] 저장 폴더를 선택하지 않아 현재 작업 디렉터리에 저장합니다.")
        SAVE_DIR = os.getcwd()

    df = pd.read_csv(ELLIPSE_CSV)
    std = {c: ''.join(ch.lower() for ch in str(c) if ch.isalnum()) for c in df.columns}
    BASE_KEYS, CLUSTER_KEYS = {"base", "file", "filename", "sample", "파일", "파일명"}, {"cluster", "clusterid", "클러스터", "클러스터id"}
    CX_KEYS, CY_KEYS = {"centerx", "cx", "x", "center_x"}, {"centery", "cy", "y", "center_y"}
    def find_col(keys):
        for col, s in std.items():
            if s in keys:
                return col
        return None
    base_col, cy_col, cx_col = map(find_col, (BASE_KEYS, CX_KEYS, CY_KEYS))
    cluster_col = find_col(CLUSTER_KEYS)
    if None in (base_col, cx_col, cy_col):
        print("[ERROR] Base/Center_X/Y 열을 찾지 못했습니다 ⇒ CSV 헤더:")
        print(*df.columns, sep=" | ")
        sys.exit(1)
    df['base_lower'] = df[base_col].astype(str).str.lower()

    summary = []
    for path in pc_files:
        base = os.path.splitext(os.path.basename(path))[0]
        rows = df[df['base_lower'] == base.lower()]
        if rows.empty:
            warnings.warn(f"{base}: ellipse_results.csv에 항목 없음 → 건너뜀")
            continue
        groups = rows.groupby(cluster_col) if cluster_col else [(None, rows)]
        for clu_id, grp in groups:
            cx, cy = grp[cx_col].mean(), grp[cy_col].mean()
            try:
                m, c_line, d = process_file(path, (cx, cy), save_plot_data=(PLOT_MODE=="loop"))
                summary.append((base, clu_id if clu_id is not None else "", cx, cy, m, c_line, d))
                print(f"{base}[{clu_id}] → d={d:.6f}")
            except Exception as e:
                warnings.warn(f"{base}[{clu_id}]: {e}")

    if summary:
        out_path = os.path.join(SAVE_DIR, OUTPUT_CSV)
        pd.DataFrame(summary, columns=["File", "Cluster", "Center_X", "Center_Y", "Slope", "Intercept", "Distance"]).to_csv(
            out_path, index=False, float_format="%.6f")
        print(f"\nSummary saved → {out_path}")
    else:
        print("처리된 결과가 없습니다.")

    if PLOT_MODE == "loop" and PLOT_RECORDS:
        plt.ion(); fig, ax = plt.subplots(figsize=(8, 5))
        for rec in PLOT_RECORDS:
            ax.clear()
            pts, edges = rec['pts'], rec['edges']
            m, c_line = rec['m'], rec['c']; x0, y0 = rec['hole']
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap="viridis", s=1)
            ax.scatter(edges[:, 0], edges[:, 1], color="orange", s=8)
            xs = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 200)
            ax.plot(xs, m * xs + c_line, "r-", lw=2)
            ax.scatter([x0], [y0], color="red", marker="x", s=60)
            ax.set_title(f"{os.path.basename(rec['path'])}  (dist={rec['dist']:.4f})")
            plt.tight_layout(); plt.draw()
            print("[Enter] → 다음 / 'q'+Enter → 종료")
            if input().lower() == 'q':
                break
        plt.ioff(); plt.show()

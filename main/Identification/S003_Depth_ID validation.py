import os
import numpy as np
import pandas as pd
import open3d as o3d
import tkinter as tk
from tkinter import filedialog

def load_points(path: str) -> np.ndarray:
    """Load PLY or CSV point cloud."""
    if path.lower().endswith(".ply"):
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)
    elif path.lower().endswith(".csv"):
        with open(path, "r") as f:
            first = f.readline()
            has_header = any(ch.isalpha() for ch in first)
        return np.loadtxt(path, delimiter=",", skiprows=1 if has_header else 0)
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. (csv 또는 ply)")

def fit_plane_with_inliers(points: np.ndarray, dist_thresh=0.03, ransac_n=10, num_iter=1000):
    """RANSAC으로 평면을 찾고, inliers로 least squares 재피팅."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=dist_thresh,
        ransac_n=ransac_n,
        num_iterations=num_iter
    )
    inlier_pts = points[inliers]

    # inliers로 least squares fitting
    A = np.c_[inlier_pts[:,0], inlier_pts[:,1], np.ones_like(inlier_pts[:,0])]
    b = -inlier_pts[:,2]
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, d = x
    c = 1.0

    normal = np.array([a, b_, c], dtype=float)
    normal /= np.linalg.norm(normal)
    d /= np.linalg.norm([a, b_, c])
    return normal, d

def point_to_plane_distance(n: np.ndarray, d: float, point: np.ndarray) -> float:
    """Calculate distance between a point and a plane."""
    return abs(np.dot(n, point) + d)

def visualize_result(res):
    """Visualize point cloud, plane, and hole center in Open3D."""
    ply_path = res['PLY_path']
    hole_pt = np.array([res['Center_X'], res['Center_Y'], res['Center_Z']])

    pts = load_points(ply_path)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.paint_uniform_color([0.8,0.8,0.8])

    n, d = fit_plane_with_inliers(pts)

    # 평면 메쉬 생성
    signed = np.dot(n, hole_pt) + d
    proj = hole_pt - signed * n
    ref = np.array([0,0,1]) if abs(n[2])<0.9 else np.array([0,1,0])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    scale = (pts.max(axis=0)-pts.min(axis=0)).max() * 0.4
    gs = np.linspace(-scale, scale, 10)
    verts = np.array([proj + a*u + b*v for a in gs for b in gs])
    n_g = len(gs)
    tris = []
    for i in range(n_g-1):
        for j in range(n_g-1):
            idx = i*n_g + j
            tris += [[idx, idx+1, idx+n_g],
                     [idx+1, idx+1+n_g, idx+n_g]]
    plane = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris))
    plane.paint_uniform_color([0.7,0.7,0.7])
    plane.compute_vertex_normals()

    # 홀 중심점 표시
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale*0.05)
    sphere.translate(hole_pt)
    sphere.paint_uniform_color([1,0,0])
    sphere.compute_vertex_normals()

    o3d.visualization.draw_geometries(
        [pcd, plane, sphere],
        window_name=f"{os.path.basename(ply_path)} | Cluster {res['Cluster']}",
        width=800, height=600
    )

def main():
    root = tk.Tk(); root.withdraw()

    # 평면을 피팅할 원본 PLY 파일들 선택
    print("평면 피팅할 PLY 파일들을 선택하세요.")
    ply_paths = filedialog.askopenfilenames(
        title="Select one or more PLY files",
        filetypes=[("PLY files","*.ply")]
    )
    if not ply_paths:
        raise RuntimeError("적어도 하나의 PLY 파일을 선택해야 합니다.")

    # 클러스터 요약 CSV 선택
    print("ellipse_results.csv 파일을 선택하세요.")
    csv_path = filedialog.askopenfilename(
        title="Select ellipse_results.csv",
        filetypes=[("CSV files","*.csv")]
    )
    if not csv_path:
        raise RuntimeError("CSV 파일을 선택해야 합니다.")

    df = pd.read_csv(csv_path)
    df['Base_lower'] = df['Base'].str.lower()

    results = []
    for ply_path in ply_paths:
        base = os.path.splitext(os.path.basename(ply_path))[0]
        bl = base.lower()
        df_base = df[df['Base_lower']==bl]
        if df_base.empty:
            print(f"[경고] `{base}`에 대응되는 CSV 엔트리 없음.")
            continue

        pts = load_points(ply_path)
        n, d = fit_plane_with_inliers(pts)

        # 클러스터별 평균 및 거리 계산
        for cluster_id, group in df_base.groupby('Cluster'):
            count = len(group)
            center = group[['Center_Y','Center_X','Center_Z']].mean().values
            dist = point_to_plane_distance(n, d, center)

            results.append({
                'PLY_path'      : ply_path,
                'Base'          : base,
                'Cluster'       : int(cluster_id),
                'Cluster_size'  : count,
                'Center_X'      : center[0],
                'Center_Y'      : center[1],
                'Center_Z'      : center[2],
                'Distance'      : dist
            })

    if not results:
        print("유효한 결과가 없습니다. 종료합니다.")
        return

    # 결과 저장
    out_df = pd.DataFrame(results)
    print("결과를 저장할 CSV 경로를 선택하세요.")
    save_path = filedialog.asksaveasfilename(
        title="Save results as CSV",
        defaultextension=".csv",
        filetypes=[("CSV files","*.csv")]
    )
    if save_path:
        out_df.to_csv(save_path, index=False)
        print(f"▶ `{save_path}`에 저장되었습니다.")

    # 시각화 반복
    print("\n[Visualization] 인덱스를 입력하세요 (q 또는 빈값 = 종료):")
    for i, r in enumerate(results):
        print(f"  [{i}] {r['Base']} | Clu={r['Cluster']} | Size={r['Cluster_size']} | Dist={r['Distance']:.4f}")

    while True:
        sel = input("Index> ").strip()
        if sel.lower()=='q' or sel=='':
            print("시각화 종료.")
            break
        try:
            idx = int(sel)
            visualize_result(results[idx])
        except Exception as e:
            print("잘못된 입력입니다:", e)

if __name__ == "__main__":
    main()

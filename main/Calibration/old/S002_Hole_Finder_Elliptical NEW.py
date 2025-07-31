import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import pandas as pd
from skimage.measure import EllipseModel
from scipy.spatial import ConvexHull


def process_pointcloud(input_path):
    # CSV 로드
    df = pd.read_csv(input_path)
    if not {'x', 'y', 'z'}.issubset(df.columns):
        print(f"{os.path.basename(input_path)}: CSV must contain 'x','y','z' columns")
        return None

    pts = df[['x', 'y', 'z']].to_numpy()
    xy = pts[:, :2]
    z_vals = pts[:, 2]

    # 경계 검출을 위한 Convex Hull
    try:
        hull = ConvexHull(xy)
        boundary_idx = hull.vertices
    except Exception as e:
        print(f"{os.path.basename(input_path)}: Convex hull failed - {e}")
        return None

    boundary_xy = xy[boundary_idx]
    boundary_pts_3d = pts[boundary_idx]

    # Ellipse fitting
    model = EllipseModel()
    if not model.estimate(boundary_xy):
        print(f"{os.path.basename(input_path)}: Ellipse fitting failed")
        return None
    xc, yc, a, b, theta = model.params

    # z 중간값(홀 내부 평균)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx = xy[:, 0] - xc
    dy = xy[:, 1] - yc
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t
    inside_mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1
    zc = np.mean(z_vals[inside_mask]) if inside_mask.any() else np.nan

    # 평면 법선 벡터 계산 (PCA)
    centroid = boundary_pts_3d.mean(axis=0)
    cov = np.cov((boundary_pts_3d - centroid).T)
    _, _, vt = np.linalg.svd(cov)
    normal = vt[2, :]
    normal_unit = normal / np.linalg.norm(normal)
    nx, ny, nz = normal_unit

    return xc, yc, zc, nx, ny, nz


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    # 입력 폴더 선택
    input_dir = filedialog.askdirectory(title='Select Input Folder (CSV files)')
    if not input_dir:
        print('No input folder selected. Exiting.')
        exit(1)

    # 결과 저장 파일 선택
    output_file = filedialog.asksaveasfilename(
        title='Save Combined Results As',
        defaultextension='.csv',
        filetypes=[('CSV files', '*.csv')]
    )
    if not output_file:
        print('No output file specified. Exiting.')
        exit(1)

    # 모든 CSV 처리 및 결과 수집
    results = []
    for fname in sorted(os.listdir(input_dir)):
        if fname.lower().endswith('.csv'):
            path = os.path.join(input_dir, fname)
            res = process_pointcloud(path)
            if res is not None:
                xc, yc, zc, nx, ny, nz = res
                results.append({
                    'File_Name': fname,
                    'Center_X': round(-yc, 3),
                    'Center_Y': round(-xc, 3),
                    'Center_Z': round(zc, 3),
                    'Normal_X': round(nx, 3),
                    'Normal_Y': round(ny, 3),
                    'Normal_Z': round(nz, 3)
                })

    # 결과 저장
    if results:
        df_out = pd.DataFrame(results,
                              columns=['File_Name', 'Center_X', 'Center_Y', 'Center_Z',
                                       'Normal_X', 'Normal_Y', 'Normal_Z'])
        df_out.to_csv(output_file, index=False)
        print(f'Saved combined results to {output_file}')
    else:
        print('No valid results to save.')

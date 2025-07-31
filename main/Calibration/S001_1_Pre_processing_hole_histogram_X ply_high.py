import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import open3d as o3d  # 추가
import os
import tkinter as tk
from tkinter import filedialog
import Plot
import Plot_Origin

def fit_plane(points):
    """Fit a plane to the 3D points using least squares."""
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    Z = points[:, 2]
    coeffs, _, _, _ = lstsq(A, Z)
    return coeffs

def read_ply_as_dataframe(file_path):
    """Read a PLY file and return a DataFrame with x, y, z columns."""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    return df

def filter_points_by_plane(file_path, output_path, choice, min_z, max_z, idx):
    border_margin_Y_upper = 0
    border_margin_Y_low = 0
    border_margin_X_right = 10
    border_margin_X_left = 3

    print(f"Filtering point cloud {file_path} based on the plane...")

    # 파일 확장자에 따라 읽기 방법 선택
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.ply'):
        df = read_ply_as_dataframe(file_path)
    else:
        raise ValueError("Unsupported file format. Only CSV and PLY are supported.")

    points = df[['x', 'y', 'z']].to_numpy()

    if choice == 1:
        # Z 값에 대한 히스토그램 출력
        plt.hist(points[:, 2], bins=100, color='blue', alpha=0.7)
        plt.xlabel('Z value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Z values')
        plt.show()
        # 사용자로부터 최소, 최대 Z값 입력받기
        min_z = float(input("Enter minimum Z value for plane fitting: "))
        max_z = float(input("Enter maximum Z value for plane fitting: "))

    # 사용자 입력 범위 내의 점으로 평면 정의
    filtered_points = points[(points[:, 2] >= min_z) & (points[:, 2] <= max_z)]

    # margin 기준 추가 필터링
    filtered_points = filtered_points[
        (filtered_points[:, 0] > border_margin_X_left) &
        (filtered_points[:, 0] < filtered_points[:, 0].max() - border_margin_X_right) &
        (filtered_points[:, 1] > border_margin_Y_low) &
        (filtered_points[:, 1] < filtered_points[:, 1].max() - border_margin_Y_upper)
    ]

    print(f"Points before filtering: {len(points)}, after filtering: {len(filtered_points)}")

    filtered_df = pd.DataFrame(filtered_points, columns=['x', 'y', 'z'])
    filtered_df.to_csv(output_path, index=False)
    print(f"Filtered point cloud saved to {output_path}")

def main():
    root = tk.Tk()
    root.withdraw()

    input_folder = filedialog.askdirectory(title="Select the folder containing the point cloud files")
    if not input_folder:
        print("No folder selected. Exiting...")
        return

    output_folder = filedialog.askdirectory(title="Select the folder to save the filtered files")
    if not output_folder:
        print("No folder selected. Exiting...")
        return

    # csv 또는 ply만 리스트업
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.csv') or f.endswith('.ply')]
    if not input_files:
        print(f"No CSV or PLY files found in {input_folder}. Exiting...")
        return

    choice_input = float(input("Enter Histogram show, 1: "))
    if choice_input != 1:
        min_z = float(input("Enter minimum Z value for plane fitting: "))
        max_z = float(input("Enter maximum Z value for plane fitting: "))
    else:
        min_z = 0
        max_z = 0

    idx = 1
    for input_file in input_files:
        input_file_path = os.path.join(input_folder, input_file)
        output_file_name = os.path.splitext(input_file)[0] + '_filtered.csv'  # 항상 csv로 저장
        output_file_path = os.path.join(output_folder, output_file_name)

        filter_points_by_plane(input_file_path, output_file_path, choice_input, min_z, max_z, idx)
        idx += 1
        # Plot_Origin.main()  # 필요시 활성화

if __name__ == "__main__":
    main()

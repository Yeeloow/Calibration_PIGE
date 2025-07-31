import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

# 동차좌표 변환 함수
def to_homogeneous(points):
    """
    (N,3) 배열을 (N,4) 동차좌표로 변환 (마지막 열에 1 추가).
    이미 (N,4)인 경우 그대로 반환.
    """
    if points.shape[1] == 4:
        return points
    elif points.shape[1] == 3:
        ones = np.ones((points.shape[0], 1))
        return np.hstack((points, ones))
    else:
        raise ValueError("입력 데이터는 (N,3) 또는 (N,4) 형태여야 합니다.")

# 순수 평행이동 동차변환 행렬 생성
def make_translation_matrix(p):
    """3D 점 p = [x, y, z]를 4x4 동차변환 행렬(평행이동만)로 생성"""
    T = np.eye(4)
    T[0:3, 3] = p
    return T

############################################
# 보정 변환 X 행렬 구성 (식별된 파라미터 이용)
############################################
def make_X_matrix_from_params(params):
    """
    params = [alpha, beta, gamma, DeltaX, DeltaY, DeltaZ]
    소각 근사를 사용하여 회전 행렬 X_R를 구성하고, 평행이동 행렬 X_T와 결합하여 X = X_T @ X_R 반환.
    """
    alpha, beta, gamma, dX, dY, dZ = params
    X_R = np.array([
        [1,       -gamma,   beta,   0],
        [gamma,    1,      -alpha, 0],
        [-beta,    alpha,    1,     0],
        [0,        0,        0,     1]
    ])
    X_T = np.eye(4)
    X_T[0:3, 3] = np.array([dX, dY, dZ])
    return X_T @ X_R

############################################
# 식별된 파라미터 불러오기
############################################
def load_identified_parameters():
    """
    파일 탐색기를 통해 식별된 파라미터 CSV 파일을 선택하여, 
    "alpha, beta, gamma, DeltaX, DeltaY, DeltaZ" 형태의 헤더를 가진 파일에서 파라미터를 불러옵니다.
    """
    root = tk.Tk()
    root.withdraw()
    param_file = filedialog.askopenfilename(
        title="Select Identified Parameters CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    if param_file:
        # 헤더가 있으므로 skiprows=1을 사용합니다.
        params = np.loadtxt(param_file, delimiter=",", skiprows=1)
        return params
    else:
        raise FileNotFoundError("식별된 파라미터 파일을 선택하지 않았습니다.")

############################################
# 측정 데이터 불러오기 (MCS_OMM와 all_circles_f1)
############################################
def load_measurement_data():
    """
    파일 탐색기를 통해:
      1) MCS_OMM.csv: 기계 TCP 위치 (센서 설치 위치) → T_sensor_all (각 행을 동차변환 행렬로)
      2) all_circles_f1.csv: 스캐너 측정 좌표 → p_sensor_all (동차좌표)
    반환:
      T_sensor_all: (N,4,4) 배열
      p_sensor_all: (N,4) 배열
    """
    root = tk.Tk()
    root.withdraw()
    
    mcs_file = filedialog.askopenfilename(
        title="Select MCS_OMM CSV file (기계 TCP 위치)",
        filetypes=[("CSV Files", "*.csv")]
    )
    circles_file = filedialog.askopenfilename(
        title="Select all_circles_f1 CSV file (스캐너 측정 좌표)",
        filetypes=[("CSV Files", "*.csv")]
    )
    
    # MCS_OMM.csv: 첫 번째 열은 인덱스로 처리
    mcs_df = pd.read_csv(mcs_file, index_col=0)
    if set(['MCS_X', 'MCS_Y', 'MCS_Z']).issubset(mcs_df.columns):
        T_points = mcs_df[['MCS_X', 'MCS_Y', 'MCS_Z']].to_numpy()
    else:
        T_points = mcs_df.iloc[:, :3].to_numpy()
    
    # all_circles_f1.csv: 스캐너 측정 좌표
    circles_df = pd.read_csv(circles_file)
    if set(['Center_X', 'Center_Y', 'Center_Z']).issubset(circles_df.columns):
        p_sensor_raw = circles_df[['Center_X', 'Center_Y', 'Center_Z']].to_numpy()
    else:
        p_sensor_raw = circles_df.iloc[:, :3].to_numpy()
    
    # 동차좌표로 변환
    p_sensor_all = to_homogeneous(p_sensor_raw)  # (N,4)
    
    # T_sensor_all: 각 MCS_OMM 좌표를 순수 평행이동 행렬로 변환
    N = T_points.shape[0]
    T_sensor_all = np.zeros((N, 4, 4))
    for i in range(N):
        T_sensor_all[i] = make_translation_matrix(T_points[i])
    
    return T_sensor_all, p_sensor_all

############################################
# 센서 측정값을 보정 변환 X를 이용해 MCS 좌표로 변환
############################################
def transform_sensor_to_MCS(T_sensor_all, p_sensor_all, X_est):
    """
    각 샘플에 대해, 보정 변환 X_est를 적용하여
      H_est = T_sensor * X_est * p_sensor
    를 계산하고, H_est의 앞 3개 성분을 반환합니다.
    반환:
      H_est_all: (N,3) 배열
    """
    N = T_sensor_all.shape[0]
    H_est_all = []
    for i in range(N):
        p_vec = p_sensor_all[i].reshape(4, 1)
        H_est = T_sensor_all[i] @ X_est @ p_vec
        H_est_all.append(np.squeeze(H_est)[:3])
    return np.array(H_est_all)

############################################
# 결과 저장 함수
############################################
def save_transformed_results(output_dir, H_est_all):
    """
    output_dir: 결과를 저장할 폴더 경로
    H_est_all: (N,3) 배열, 변환된 MCS 좌표
    CSV 파일("transformed_coordinates.csv")로 저장
    """
    file_path = os.path.join(output_dir, "transformed_coordinates.csv")
    df = pd.DataFrame(H_est_all, columns=["X", "Y", "Z"])
    df.to_csv(file_path, index=False)
    print(f"변환된 MCS 좌표가 {file_path}에 저장되었습니다.")

############################################
# 메인 함수: 식별된 X를 이용해 센서 측정값을 MCS 좌표로 변환하고 결과 출력 및 저장
############################################
def main():
    # 1. 식별된 파라미터 불러오기
    params = load_identified_parameters()  # [alpha, beta, gamma, DeltaX, DeltaY, DeltaZ]
    print("불러온 식별 파라미터:", params)
    X_est = make_X_matrix_from_params(params)
    
    # 2. 측정 데이터 불러오기 (MCS_OMM 및 all_circles_f1)
    T_sensor_all, p_sensor_all = load_measurement_data()
    
    # 3. 보정 변환 적용: 센서 측정값을 MCS 좌표계로 변환
    H_est_all = transform_sensor_to_MCS(T_sensor_all, p_sensor_all, X_est)
    
    # 4. 변환 결과 출력
    print("변환된 MCS 좌표 (각 샘플):")
    for i, coord in enumerate(H_est_all):
        print(f"샘플 {i}: X = {coord[0]:.6f}, Y = {coord[1]:.6f}, Z = {coord[2]:.6f}")
    
    # 5. 결과 저장: 결과 저장 폴더 선택 후 CSV 파일로 저장
    root = tk.Tk()
    root.withdraw()
    output_dir = filedialog.askdirectory(title="Select Folder to Save Transformed Results")
    if output_dir:
        save_transformed_results(output_dir, H_est_all)
    else:
        print("결과 저장 폴더가 선택되지 않았습니다.")

if __name__ == '__main__':
    main()

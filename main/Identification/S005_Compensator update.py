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

# 보정 변환 X 행렬 구성 (식별된 파라미터 이용)
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

# 식별된 파라미터 불러오기
def load_identified_parameters():
    """
    파일 탐색기를 통해 식별된 파라미터 CSV 파일을 선택하여,
    "alpha, beta, gamma, DeltaX, DeltaY, DeltaZ" 형태의 헤더를 가진 파일에서 파라미터를 불러옵니다.
    """
    root = tk.Tk(); root.withdraw()
    param_file = filedialog.askopenfilename(
        title="Select Identified Parameters CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not param_file:
        raise FileNotFoundError("식별된 파라미터 파일을 선택하지 않았습니다.")
    params = np.loadtxt(param_file, delimiter=",", skiprows=1)
    print(f"Loaded parameters from {os.path.abspath(param_file)}")
    return params

# 측정 데이터 불러오기 (MCS_OMM와 all_circles_f1)
def load_measurement_data():
    root = tk.Tk(); root.withdraw()
    mcs_file = filedialog.askopenfilename(
        title="Select MCS_OMM CSV file (기계 TCP 위치)", filetypes=[("CSV Files", "*.csv")]
    )
    circles_file = filedialog.askopenfilename(
        title="Select all_circles_f1 CSV file (스캐너 측정 좌표)", filetypes=[("CSV Files", "*.csv")]
    )
    if not mcs_file or not circles_file:
        raise FileNotFoundError("측정 데이터 파일을 선택하지 않았습니다.")
    mcs_df = pd.read_csv(mcs_file, index_col=0)
    if set(['MCS_X','MCS_Y','MCS_Z']).issubset(mcs_df.columns):
        T_points = mcs_df[['MCS_X','MCS_Y','MCS_Z']].to_numpy()
    else:
        T_points = mcs_df.iloc[:, :3].to_numpy()
    circles_df = pd.read_csv(circles_file)
    if set(['Center_X','Center_Y','Center_Z']).issubset(circles_df.columns):
        p_sensor_raw = circles_df[['Center_X','Center_Y','Center_Z']].to_numpy()
    else:
        p_sensor_raw = circles_df.iloc[:, :3].to_numpy()
    p_sensor_all = to_homogeneous(p_sensor_raw)
    N = T_points.shape[0]
    T_sensor_all = np.zeros((N,4,4))
    for i in range(N): T_sensor_all[i] = make_translation_matrix(T_points[i])
    return T_sensor_all, p_sensor_all

# 센서 측정값 보정 및 변환
def transform_sensor_to_MCS(T_sensor_all, p_sensor_all, X_est, Z_reference):
    N = T_sensor_all.shape[0]; H_est_all = []
    for i in range(N):
        x_local,y_local,z_measured = p_sensor_all[i][:3]
        Z_sensor = T_sensor_all[i][2,3]
        z_corrected = z_measured - (Z_sensor - Z_reference)
        p_corr_h = np.array([x_local,y_local,z_corrected,1]).reshape(4,1)
        H = T_sensor_all[i] @ X_est @ p_corr_h
        H_est_all.append(np.squeeze(H)[:3])
    return np.array(H_est_all)

# 결과 저장
def save_transformed_results(output_dir, H_est_all):
    file_path = os.path.join(output_dir, "transformed_coordinates.csv")
    pd.DataFrame(H_est_all,columns=["X","Y","Z"]).to_csv(file_path,index=False)
    print(f"변환된 MCS 좌표가 {file_path}에 저장되었습니다.")

# 로컬 경로로 Z_reference 지정
Z_REFERENCE_PATH = r"D:\001_Calibration\000. Experimental\Data\250508\250508\Z_reference.csv"

# 메인
if __name__=='__main__':
    # 1. 파라미터
    params = load_identified_parameters()
    X_est = make_X_matrix_from_params(params)

    # 2. Z_reference 읽기 (UTF-8-sig로 안전 처리)
    zref_file = os.path.abspath(Z_REFERENCE_PATH)
    try:
        Z_reference = float(pd.read_csv(zref_file,header=None,encoding='utf-8-sig').iat[0,0])
    except Exception:
        with open(zref_file,'r',encoding='utf-8-sig') as f: first_line=f.readline()
        Z_reference = float(first_line.strip().split(',')[0])
    print(f"Loaded Z_reference: {Z_reference} from {zref_file}")

    # 3. 측정 데이터
    T_sensor_all, p_sensor_all = load_measurement_data()

    # 4. 변환 및 보정
    H_est_all = transform_sensor_to_MCS(T_sensor_all,p_sensor_all,X_est,Z_reference)

    # 5. 출력 및 저장
    print("변환된 MCS 좌표:")
    for i,coord in enumerate(H_est_all):
        print(f"샘플 {i}: X={coord[0]:.6f},Y={coord[1]:.6f},Z={coord[2]:.6f}")
    root=tk.Tk();root.withdraw()
    out_dir=filedialog.askdirectory(title="Select Folder to Save Transformed Results")
    if out_dir: save_transformed_results(out_dir,H_est_all)
    else: print("결과 저장 폴더가 선택되지 않았습니다.")

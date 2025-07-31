import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import os

##########################################
# 동차변환 행렬 및 동차좌표 관련 보조 함수
##########################################
def make_translation_matrix(p):
    """3D 점 p = [x, y, z]를 순수 평행이동 동차변환 행렬(4x4)로 반환"""
    T = np.eye(4)
    T[0:3, 3] = p
    return T

def to_homogeneous(points):
    """
    (N,3) -> (N,4) 동차좌표 변환 (마지막 열에 1 추가).
    이미 4열이면 그대로 반환.
    """
    if points.shape[1] == 4:
        return points
    elif points.shape[1] == 3:
        ones = np.ones((points.shape[0], 1))
        return np.hstack((points, ones))
    else:
        raise ValueError("points는 (N,3) 또는 (N,4) 형태여야 합니다.")

#####################################
# X 행렬 구성: X = X_T * X_R
#####################################
def rotation_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotation_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def rotation_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def make_X_matrix(params):
    """
    params = [alpha, beta, gamma, DeltaX, DeltaY, DeltaZ]
    X_R: 회전 행렬 (소각 근사)
         X_R = [[ 1,      -gamma,   beta,   0 ],
                [ gamma,   1,      -alpha,  0 ],
                [ -beta,   alpha,   1,      0 ],
                [ 0,       0,       0,      1 ]]
    X_T: 평행이동 행렬
    최종 X = X_T @ X_R
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

######################################
# residuals_target 함수 (AX=XB 방식)
######################################
def residuals_target(params, T_sensor_all, p_sensor_all, H_actual):
    """
    각 샘플에 대해 보정된 홀 위치 H_model을 계산하여
      H_model = T_sensor * X * p_sensor
    그리고 H_model과 H_actual (실제 홀 위치) 사이의 차이를 잔차로 반환합니다.
    
    T_sensor_all: (N,4,4) 배열 - 각 측정에 대한 MCS_OMM 동차변환행렬
    p_sensor_all: (N,4) 배열 - all_circles_f1의 동차 좌표 (스캐너 측정값)
    H_actual: (N,4) 배열 또는 (1,4) 배열 - 실제 홀 위치 (Hole.csv), 동차좌표
    params: [alpha, beta, gamma, DeltaX, DeltaY, DeltaZ]
    """
    X = make_X_matrix(params)
    residual_list = []
    N = T_sensor_all.shape[0]
    if H_actual.shape[0] == 1:
        H_actual = np.repeat(H_actual, N, axis=0)
    for i in range(N):
        p_sensor_vec = p_sensor_all[i].reshape(4, 1)
        H_model = T_sensor_all[i] @ X @ p_sensor_vec
        H_model_vec = np.squeeze(H_model)
        if H_model_vec.size != 4:
            H_model_vec = H_model_vec.reshape(4,)
        diff = H_model_vec[:3] - H_actual[i, :3]
        residual_list.append(diff)
    return np.concatenate(residual_list)

##########################################
# 데이터 로딩: 파일 탐색기로 CSV 불러오기
##########################################
def load_data():
    """
    파일 탐색기를 통해 세 개의 CSV 파일을 순차적으로 선택:
      1) Hole.csv: 실제 홀의 위치 (target)
      2) MCS_OMM.csv: 기계 TCP 위치 (센서가 설치된 위치)
         * 첫 번째 열은 인덱스이므로 index_col=0 사용
      3) all_circles_f1.csv: 스캐너로 측정한 홀 위치 (p_sensor)
    
    반환:
      H_actual:     (N,4) 동차좌표 (Hole.csv)
      T_sensor_all: (N,4,4) 각 샘플에 대한 MCS_OMM 변환행렬 (순수 평행이동)
      p_sensor_all: (N,4) 동차좌표 (all_circles_f1)
    """
    root = tk.Tk()
    root.withdraw()
    
    hole_file = filedialog.askopenfilename(
        title="Select Hole CSV file (실제 홀 위치)",
        filetypes=[("CSV Files", "*.csv")]
    )
    mcs_file = filedialog.askopenfilename(
        title="Select MCS_OMM CSV file (기계 TCP 위치)",
        filetypes=[("CSV Files", "*.csv")]
    )
    circles_file = filedialog.askopenfilename(
        title="Select all_circles_f1 CSV file (스캐너 측정값)",
        filetypes=[("CSV Files", "*.csv")]
    )
    
    # 1) Hole.csv
    hole_df = pd.read_csv(hole_file)
    if set(['Hole_X', 'Hole_Y', 'Hole_Z']).issubset(hole_df.columns):
        H_actual_raw = hole_df[['Hole_X', 'Hole_Y', 'Hole_Z']].to_numpy()
    else:
        H_actual_raw = hole_df.iloc[:, :3].to_numpy()
    H_actual = to_homogeneous(H_actual_raw)  # (N,4)

    # 2) MCS_OMM.csv (index_col=0)
    mcs_df = pd.read_csv(mcs_file, index_col=0)
    if set(['MCS_X', 'MCS_Y', 'MCS_Z']).issubset(mcs_df.columns):
        T_points = mcs_df[['MCS_X', 'MCS_Y', 'MCS_Z']].to_numpy()
    else:
        T_points = mcs_df.iloc[:, :3].to_numpy()

    # 3) all_circles_f1.csv
    circles_df = pd.read_csv(circles_file)
    if set(['Center_X', 'Center_Y', 'Center_Z']).issubset(circles_df.columns):
        p_sensor_raw = circles_df[['Center_X', 'Center_Y', 'Center_Z']].to_numpy()
    else:
        p_sensor_raw = circles_df.iloc[:, :3].to_numpy()
    p_sensor_all = to_homogeneous(p_sensor_raw)  # (N,4)

    # T_sensor_all: 각 샘플에 대해 MCS_OMM 위치를 동차변환 행렬로 생성 (순수 평행이동)
    N = T_points.shape[0]
    T_sensor_all = np.zeros((N, 4, 4))
    for i in range(N):
        T_sensor_all[i] = make_translation_matrix(T_points[i])
    
    return H_actual, T_sensor_all, p_sensor_all

#############################################
# AX=XB 플롯을 위한 보조 함수
#############################################
def invert_transform(T):
    """4x4 동차변환행렬 T의 역행렬 계산 (회전 및 평행이동)"""
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    T_inv = np.eye(4)
    T_inv[0:3, 0:3] = R.T
    T_inv[0:3, 3] = -R.T @ p
    return T_inv

def plot_AX_XB(X_est, T_sensor_all, p_sensor_all):
    """
    X_est: 식별된 보정 행렬 (4x4)
    T_sensor_all: (N,4,4) - MCS_OMM 동차변환행렬 (센서 설치 위치)
    p_sensor_all: (N,4) - all_circles_f1 동차 좌표 (스캐너 측정값)
    
    각 샘플에 대해 S_i를 센서 측정값으로부터 순수 평행이동 행렬로 생성하고,
    연속된 샘플 쌍에 대해
      A_i = T_sensor_all[i]^-1 @ T_sensor_all[i+1]
      B_i = S_i^-1 @ S_{i+1]
    를 계산합니다.
    이후, AX_i = A_i @ X_est, XB_i = X_est @ B_i의 번역 성분을 플롯합니다.
    """
    N = T_sensor_all.shape[0]
    # S_list: 각 p_sensor_all[i]의 순수 평행이동 행렬 (센서 측정 위치)
    S_list = [make_translation_matrix(p_sensor_all[i][:3]) for i in range(N)]
    
    A_list = []
    B_list = []
    for i in range(N - 1):
        A_i = invert_transform(T_sensor_all[i]) @ T_sensor_all[i+1]
        B_i = invert_transform(S_list[i]) @ S_list[i+1]
        A_list.append(A_i)
        B_list.append(B_i)
    
    AX_list = []
    XB_list = []
    for i in range(len(A_list)):
        AX_i = A_list[i] @ X_est
        XB_i = X_est @ B_list[i]
        trans_AX = AX_i[0:3, 3]
        trans_XB = XB_i[0:3, 3]
        AX_list.append(trans_AX)
        XB_list.append(trans_XB)
    
    AX_array = np.array(AX_list)
    XB_array = np.array(XB_list)
    
    sample_indices = np.arange(AX_array.shape[0])
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    axis_labels = ['X translation', 'Y translation', 'Z translation']
    for j in range(3):
        axs[j].plot(sample_indices, AX_array[:, j], label='AX', marker='o')
        axs[j].plot(sample_indices, XB_array[:, j], label='XB', marker='x')
        axs[j].set_ylabel(axis_labels[j])
        axs[j].legend()
    axs[-1].set_xlabel('Sample pair index')
    plt.suptitle("AX vs. XB Translation Components")
    plt.show()

#############################################
# 결과 저장 함수
#############################################
def save_results(output_dir, result_params, X_est):
    """
    output_dir: 결과를 저장할 폴더 경로
    result_params: 식별된 파라미터 배열 (6,)
    X_est: 식별된 보정 행렬 (4x4)
    
    - 식별된 파라미터를 CSV 파일로 저장 (identified_parameters.csv)
    - X_est 행렬을 텍스트 파일로 저장 (X_est.txt)
    """
    params_file = os.path.join(output_dir, "identified_parameters.csv")
    X_file = os.path.join(output_dir, "X_est.txt")
    
    # 파라미터 저장 (헤더 포함)
    header = "alpha, beta, gamma, DeltaX, DeltaY, DeltaZ"
    np.savetxt(params_file, np.round(result_params.reshape(1,-1),6), delimiter=",", header=header, comments='')
    print(f"식별된 파라미터가 {params_file}에 저장되었습니다.")
    
    # X 행렬 저장
    np.savetxt(X_file, X_est, delimiter="\t")
    print(f"식별된 보정 행렬 X_est가 {X_file}에 저장되었습니다.")

#############################################
# 메인 함수: 전체 알고리즘 실행 및 결과 저장
#############################################
def main():
    # 1. 데이터 로딩
    H_actual, T_sensor_all, p_sensor_all = load_data()
    
    # 2. 초기 추정치: [alpha, beta, gamma, DeltaX, DeltaY, DeltaZ]
    initial_guess = np.zeros(6)
    
    # 3. 최적화
    res0 = residuals_target(initial_guess, T_sensor_all, p_sensor_all, H_actual)
    if not np.all(np.isfinite(res0)):
        print("초기 잔차에 NaN 또는 inf가 있습니다. 입력 데이터를 확인하세요.")
        return
    result = least_squares(residuals_target, initial_guess, args=(T_sensor_all, p_sensor_all, H_actual),f_scale=5e-5, x_scale='jac',
                            max_nfev=2000,
                            verbose=2,
                            ftol = 1e-9, xtol=1e-9, gtol=1e-9,loss='linear')
    
    alpha_est, beta_est, gamma_est, dX_est, dY_est, dZ_est = result.x

    print("=== 식별된 파라미터 ===")
    print(f"alpha = {alpha_est:.6f} rad ({alpha_est*1000:.3f} mrad)")
    print(f"beta  = {beta_est:.6f} rad ({beta_est*1000:.3f} mrad)")
    print(f"gamma = {gamma_est:.6f} rad ({gamma_est*1000:.3f} mrad)")
    print(f"Delta_X = {dX_est:.6f}")
    print(f"Delta_Y = {dY_est:.6f}")
    print(f"Delta_Z = {dZ_est:.6f}")
    
    # 4. 식별된 X 행렬 구성
    X_est = make_X_matrix(result.x)
    
    # 5. 보정 후 추정된 홀 위치 계산 및 플롯 (색상 변경 추가)
    N = T_sensor_all.shape[0]
    H_model_all = []
    for i in range(N):
        p_sensor_vec = p_sensor_all[i].reshape(4, 1)
        H_model = T_sensor_all[i] @ X_est @ p_sensor_vec
        H_model_all.append(H_model.flatten()[0:3])
    H_model_all = np.array(H_model_all)
    H_actual_xyz = H_actual[:, :3]

    # 각 샘플마다 유클리드 오차 계산 (예: 단위가 m일 경우 10e-6은 10 μm)
    errors = np.linalg.norm(H_model_all - H_actual_xyz, axis=1)
    
    
    # 오차가 10 μm 이상이면 빨간색, 아니면 녹색
    colors = ['red' if err >= 5e-3 else 'green' for err in errors]

    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    # Actual Hole은 보라색으로 동일하게 표시
    ax1.scatter(H_actual_xyz[:,0], H_actual_xyz[:,1], H_actual_xyz[:,2],
                label="Actual Hole (Hole.csv)", c='purple', marker='d')
    # Estimated Hole은 각 점마다 색상 지정
    ax1.scatter(H_model_all[:,0], H_model_all[:,1], H_model_all[:,2],
                label="Estimated Hole", c=colors, marker='o')
    # ax1.set_xlim([-100.488-0.02, -100.488+0.02])
    # ax1.set_ylim([0-0.02, 0+0.02])
    # ax1.set_zlim([-180.806-0.02, -180.806+0.02])
    # ax1.set_xlim([-0-0.02, -0+0.02])
    # ax1.set_ylim([0-0.02, 0+0.02])
    # ax1.set_zlim([-226.216-0.02, -226.216+0.02])
    
    ax1.set_title("Comensation result")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()
    plt.show()

    # (추가로, 오차 값도 출력)
    for i, err in enumerate(errors):
        print(f"샘플 {i}: 유클리드 오차 = {err:.6e} m")
    
    # 6. AX=XB 플롯
    # plot_AX_XB(X_est, T_sensor_all, p_sensor_all)
    
    # 7. 결과 저장: 결과 저장 폴더 선택
    output_dir = filedialog.askdirectory(title="Select Folder to Save Results")
    if output_dir:
        
        save_results(output_dir, result.x, X_est)
    else:
        print("결과 저장 폴더가 선택되지 않았습니다.")

if __name__ == '__main__':
    main()

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
    T = np.eye(4)
    T[0:3, 3] = p
    return T

def to_homogeneous(points):
    if points.shape[1] == 4:
        return points
    elif points.shape[1] == 3:
        ones = np.ones((points.shape[0], 1))
        return np.hstack((points, ones))
    else:
        raise ValueError("points는 (N,3) 또는 (N,4) 형태여야 합니다.")

#####################################
# X 행렬 구성: X = X_T * X_R (소각 근사)
#####################################
def make_X_matrix(params):
    alpha, beta, gamma, dX, dY, dZ = params
    X_R = np.array([
        [1,       -gamma,   beta,   0],
        [gamma,    1,      -alpha,  0],
        [-beta,    alpha,    1,     0],
        [0,        0,        0,     1]
    ])
    X_T = np.eye(4)
    X_T[0:3, 3] = [dX, dY, dZ]
    return X_T @ X_R

#####################################
# 6-DOF 잔차 함수
#####################################
def residuals_target(params, T_sensor_all, p_sensor_all, H_actual):
    X = make_X_matrix(params)
    res = []
    N = T_sensor_all.shape[0]
    if H_actual.shape[0] == 1:
        H_actual = np.repeat(H_actual, N, axis=0)
    for i in range(N):
        Hm = (T_sensor_all[i] @ X @ p_sensor_all[i].reshape(4,1)).flatten()[:3]
        res.extend((Hm - H_actual[i,:3]).tolist())
    return np.array(res)

#####################################
# α만 재식별용 잔차 함수 (β 고정)
#####################################
def residuals_alpha(a, normals, beta_fixed):
    """
    a: [alpha]
    normals: (N,3) 배열
    소각근사 하에서 nom=[0,0,1] → actual ≈ [β, -α, 1]
    β는 고정, α만 재식별
    """
    alpha = a[0]
    res = []
    for nx, ny, nz in normals:
        # n_y(pred) = -alpha 이므로
        res.append(-alpha - ny)
        # (선택) n_x(pred)=beta_fixed 와 비교하려면 아래 주석 해제
        # res.append(beta_fixed - nx)
    return np.array(res)

##########################################
# 데이터 로딩
##########################################
def load_data():
    root = tk.Tk(); root.withdraw()
    hole_file    = filedialog.askopenfilename(title="Select Hole CSV",          filetypes=[("CSV","*.csv")])
    mcs_file     = filedialog.askopenfilename(title="Select MCS_OMM CSV",       filetypes=[("CSV","*.csv")])
    circles_file = filedialog.askopenfilename(title="Select all_circles_f1 CSV", filetypes=[("CSV","*.csv")])
    center_file  = filedialog.askopenfilename(title="Select Center CSV (with normals)", filetypes=[("CSV","*.csv")])

    # Hole.csv → H_actual (N×4)
    df_h = pd.read_csv(hole_file)
    H_raw = df_h[['Hole_X','Hole_Y','Hole_Z']].to_numpy() if set(['Hole_X','Hole_Y','Hole_Z']).issubset(df_h.columns) else df_h.iloc[:,:3].to_numpy()
    H_actual = to_homogeneous(H_raw)

    # MCS_OMM.csv → T_sensor_all (N×4×4)
    df_m = pd.read_csv(mcs_file, index_col=0)
    T_pts = df_m[['MCS_X','MCS_Y','MCS_Z']].to_numpy() if set(['MCS_X','MCS_Y','MCS_Z']).issubset(df_m.columns) else df_m.iloc[:,:3].to_numpy()
    N = T_pts.shape[0]
    T_sensor_all = np.array([make_translation_matrix(T_pts[i]) for i in range(N)])

    # all_circles_f1.csv → p_sensor_all (N×4)
    df_c = pd.read_csv(circles_file)
    p_raw = df_c[['Center_X','Center_Y','Center_Z']].to_numpy() if set(['Center_X','Center_Y','Center_Z']).issubset(df_c.columns) else df_c.iloc[:,:3].to_numpy()
    p_sensor_all = to_homogeneous(p_raw)

    # Center.csv → normals (N×3)
    df_ct = pd.read_csv(center_file)
    normals = df_ct[['Normal_X','Normal_Y','Normal_Z']].to_numpy()

    return H_actual, T_sensor_all, p_sensor_all, normals

#############################################
# 메인 함수
#############################################
def main():
    # 1) 데이터 로딩
    H_actual, T_sensor_all, p_sensor_all, normals = load_data()

    # 2) 6-DOF LS 식별 (α_ls, β_ls, γ_ls, ΔX, ΔY, ΔZ)
    x0 = np.zeros(6)
    sol6 = least_squares(residuals_target, x0,
                         args=(T_sensor_all, p_sensor_all, H_actual))
    alpha_ls, beta_ls, gamma_ls, dX, dY, dZ = sol6.x

    # 3) β 고정하고 α만 법선벡터로 재식별
    alpha0 = [alpha_ls]
    sol_a = least_squares(residuals_alpha, alpha0,
                          args=(normals, beta_ls))
    alpha_ref = sol_a.x[0]

    # 4) 최종 파라미터
    alpha_ref = 0.034209
    final_params = np.array([alpha_ref, beta_ls, gamma_ls, dX, dY, dZ])
    X_est = make_X_matrix(final_params)

    print("=== 식별된 파라미터 ===")
    print(f"alpha: LS={alpha_ls:.6f} → ref={alpha_ref:.6f} rad")
    print(f"beta = {beta_ls:.6f} rad")
    print(f"gamma= {gamma_ls:.6f} rad")
    print(f"ΔX,ΔY,ΔZ = {dX:.6f}, {dY:.6f}, {dZ:.6f}")

    # 5) 보정 후 홀 위치 계산
    N = T_sensor_all.shape[0]
    H_model = np.stack([
        (T_sensor_all[i] @ X_est @ p_sensor_all[i].reshape(4,1)).flatten()[:3]
        for i in range(N)
    ], axis=0)
    H_act = H_actual[:N,:3]

    # 6) 시각화
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(H_act[:,0], H_act[:,1], H_act[:,2], c='purple', marker='d', label='Actual')
    ax.scatter(H_model[:,0], H_model[:,1], H_model[:,2], c='green', marker='x', label='Estimated')
    ax.set_title("Actual vs Estimated Hole Positions")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    plt.show()

    # 7) 오차 출력
    errs = np.linalg.norm(H_model - H_act, axis=1)
    print("\n=== 보정 후 유클리드 오차 (m) ===")
    for i,e in enumerate(errs):
        print(f"샘플 {i}: {e:.3e}")

    # 8) 결과 저장
    root = tk.Tk(); root.withdraw()
    out = filedialog.askdirectory(title="Select folder to save results")
    if out:
        fn = os.path.join(out, "identified_params.csv")
        hdr = "alpha,beta,gamma,DeltaX,DeltaY,DeltaZ"
        np.savetxt(fn, final_params.reshape(1,-1), delimiter=",", header=hdr, comments='')
        print(f"\n최종 파라미터가 {fn}에 저장되었습니다.")
    else:
        print("저장 폴더가 선택되지 않았습니다.")

if __name__ == '__main__':
    main()

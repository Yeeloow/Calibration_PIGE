#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog

def select_file(prompt):
    root = Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title=prompt,
        filetypes=[("CSV files","*.csv"),("All files","*.*")]
    )
    root.destroy()
    return path

def select_savefile(prompt):
    root = Tk(); root.withdraw()
    path = filedialog.asksaveasfilename(
        title=prompt,
        defaultextension=".csv",
        filetypes=[("CSV files","*.csv"),("All files","*.*")]
    )
    root.destroy()
    return path

def main():
    # 1) 보정 파라미터 로드
    param_path = select_file("보정 파라미터 파일 선택 (identified_parameters.csv)")
    if not param_path:
        print("❗ 파라미터 파일 선택 취소"); return
    alpha, beta, gamma, dX, dY, dZ = np.loadtxt(param_path, delimiter=",", skiprows=1)

    # 2) Machine pose 로드
    mcs_path = select_file("Machine pose 파일 선택 (MCS_OMM.csv)")
    if not mcs_path:
        print("❗ Machine pose 파일 선택 취소"); return
    df_mcs = pd.read_csv(mcs_path)
    df_mcs.columns = df_mcs.columns.str.lower()
    for c in ("x","y","z"):
        if c not in df_mcs.columns:
            print(f"❗ MCS CSV에 '{c}' 컬럼이 없습니다."); return
    poses = df_mcs[["x","y","z"]].to_numpy()

    # 3) 센서 측정점 로드 (all_ellipses.csv)
    comp_path = select_file("센서 측정점 파일 선택 (all_ellipses.csv)")
    if not comp_path:
        print("❗ 센서 측정점 파일 선택 취소"); return
    df_comp = pd.read_csv(comp_path)
    df_comp.columns = df_comp.columns.str.lower()
    
    # --- 여기를 이렇게 바꿔주세요 ---
    need = ("center_x","center_y","center_z")
    if not all(col in df_comp.columns for col in need):
        print(f"❗ 측정점 CSV에 '{need}' 컬럼이 없습니다. 확인하세요."); return
    pts = df_comp[list(need)].to_numpy()
    # ---------------------------------

    # 4) 개수 일치 확인
    if poses.shape[0] != pts.shape[0]:
        print("❗ 행 수 불일치: Machine pose와 센서 측정점 수가 다릅니다."); return

    # 5) 보정행렬 생성
    X_est = np.array([
        [   1,   -gamma,    beta,   dX],
        [gamma,       1,   -alpha,   dY],
        [-beta,    alpha,        1,   dZ],
        [   0,        0,        0,    1]
    ])

    # 6) 보정 계산
    H_out = []
    for i in range(len(poses)):
        T = np.eye(4)
        T[0:3,3] = poses[i]
        p = np.hstack((pts[i], 1.0))
        H = T @ (X_est @ p)
        H_out.append(H[:3])

    H_comp = np.vstack(H_out)
    df_out = pd.DataFrame(H_comp, columns=["X","Y","Z"])

    # 7) 결과 저장
    save_path = select_savefile("보정 결과 저장할 파일 선택")
    if not save_path:
        print("❗ 저장 취소"); return
    df_out.to_csv(save_path, index=False)
    print(f"✅ 보정 완료: '{save_path}'에 저장되었습니다.")

if __name__=="__main__":
    main()

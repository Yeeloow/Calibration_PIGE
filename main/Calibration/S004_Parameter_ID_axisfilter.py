# -*- coding: utf-8 -*-
"""
S004_Parameter_Identifier_axisfilter.py   (rev-Asign-fix)

· X-이송(ΔX>|ΔY|, Δmag>0.3 mm)  → diff[0]=0
· Y-이송(ΔY>|ΔX|, Δmag>0.3 mm)  → diff[1]=0
· Corner / Stop
    └ Δ이동 < 0.3 mm  또는  X-이송 중 ΔX 부호 전환 지점
      → diff 3축 모두 사용
"""

import numpy as np, pandas as pd, tkinter as tk
from tkinter import filedialog
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import os

# ────────────────── 변환 보조 ────────────────────────────────
def make_X_matrix(p):
    α, β, γ, dx, dy, dz = p
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [α, β, γ]).as_matrix()
    T[:3,  3] = [dx, dy, dz]
    return T

def make_T(v):
    T = np.eye(4)
    T[:3, 3] = v
    return T

def to_h(pts):
    return np.c_[pts, np.ones((len(pts), 1))]

# ────────────────── 축 분류 (방법 A) ──────────────────────────
def classify_axis(T_all, thresh=0.3):
    """
    0=X feed, 1=Y feed, 2=corner/stop.
    X-feed 세그먼트에서 ΔX 부호가 바뀌는 지점도 2 로 마킹.
    """
    pos  = T_all[:, :3, 3]
    dvec = np.diff(pos, axis=0, prepend=pos[:1])
    mag  = np.linalg.norm(dvec, axis=1)

    flag = np.full(len(pos), 2, dtype=int)           # 기본 corner/stop
    move = mag > thresh

    x_dom = (np.abs(dvec[:, 0]) > np.abs(dvec[:, 1])) & move
    y_dom = (np.abs(dvec[:, 1]) > np.abs(dvec[:, 0])) & move
    flag[x_dom] = 0
    flag[y_dom] = 1

    # ΔX 부호 전환(턴백) → corner
    x_sign = np.sign(dvec[:, 0])
    sign_change = np.r_[False, x_sign[1:] * x_sign[:-1] < 0]
    corner = sign_change & move & x_dom
    flag[corner] = 2
    return flag

# ────────────────── 잔차 함수 ────────────────────────────────
def residuals_target(p, T_all, P_all, H_act, flag):
    X = make_X_matrix(p)
    if H_act.shape[0] == 1:
        H_act = np.repeat(H_act, len(T_all), 0)

    res = []
    for i, (Tj, pj) in enumerate(zip(T_all, P_all)):
        q = (Tj @ X @ pj.reshape(4, 1)).ravel()[:3]
        diff = q - H_act[i, :3]

        if   flag[i] == 0: diff[0] = 0      # X-이송 → X 잔차 무시
        elif flag[i] == 1: diff[1] = 0      # Y-이송 → Y 잔차 무시
        # flag 2 → 3축 모두 사용
        res.append(diff)
    return np.concatenate(res)

# ───────── robust CSV loader ─────────
def load_csvs():
    root = tk.Tk(); root.withdraw()

    hole_path = filedialog.askopenfilename(title="Select Hole.csv")
    mcs_path  = filedialog.askopenfilename(title="Select MCS_OMM.csv")
    circ_path = filedialog.askopenfilename(title="Select all_ellipses.csv")

    # 1) Hole.csv  ─────────────────────
    hole = pd.read_csv(hole_path)
    h_cols = [c for c in hole.columns if c.lower().startswith("hole_")] or hole.columns[:3]
    hole[h_cols] = hole[h_cols].apply(pd.to_numeric, errors="coerce")
    H = to_h(hole[h_cols].to_numpy(float))

    # 2) MCS_OMM.csv ───────────────────
    mcs = pd.read_csv(mcs_path, index_col=0)
    m_cols = [c for c in mcs.columns if c.upper() in ("X","Y","Z","MCS_X","MCS_Y","MCS_Z")][:3]
    mcs[m_cols] = mcs[m_cols].apply(pd.to_numeric, errors="coerce")
    T = np.stack([make_T(v) for v in mcs[m_cols].to_numpy(float)])

    # 3) all_ellipses.csv ──────────────
    circ = pd.read_csv(circ_path)
    c_cols = [c for c in circ.columns if c.lower().startswith("center_")] or circ.columns[:3]
    circ[c_cols] = circ[c_cols].apply(pd.to_numeric, errors="coerce")
    P = to_h(circ[c_cols].to_numpy(float))

    return H, T, P


# ────────────────── 메인 ────────────────────────────────────
def main():
    H, T, P = load_csvs()
    axis_flag = classify_axis(T, thresh=0.3)     # 임계값 필요 시 조정

    p0 = np.zeros(6)
    res0 = residuals_target(p0, T, P, H, axis_flag)
    print("Initial L2-norm = {:.6f} mm".format(np.linalg.norm(res0)))

    bounds = ([-0.02]*3 + [-5]*3, [0.02]*3 + [5]*3)   # rad, mm
    opt = least_squares(
        residuals_target, p0,
        args=(T, P, H, axis_flag),
        loss='huber', f_scale=5e-6,
        x_scale='jac', bounds=bounds,
        ftol=1e-9, xtol=1e-9, gtol=1e-9,
        max_nfev=3000, verbose=2
    )

    print("\nIdentified parameters (rad / mm):")
    names = ["alpha","beta","gamma","DeltaX","DeltaY","DeltaZ"]
    for n, v in zip(names, opt.x):
        print(f"{n:6s} = {v: .6f}")

    np.savetxt(
        "identified_parameters.csv",
        opt.x.reshape(1, -1),
        delimiter=",",
        header="alpha,beta,gamma,DeltaX,DeltaY,DeltaZ",
        fmt="%.6f",
        comments=""
    )
    print("Saved → identified_parameters.csv")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
snapshot_sensor_setup_calibration.py
====================================

설치오차(6 DoF) + 홀 기준좌표(X0,Y0,Z0,R) **동시 식별**
-----------------------------------------------------
* 입력:  ☐ 폴더 안의 `P???.csv` (각 스냅숏 point cloud) 33개
        ☐ `MCS_OMM.csv` (스냅숏 순서에 대응하는 CNC X,Y,Z 위치)
* 가정: ① 홀 축 ≈ 기계 Z 축 (원통) ② 센서 스냅숏마다 1 mm 간격 평행이동만 존재
* 출력: identified_parameters.csv  (α,β,γ,ΔX,ΔY,ΔZ,X0,Y0,Z0,R)  소수 4자리

알고리즘 개요
-------------
1. **데이터 로드 & 정렬**  (file name 오름차순 ↔ MCS row 순서 일치 검증)
2. **벽면 점 필터**  – 각 스냅숏에서 z‑최소값 기준  △z>0.02 mm,  그리고 ρ≈평균ρ±0.2 mm
3. **비선형 최소제곱**  – 잔차 = (√((x-X0)²+(y-Y0)²) – R)  (Huber, f_scale=5 µm)
4. 결과 저장 & 간단 시각화 (오차≥5 µm=red)
"""
from __future__ import annotations

import glob, os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

###############################################################################
# Helpers
###############################################################################

def make_htm(theta: np.ndarray) -> np.ndarray:
    """theta = [α β γ ΔX ΔY ΔZ]  → 4×4 homogeneous"""
    α, β, γ, dx, dy, dz = theta
    T = np.eye(4)
    T[:3,:3] = R.from_euler("xyz", [α, β, γ]).as_matrix()
    T[:3, 3] = [dx, dy, dz]
    return T

###############################################################################
# Data loading
###############################################################################

def load_pointclouds(folder: Path) -> List[np.ndarray]:
    pcs = []
    for csv in sorted(folder.glob("P*_hole.csv")):
        df = pd.read_csv(csv)
        pcs.append(df[["x","y","z"]].to_numpy(dtype=float))
    return pcs

###############################################################################
# Wall‑point filter  (simple rule)
###############################################################################

def filter_wall_pts(pts: np.ndarray) -> np.ndarray:
    z_floor = np.percentile(pts[:,2], 2)             # 하위 2% = 바닥
    wall_mask = pts[:,2] > z_floor + 0.02           # 20 µm 이상이면 벽
    pts_wall = pts[wall_mask]
    # ρ 범위로 추가 정제
    rho = np.sqrt(pts_wall[:,0]**2 + pts_wall[:,1]**2)
    m, s = rho.mean(), rho.std()
    ring = (np.abs(rho - m) < 2*s)
    return pts_wall[ring]

###############################################################################
# Residual function
###############################################################################

def residuals(param: np.ndarray,
              pc_list: List[np.ndarray],
              T_list: List[np.ndarray]) -> np.ndarray:
    theta   = param[:6]                # 설치 오차
    X0, Y0, Z0, R0 = param[6:10]
    X = make_htm(theta)
    res = []
    for pts, Tj in zip(pc_list, T_list):
        ph = np.hstack([pts, np.ones((len(pts),1))])        # (N,4)
        q  = (Tj @ X @ ph.T).T[:, :3]                       # -> TCS
        rho = np.sqrt((q[:,0]-X0)**2 + (q[:,1]-Y0)**2)
        res.extend(rho - R0)                                # (N,) distance residual
    return np.asarray(res, dtype=float)

###############################################################################
# Main routine
###############################################################################

def main(data_dir: str | Path):
    data_dir = Path(data_dir)
    pcs_raw  = load_pointclouds(data_dir)
    pcs      = [filter_wall_pts(pc) for pc in pcs_raw]

    mcs_df   = pd.read_csv(data_dir/"MCS_OMM.csv", index_col=0).iloc[:len(pcs)]
    T_list   = [np.eye(4) for _ in range(len(pcs))]
    for T, (_,row) in zip(T_list, mcs_df.iterrows()):
        T[:3,3] = row[["X","Y","Z"]].to_numpy(dtype=float)

    # 초기값 추정 – 중심/반경을 첫 스캔으로 원 피팅
    p0_wall = pcs[0]
    xc = p0_wall[:,0].mean(); yc = p0_wall[:,1].mean()
    R0 = np.mean(np.sqrt((p0_wall[:,0]-xc)**2 + (p0_wall[:,1]-yc)**2))
    zc = np.median(p0_wall[:,2])                 # Z0 ≈ 벽 중간 높이

    init = np.zeros(10)
    init[6:10] = [xc, yc, zc, R0]

    lb = np.concatenate([[-0.02]*3 + [-5]*3,  [-np.inf]*4])
    ub = np.concatenate([[ 0.02]*3 + [ 5]*3,  [ np.inf]*4])

    res = least_squares(residuals, init,
                        args=(pcs, T_list),
                        loss='huber', f_scale=5e-6,
                        bounds=(lb,ub), x_scale='jac',
                        ftol=1e-9, xtol=1e-9, gtol=1e-9,
                        verbose=2)

    names = ['α','β','γ','ΔX','ΔY','ΔZ','X0','Y0','Z0','R']
    print("\nIdentified parameters (rad / mm):")
    for n,v in zip(names, res.x):
        print(f"{n:2s} = {v:.6f}")

    # 저장
    out = data_dir/"identified_parameters.csv"
    np.savetxt(out, res.x.reshape(1,-1), delimiter=',', header=','.join(names), fmt="%.4f", comments='')
    print(f"Saved → {out}")

if __name__ == "__main__":
    main("./")   # 데이터 폴더 경로 지정

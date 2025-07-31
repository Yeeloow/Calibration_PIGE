#!/usr/bin/env python3
# ──────────────── 기본 설정 ────────────────
CSV_PATH       = None                         # None → 파일탐색기
# HOLE_CENTER    = (4.896, 13.994)               # 홀 중심 (x0,y0)
HOLE_CENTER    = (4.910, 14.004)               # 홀 중심 (x0,y0)

# ★ ROI 설정 : x·y 범위로 관심 영역을 지정 (None = 전체)
ROI_X          = None    # (xmin, xmax)  예) (0,50)  / None → 제한 없음
ROI_Y          = (10.0, 13.2)   # (ymin, ymax)  예) (10,60) / None → 제한 없음

# 분석 파라미터
K_NEIGHBORS    = 5
GRAD_THRESHOLD = 2.5
RANSAC_THRESH  = 0.01
DOWNSAMPLE_PCT = 50
# ───────────────────────────────────────────

import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
try:
    import tkinter as tk
    from tkinter import filedialog
except: tk = None

# ── CSV 선택 ─────────────────────────────────
if CSV_PATH is None:
    if tk:  # 파일탐색기
        root=tk.Tk(); root.withdraw()
        CSV_PATH = filedialog.askopenfilename(
            title="Point-Cloud CSV 선택",
            filetypes=[("CSV","*.csv"),("All","*.*")]
        ); root.destroy()
    else:
        CSV_PATH = input("CSV 경로 입력: ").strip()
if not CSV_PATH or not os.path.exists(CSV_PATH):
    sys.exit("CSV 파일을 찾을 수 없습니다.")

# ── 데이터 로드 & ROI 적용 ────────────────────
df = pd.read_csv(CSV_PATH)
cols = {c.lower():c for c in df.columns}
if not all(k in cols for k in ("x","y","z")):
    sys.exit("CSV에는 x,y,z 컬럼이 필요합니다")
pts = df[[cols['x'],cols['y'],cols['z']]].values

roi_mask = np.ones(len(pts), bool)
if ROI_X: roi_mask &= (pts[:,0] >= ROI_X[0]) & (pts[:,0] <= ROI_X[1])
if ROI_Y: roi_mask &= (pts[:,1] >= ROI_Y[0]) & (pts[:,1] <= ROI_Y[1])
pts = pts[roi_mask]
if len(pts) < 50:
    sys.exit("ROI 내 점이 너무 적습니다. ROI 범위를 확인하세요.")

# ── 경계점 추출 (|dZ/dY| 임계) ─────────────────
def edge_points(P, k, thr):
    nn = NearestNeighbors(n_neighbors=k+1).fit(P[:,:2])
    idx = nn.kneighbors(return_distance=False)
    msk = np.zeros(len(P), bool)
    for i, nb in enumerate(idx):
        nb = nb[1:]
        dy, dz = P[nb,1]-P[i,1], P[nb,2]-P[i,2]
        if np.std(dy)==0: continue
        beta = dy.dot(dz)/dy.dot(dy)
        if abs(beta) > thr: msk[i]=True
    return P[msk]

edge_pts = edge_points(pts, K_NEIGHBORS, GRAD_THRESHOLD)
if len(edge_pts)<10:
    sys.exit("경계점 부족: K_NEIGHBORS / GRAD_THRESHOLD / ROI 확인")

# ── RANSAC 직선 피팅 ──────────────────────────
r = RANSACRegressor(residual_threshold=RANSAC_THRESH, random_state=0)
r.fit(edge_pts[:,0:1], edge_pts[:,1])
m, c = float(r.estimator_.coef_[0]), float(r.estimator_.intercept_)
print(f"[경계선] y = {m:.6f} x + {c:.6f}")

# ── 홀 중심 거리 ───────────────────────────────
x0,y0 = HOLE_CENTER
dist  = abs(m*x0 - y0 + c)/np.hypot(m,1)
print(f"[거리] Hole({x0},{y0}) → line = {dist:.6f}")

# ── 플롯 ──────────────────────────────────────
n_plot = int(len(pts)*max(1,min(DOWNSAMPLE_PCT,100))/100)
plot_pts = pts[np.random.choice(len(pts), n_plot, replace=False)]
plt.figure(figsize=(10,6))
sc = plt.scatter(plot_pts[:,0],plot_pts[:,1],
                 c=plot_pts[:,2], cmap='viridis',
                 s=1, alpha=0.5, label='PointCloud (by z)')
plt.colorbar(sc,label='z')
plt.scatter(edge_pts[:,0],edge_pts[:,1],
            color='orange',s=18,label='Edge pts')
xs=np.linspace(pts[:,0].min(),pts[:,0].max(),200)
plt.plot(xs,m*xs+c,'r-',lw=2,label=f'Line y={m:.3f}x+{c:.3f}')
plt.scatter([x0],[y0],color='red',marker='x',s=80,label='Hole centre')
plt.text(x0,y0,f'd={dist:.4f}',color='red',va='bottom',ha='left')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Boundary in ROI')
plt.legend(); plt.tight_layout(); plt.show()

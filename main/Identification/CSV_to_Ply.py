# csv_to_ply_grid_manual_vscode.py
# VSCode에서 바로 실행하여 Z-only CSV(격자형)를 x,y,z 포인트로 변환 후 PLY로 저장
# main_binary_saving.py와 동일한 PLY 포맷(바이너리) 사용

import os
import numpy as np

# ---------------- 사용자 지정 경로 및 파라미터 ----------------
# 변환하려는 CSV 파일 경로 (x,y,z 컬럼 필수)
CSV_FILE = r"C:\Users\heemi\Documents\Data\250506\Data\V001\001_Psss\T001\Raw\P005_005.csv"
# 생성할 PLY 파일 경로 (.ply 확장자)
PLY_FILE = r"C:\Users\heemi\Documents\Data\250506\Data\V001\002_CSV\T001\001_Raw\P005_005.ply"
# CSV에서 건너뛸 행 수 (헤더가 있을 경우), 격자형 CSV는 보통 0
SKIPROWS    = 0
# CSV 구분자
DELIMITER   = ','
# 격자 가로·세로 간격 (단위: micrometers)
SPACING_UM  = 5.0
# 무시할 Z값 (예: -99.9999), 없으면 None
SENTINEL    = None
# ----------------------------------------------------------

def load_csv_data(csv_path, skiprows, delimiter):
    data = np.loadtxt(csv_path, delimiter=delimiter, skiprows=skiprows)
    if data.ndim == 1:
        # 한 줄로 읽혔거나 1개 열인 경우, 최소 2차원으로 복원
        data = data.reshape(-1, 1)
    return data


def csv_to_ply(csv_path, ply_path, skiprows, delimiter, spacing_um, sentinel):
    # Z-only 격자 CSV 읽기
    data = load_csv_data(csv_path, skiprows, delimiter)

    # grid 형태 (data.shape[1] != 3)로 판단
    if data.ndim == 2 and data.shape[1] != 3:
        grid = data.astype(np.float32)
        rows, cols = grid.shape
        # X: 열 인덱스 * spacing, Y: 행 인덱스 * spacing
        # SPACING_UM (µm 단위)을 미터로 변환
        spacing_m = spacing_um * 1e-3
        xs = np.arange(cols, dtype=np.float32) * spacing_m
        ys = np.arange(rows, dtype=np.float32) * spacing_m
        X, Y = np.meshgrid(xs, ys)
        Z = grid
        # 유효점 마스크
        if sentinel is not None:
            mask = (Z != sentinel)
        else:
            mask = np.ones_like(Z, dtype=bool)
        x_flat = X[mask]
        y_flat = Y[mask]
        z_flat = Z[mask]
        pts = np.stack((x_flat, y_flat, z_flat), axis=-1)
    else:
        # x,y,z CSV 형태
        arr = data.astype(np.float32)
        if arr.shape[1] < 3:
            raise ValueError("CSV에 3개 이상의 열이 없습니다.")
        pts = arr[:, :3]

    n_pts = pts.shape[0]
    # PLY 헤더 작성
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_pts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode('utf-8')

    # 출력 디렉터리 생성
    out_dir = os.path.dirname(ply_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 바이너리 PLY 쓰기
    with open(ply_path, 'wb') as f:
        f.write(header)
        # little-endian float32
        f.write(pts.astype('<f4').tobytes())

    print(f"Converted grid CSV ({data.shape[0]}×{data.shape[1]}) → PLY ({n_pts} points): {ply_path}")

if __name__ == '__main__':
    csv_to_ply(CSV_FILE, PLY_FILE, SKIPROWS, DELIMITER, SPACING_UM, SENTINEL)

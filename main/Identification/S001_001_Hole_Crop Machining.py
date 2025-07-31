import os
import numpy as np
import open3d as o3d
import multiprocessing as mp
import pandas as pd

# ------------------- 사용자 지정 절대 경로 및 Z 범위 설정 -------------------
INPUT_DIR = r"D:\001_Calibration\000. Experimental\Data\250508\250508\Data\V001\002_CSV\T002\Ts001\001_Raw"
OUTPUT_DIR = r"D:\001_Calibration\000. Experimental\Data\250508\250508\Data\V001\002_CSV\T002\Ts001\002_Crop"
CROP_CSV_PATH = r"D:\001_Calibration\000. Experimental\Data\250508\250508\Crop_T002.csv"
MIN_Z = -5.0
MAX_Z = 5.0
# ---------------------------------------------------------------

def read_point_cloud(file_path):
    """CSV 또는 PLY 파일을 읽어 Nx3 float32 배열로 반환."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        pts = np.loadtxt(file_path, delimiter=',', skiprows=1)
    elif ext == '.ply':
        pcd = o3d.io.read_point_cloud(file_path)
        pts = np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return pts.astype(np.float32)


def save_as_ply(points: np.ndarray, ply_path: str):
    """Nx3 float32 포인트를 바이너리 PLY로 저장."""
    n_pts = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_pts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode('utf-8')
    body = points.astype(np.float32).tobytes()
    with open(ply_path, 'wb') as f:
        f.write(header)
        f.write(body)


def worker(args):
    """단일 feature에 대해 포인트 클라우드 읽기→crop 적용→저장 수행"""
    (input_path, output_dir, min_z, max_z,
     prefix, rep_str, feature_str,
     cx, cy, L, R, U, B) = args

    pts = read_point_cloud(input_path)
    if pts.size == 0:
        print(f"→ {prefix}_{feature_str}_{rep_str}: no points, skipping.")
        return
    # Crop 범위 계산
    x_min, x_max = cx + L, cx + R
    y_min, y_max = cy + B, cy + U

    mask = (
        (pts[:,0] >= x_min) & (pts[:,0] <= x_max) &
        (pts[:,1] >= y_min) & (pts[:,1] <= y_max) &
        (pts[:,2] >= min_z)  & (pts[:,2] <= max_z)
    )
    filtered = pts[mask]
    print(f"→ {prefix}_{feature_str}_{rep_str}: {pts.shape[0]}→{filtered.shape[0]} points")
    if filtered.size == 0:
        print(f"   No points remain after crop, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    out_name = f"{prefix}_{feature_str}_{rep_str}.ply"
    out_path = os.path.join(output_dir, out_name)
    save_as_ply(filtered, out_path)
    print(f"   Saved: {out_name}")


def main():
    # 경로 유효성 검사
    if not os.path.isdir(INPUT_DIR):
        print(f"Invalid input directory: {INPUT_DIR}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.isfile(CROP_CSV_PATH):
        print(f"Invalid Crop.csv path: {CROP_CSV_PATH}")
        return

    # Crop.csv 읽기 및 글로벌 인덱스 설정
    crop_df = pd.read_csv(CROP_CSV_PATH, header=0)
    # 첫 번째 컬럼명이 그룹명(P001 등)이 아니면 'Name'으로 통일
    if crop_df.columns[0] != 'Name':
        crop_df = crop_df.rename(columns={crop_df.columns[0]: 'Name'})
    # 원본 DataFrame 인덱스를 글로벌 feature 번호로 사용
    crop_df = crop_df.reset_index().rename(columns={'index': 'global_idx'})

    # 입력 파일 목록
    files = sorted(
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(('.csv', '.ply'))
    )
    if not files:
        print("No files found in input directory.")
        return

    # 워커 태스크 리스트 생성
    tasks = []
    for input_path in files:
        base = os.path.splitext(os.path.basename(input_path))[0]
        parts = base.split('_')
        if len(parts) != 2:
            print(f"Warning: Unexpected filename '{base}', skipping.")
            continue
        prefix, rep_str = parts
        # 해당 그룹의 모든 feature 행 추출
        grp_df = crop_df[crop_df['Name'] == prefix]
        if grp_df.empty:
            print(f"Warning: No group '{prefix}' in Crop.csv, skipping '{base}'.")
            continue
        # 각 행(row)에 대해 글로벌 feature 번호 사용
        for _, row in grp_df.iterrows():
            feature_str = f"F{int(row['global_idx'])+1:03d}"
            cx, cy = float(row['Cx']), float(row['Cy'])
            L, R, U, B = float(row['L']), float(row['R']), float(row['U']), float(row['B'])
            tasks.append((
                input_path, OUTPUT_DIR, MIN_Z, MAX_Z,
                prefix, rep_str, feature_str,
                cx, cy, L, R, U, B
            ))

    if not tasks:
        print("No tasks to process after matching crop rules.")
        return

    # 병렬 실행
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(worker, tasks)

if __name__ == "__main__":
    main()

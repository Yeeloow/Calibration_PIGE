import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

def convert_z_to_xyz(z_file_path, x_spacing=0.005, y_spacing=0.005):
    # z값만 있는 csv 불러오기
    z_data = pd.read_csv(z_file_path, header=None)
    
    n_rows, n_cols = z_data.shape
    
    # x, y 좌표 생성
    x_coords = np.arange(n_cols) * x_spacing
    y_coords = np.arange(n_rows) * y_spacing
    
    # meshgrid를 만들어 (x, y, z) 조합
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = z_data.values
    
    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    
    return pd.DataFrame(points, columns=['x', 'y', 'z'])

def main():
    root = tk.Tk()
    root.withdraw()
    
    # 입력 폴더 선택
    input_folder = filedialog.askdirectory(title="z 데이터가 들어있는 폴더를 선택하세요")
    if not input_folder:
        print("입력 폴더가 선택되지 않았습니다.")
        return
    
    # 출력 폴더 선택
    output_folder = filedialog.askdirectory(title="변환된 파일을 저장할 폴더를 선택하세요")
    if not output_folder:
        print("출력 폴더가 선택되지 않았습니다.")
        return
    
    # 폴더 내 파일 변환
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            df_xyz = convert_z_to_xyz(input_path, x_spacing=0.005, y_spacing=0.005)
            
            df_xyz.to_csv(output_path, index=False)
            print(f"변환 완료: {filename}")
    
    print("모든 파일 변환이 완료되었습니다.")

if __name__ == "__main__":
    main()

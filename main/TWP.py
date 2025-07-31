import math

def euler_xyz_to_direction(i_deg, j_deg, k_deg, base_vec=(0, 0, 1)):
    """
    i_deg, j_deg, k_deg: 오일러 각 (X, Y, Z 순서로 회전), 단위: 도(degree)
    base_vec: 기본 툴 벡터 (디폴트는 Z축을 바라봄)
    반환값: (x, y, z) - 최종 방향 벡터
    """
    i = math.radians(i_deg)
    j = math.radians(j_deg)
    k = math.radians(k_deg)

    # Rotation matrix around X
    Rx = [
        [1,          0,           0         ],
        [0,  math.cos(i), -math.sin(i)],
        [0,  math.sin(i),  math.cos(i)]
    ]
    # Rotation matrix around Y
    Ry = [
        [ math.cos(j), 0, math.sin(j)],
        [           0, 1,          0],
        [-math.sin(j), 0, math.cos(j)]
    ]
    # Rotation matrix around Z
    Rz = [
        [math.cos(k), -math.sin(k), 0],
        [math.sin(k),  math.cos(k), 0],
        [          0,            0, 1]
    ]

    # 행렬 곱: Rz * Ry * Rx (3×3 행렬간 곱 → 3×1 벡터)
    # Python에서는 numpy를 쓰면 편하지만, 여기서는 리스트로 직접 곱해볼 수 있음
    def matmul_3x3_3x1(M, v):
        return [
            M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
            M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
            M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2]
        ]

    def matmul_3x3_3x3(A, B):
        # A, B 둘 다 3x3 행렬이라 가정
        C = [[0]*3 for _ in range(3)]
        for r in range(3):
            for c in range(3):
                C[r][c] = (A[r][0]*B[0][c] +
                           A[r][1]*B[1][c] +
                           A[r][2]*B[2][c])
        return C

    R_temp = matmul_3x3_3x3(Rz, Ry)      # Rz * Ry
    R = matmul_3x3_3x3(R_temp, Rx)       # (Rz * Ry) * Rx

    # base_vec (tuple) → 리스트
    bx, by, bz = base_vec
    vtemp = matmul_3x3_3x1(R, [bx, by, bz])

    return (vtemp[0], vtemp[1], vtemp[2])

if __name__ == "__main__":
    # 예시
    i_test, j_test, k_test = (10, 20, 30)  # X=10°, Y=20°, Z=30°
    vx, vy, vz = euler_xyz_to_direction(i_test, j_test, k_test)
    print(f"i={i_test}, j={j_test}, k={k_test} -> direction=({vx:.4f}, {vy:.4f}, {vz:.4f})")

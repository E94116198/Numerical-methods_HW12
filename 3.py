import numpy as np
import pandas as pd

# 定義網格大小
Nr = 50  # r 的格點數
Ntheta = 60  # theta 的格點數

# 定義範圍
r_min, r_max = 0.5, 1.0
theta_min, theta_max = 0.0, np.pi / 3

# 建立網格
r = np.linspace(r_min, r_max, Nr)
theta = np.linspace(theta_min, theta_max, Ntheta)
dr = (r_max - r_min) / (Nr - 1)
dtheta = (theta_max - theta_min) / (Ntheta - 1)

# 初始化解矩陣
T = np.zeros((Nr, Ntheta))

# 邊界條件
T[0, :] = 50         # r = 0.5
T[-1, :] = 100       # r = 1.0
T[:, 0] = 0          # theta = 0
T[:, -1] = 0         # theta = pi/3

# 數值參數
tolerance = 1e-5
max_iter = 10000
error = 1.0
iteration = 0

# Gauss-Seidel 迭代法
while error > tolerance and iteration < max_iter:
    error = 0.0
    for i in range(1, Nr - 1):
        r_i = r[i]
        for j in range(1, Ntheta - 1):
            T_new = (
                (1 / dr**2) * (T[i+1, j] + T[i-1, j]) +
                (1 / (2 * r_i * dr)) * (T[i+1, j] - T[i-1, j]) +
                (1 / (r_i**2 * dtheta**2)) * (T[i, j+1] + T[i, j-1])
            ) / (
                2 / dr**2 + 2 / (r_i**2 * dtheta**2)
            )
            error = max(error, abs(T_new - T[i, j]))
            T[i, j] = T_new
    iteration += 1

print(f"Converged in {iteration} iterations, final max error = {error:.2e}")

# 轉成 DataFrame 以表格方式顯示
theta_deg = np.round(np.degrees(theta), 2)  # 弧度轉角度
r_str = [f"r={round(val, 3)}" for val in r]
theta_str = [f"{val}°" for val in theta_deg]
T_df = pd.DataFrame(T, index=r_str, columns=theta_str)

# 顯示部分或儲存完整表格
print(T_df.head(10))  # 顯示前 10 行
# T_df.to_csv("T_result.csv")  # 可選：儲存為 CSV

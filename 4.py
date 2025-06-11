import numpy as np
import pandas as pd

# 網格參數
L = 1.0
T = 1.0
dx = dt = 0.1
c = 1.0  # 波速，題目中為1

nx = int(L/dx) + 1  # x 節點數
nt = int(T/dt) + 1  # t 節點數
r = (c * dt / dx)**2

# 空間與時間座標
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# 初始條件與解矩陣
p = np.zeros((nt, nx))

# 初始位置 t = 0
p[0, :] = np.cos(2 * np.pi * x)

# 初始速度 ∂p/∂t(x, 0) → 用前向差分估計 p[1, i]
for i in range(1, nx - 1):
    p[1, i] = p[0, i] + dt * 2 * np.pi * np.sin(2 * np.pi * x[i]) + \
              0.5 * r * (p[0, i+1] - 2*p[0, i] + p[0, i-1])

# 邊界條件
p[:, 0] = 1
p[:, -1] = 2

# 時間迴圈（從第2步開始）
for n in range(1, nt - 1):
    for i in range(1, nx - 1):
        p[n+1, i] = 2*p[n, i] - p[n-1, i] + r * (p[n, i+1] - 2*p[n, i] + p[n, i-1])

print("p(x, t):");
# 輸出數值表格：t為列，x為欄
df = pd.DataFrame(p, index=[f"t={round(ti,1)}" for ti in t], columns=[f"x={round(xi,1)}" for xi in x])
print(df.round(4))

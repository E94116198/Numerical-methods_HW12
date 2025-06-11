import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 設定網格參數
pi = np.pi
h = k = 0.1 * pi
nx = int(pi / h) + 1       # x 方向點數 = 11
ny = int((pi / 2) / k) + 1 # y 方向點數 = 6

x = np.linspace(0, pi, nx)
y = np.linspace(0, pi / 2, ny)

# 初始化 u 與 f
u = np.zeros((nx, ny))
f = np.zeros((nx, ny))

# 設定邊界條件
for j in range(ny):
    u[0, j] = np.cos(y[j])        # u(0, y)
    u[-1, j] = -np.cos(y[j])      # u(pi, y)

for i in range(nx):
    u[i, 0] = np.cos(x[i])        # u(x, 0)
    u[i, -1] = 0                  # u(x, pi/2)

# 計算 f(x, y) = x * y
for i in range(nx):
    for j in range(ny):
        f[i, j] = x[i] * y[j]

# Jacobi 迭代法求解
u_new = u.copy()
alpha = h**2 / k**2
beta = 1 / (2 * (1 + alpha))
tolerance = 1e-6
max_iter = 10000

for it in range(max_iter):
    u_old = u_new.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u_new[i, j] = beta * (
                u_old[i + 1, j] + u_old[i - 1, j] +
                alpha * (u_old[i, j + 1] + u_old[i, j - 1]) -
                h**2 * f[i, j]
            )
    error = np.linalg.norm(u_new - u_old, ord=np.inf)
    if error < tolerance:
        print(f"✅ Converged after {it} iterations. Error = {error:.2e}")
        break
else:
    print("⚠️ Jacobi did not converge within the maximum iterations.")

# 使用 pandas 顯示表格
df = pd.DataFrame(u_new, index=[f"x={xi:.2f}" for xi in x], columns=[f"y={yj:.2f}" for yj in y])
print("數值表格：\n")
print(df)

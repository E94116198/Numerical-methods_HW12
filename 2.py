import numpy as np
from scipy.linalg import solve_banded
import pandas as pd

# Parameters
dr = 0.1
dt = 0.5
K = 0.1
r_min, r_max = 0.5, 1.0
t_max = 10.0

# Grid setup
r = np.arange(r_min, r_max + dr, dr)
t = np.arange(0, t_max + dt, dt)
N_r = len(r)
N_t = len(t)

# Initial condition
T = np.zeros((N_t, N_r))
T[0, :] = 200 * (r - 0.5)

def boundary_r1(tn):
    return 100 + 40 * tn

# --- Forward Difference Method ---
def solve_forward():
    T_fd = T.copy()
    alpha = dt / (4 * K * dr**2)

    for n in range(0, N_t - 1):
        for j in range(1, N_r - 1):
            rj = r[j]
            T_fd[n + 1, j] = T_fd[n, j] + alpha * (
                (T_fd[n, j+1] - 2 * T_fd[n, j] + T_fd[n, j-1]) +
                (dr / rj) * (T_fd[n, j+1] - T_fd[n, j-1]) / 2
            )

        T_fd[n + 1, -1] = boundary_r1(t[n + 1])
        T_fd[n + 1, 0] = T_fd[n + 1, 1] / (1 + 3 * dr)

    return T_fd

# --- Backward Difference Method ---
def solve_backward():
    T_bd = T.copy()
    alpha = dt / (4 * K * dr**2)

    for n in range(0, N_t - 1):
        A = np.zeros((3, N_r))
        b = np.zeros(N_r)

        for j in range(1, N_r - 1):
            rj = r[j]
            A[0, j] = -alpha * (1 - dr / (2 * rj))
            A[1, j] = 1 + 2 * alpha
            A[2, j] = -alpha * (1 + dr / (2 * rj))
            b[j] = T_bd[n, j]

        A[1, -1] = 1
        b[-1] = boundary_r1(t[n + 1])

        A[1, 0] = 1 + 3 * dr
        A[2, 0] = -1
        b[0] = 0

        T_bd[n + 1, :] = solve_banded((1, 1), A, b)

    return T_bd

# --- Crank-Nicolson Method ---
def solve_crank_nicolson():
    T_cn = T.copy()
    alpha = dt / (8 * K * dr**2)

    for n in range(0, N_t - 1):
        A = np.zeros((3, N_r))
        b = np.zeros(N_r)

        for j in range(1, N_r - 1):
            rj = r[j]
            A[0, j] = -alpha * (1 - dr / (2 * rj))
            A[1, j] = 1 + 2 * alpha
            A[2, j] = -alpha * (1 + dr / (2 * rj))

            b[j] = alpha * (1 + dr / (2 * rj)) * T_cn[n, j+1] + \
                   (1 - 2 * alpha) * T_cn[n, j] + \
                   alpha * (1 - dr / (2 * rj)) * T_cn[n, j-1]

        A[1, -1] = 1
        b[-1] = boundary_r1(t[n + 1])

        A[1, 0] = 1 + 3 * dr
        A[2, 0] = -1
        b[0] = 0

        T_cn[n + 1, :] = solve_banded((1, 1), A, b)

    return T_cn

# Solve all methods
T_forward = solve_forward()
T_backward = solve_backward()
T_crank = solve_crank_nicolson()

# Display as tables
print("(a) Forward-Difference Method:")
df_forward = pd.DataFrame(T_forward, columns=[f"r={rj:.2f}" for rj in r], index=[f"t={tn:.1f}" for tn in t])
print(df_forward)

print("\n(b) Backward-Difference Method:")
df_backward = pd.DataFrame(T_backward, columns=[f"r={rj:.2f}" for rj in r], index=[f"t={tn:.1f}" for tn in t])
print(df_backward)

print("\n(c) Crank-Nicolson Method:")
df_crank = pd.DataFrame(T_crank, columns=[f"r={rj:.2f}" for rj in r], index=[f"t={tn:.1f}" for tn in t])
print(df_crank)

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:29:43 2023

@author: Ghadeer SHAABAN
Ph.D. student at GIPSA-Lab
University Grenoble Alpes, France


Code for the paper:
S. Gillijns and B. De Moor, “Unbiased minimum-variance input and state 
estimation for linear discrete-time systems,” Automatica, vol. 43, no. 1,
pp. 111–116, Jan. 2007, doi: 10.1016/j.automatica.2006.08.002.
"""


import numpy as np
import math
import matplotlib.pyplot as plt


A = np.eye(3)
G = np.eye(3)
C = np.array(
    [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0]]
)

N = 10000

y_sigma = 0.1
x_sigma = 0
measurement_noises = np.random.normal(0, y_sigma, (N, 6))
process_noises = np.random.normal(0, x_sigma, (N, 3))

x_model = np.zeros((N, 3))
d_true = np.zeros((N, 3))
y_true = np.zeros((N, 6))
x_true = np.zeros((N, 3))
y_meas = np.zeros((N, 6))


for i in range(N):
    d_true[i, 0] = 2.0 * np.cos(1.2 * i * 0.01)
    d_true[i, 1] = 1.5 * np.cos(0.8 * i * 0.01)
    d_true[i, 2] = 0.5 * np.cos(0.7 * i * 0.01)


for i in range(1, N):
    x_model[i, :] = A @ x_model[i - 1, :] + G @ d_true[i - 1, :]

x_true = x_model + process_noises

for i in range(N):
    y_true[i, :] = C @ x_true[i, :]


y_meas = y_true + measurement_noises


Q = x_sigma**2 * np.eye(3)
R = y_sigma**2 * np.eye(6)


def UMV(X, P, Y):
    Xp = A @ X
    Pp = A @ P @ A.T + Q
    R_tilde = C @ Pp @ C.T + R
    R_tilde_inv = np.linalg.inv(R_tilde)
    F = C @ G
    M = np.linalg.inv(F.T @ R_tilde_inv @ F) @ F.T @ R_tilde_inv
    d = M @ (Y - C @ Xp)
    P_d = np.linalg.inv(F.T @ R_tilde_inv @ F)
    X_star = Xp + G @ d
    P_star = (np.eye(3) - G @ M @ C) @ Pp @ (
        np.eye(3) - G @ M @ C
    ).T + G @ M @ R @ M.T @ G.T
    S_star = -G @ M @ R
    V_star = P_star @ C.T + S_star
    R_star_tild = C @ P_star @ C.T + R + C @ S_star + S_star.T @ C.T
    R_star_tild = (np.eye(6) - C @ G @ M) @ R_tilde @ (np.eye(6) - C @ G @ M).T
    K = np.linalg.lstsq(R_star_tild.T, (P_star @ C.T + S_star).T, rcond=1e-6)[
        0
    ].T
    X = X_star + K @ (Y - C @ X_star)
    P = P_star - K @ (P_star @ C.T + S_star).T
    return X, P, d, P_d


X_hat = np.ones((3, 1))
P_Xhat = np.eye(3)
x_estimated = np.zeros((N, 3))
P_x = np.zeros((3, 3, N))
x_estimated[0, :] = np.copy(X_hat.reshape(-1))
P_x[:, :, 0] = np.copy(P_Xhat)

d_estimated = np.zeros((N, 3))
P_d = np.zeros((3, 3, N))


for i in range(1, N):
    [X_hat, P_Xhat, d_hat, P_dhat] = UMV(
        X_hat, P_Xhat, y_meas[i, :].reshape(-1, 1)
    )
    x_estimated[i, :] = np.copy(X_hat.reshape(-1))
    P_x[:, :, i] = np.copy(P_Xhat)
    d_estimated[i - 1, :] = np.copy(d_hat.reshape(-1))
    P_d[:, :, i - 1] = np.copy(P_dhat)


start_point = 100
end_point = N - 100
state_estimation_RMSE = np.sqrt(
    np.mean(
        np.square(
            x_estimated[start_point:end_point, :]
            - x_true[start_point:end_point, :]
        )
    )
)

print("state_estimation_RMSE:", state_estimation_RMSE)


for i in range(3):
    plt.figure()
    plt.plot(x_estimated[start_point:end_point, i])
    plt.plot(x_true[start_point:end_point, i])
    plt.legend(["UMV", "true"])
    plt.title("state estimation " + str(i))
    plt.show


error_x = x_true - x_estimated
for i in range(3):
    plt.figure()
    plt.plot(error_x[start_point:end_point, i])
    plt.plot(3 * np.sqrt(P_x[i, i, start_point:end_point]))
    plt.plot(-3 * np.sqrt(P_x[i, i, start_point:end_point]))
    plt.legend(
        [
            r"$x_{%d}-\hat{x}_{%d}$ " % (i, i),
            r"$3\sigma^x_{%d}$" % (i),
            r"$-3\sigma^x_{%d}$" % (i),
        ]
    )
    plt.title("state estimation error" + str(i))
    plt.show


error_d = d_true - d_estimated
for i in range(3):
    plt.figure()
    plt.plot(error_d[start_point:end_point, i])
    plt.plot(3 * np.sqrt(P_d[i, i, start_point:end_point]))
    plt.plot(-3 * np.sqrt(P_d[i, i, start_point:end_point]))
    plt.legend(
        [
            r"$d_{%d}-\hat{d}_{%d}$ " % (i, i),
            r"$3\sigma^d_{%d}$" % (i),
            r"$-3\sigma^d_{%d}$" % (i),
        ]
    )
    plt.title("unknown input estimation error" + str(i))
    plt.show

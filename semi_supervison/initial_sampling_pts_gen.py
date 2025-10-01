import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
from time import time

def dlqr(A, B, Q, R):
    """Solve the discrete-time LQR controller."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K, P


def mpc_fun(Ad, Bd, Q, R, x0, xref, N):
    """MPC optimization using scipy.optimize with box constraints"""
    u_min, u_max = -0.6, 0.6
    K, P = dlqr(Ad, Bd, Q, R)

    def cost_fn(u):
        u = np.array(u).reshape(-1, 1)
        x = x0.reshape(-1, 1)
        cost = 0.0
        for i in range(N):
            x = Ad @ x + Bd * u[i]
            cost += ((x - xref).T @ Q @ (x - xref))[0, 0] + (u[i].T @ R @ u[i])
        # terminal cost
        cost += ((x - xref).T @ P @ (x - xref))[0, 0]
        return cost

    # Box bounds (no nonlinear constraints → much more stable!)
    bounds = [(u_min, u_max)] * N

    u0 = np.zeros(N)
    res = minimize(cost_fn, u0, bounds=bounds, method='SLSQP')

    if not res.success:
        raise RuntimeError("MPC optimization failed: " + res.message)

    return res.x



if __name__ == "__main__":

    # === Simulation ===
    # Continuous-time system
    A = np.array([[0, 1],
                [0, -1.7873]])
    B = np.array([[0],
                [-1.7382]])
    C = np.array([[1, 0]])
    D = np.array([[0]])

    # Discretize system
    dt = 0.1
    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), dt)

    # MPC settings
    N = 50
    Q = np.diag([20, 10])
    R = np.array([[0.1]])

    # Initial state
    x0 = np.array([2.5, 1])
    xref = np.array([1, 0]).reshape(-1, 1)
    Tsim = 70

    # Storage
    xHist = np.zeros((2, Tsim + 1))
    uHist = np.zeros(Tsim)
    xHist[:, 0] = x0

    t_start = time()

    # Simulation loop
    for k in range(Tsim):
        u_seq = mpc_fun(Ad, Bd, Q, R, xHist[:, k], xref, N)
        uHist[k] = u_seq[0]  # apply first input
        xHist[:, k + 1] = Ad @ xHist[:, k] + Bd.flatten() * uHist[k]

    t_end = time()
    print(f"Total simulation time: {t_end - t_start:.2f} seconds")

    # === Plotting ===
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(range(Tsim + 1), xHist[0, :], 'b', linewidth=2, label='x1')
    plt.plot(range(Tsim + 1), xHist[1, :], 'r', linewidth=2, label='x2')
    plt.axhline(xref[0], linestyle='--', color='b')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('States')
    plt.title('MPC State Trajectory')

    plt.subplot(2, 1, 2)
    plt.step(range(Tsim), uHist, 'k', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Control Input')
    plt.title('MPC Control Input')

    plt.tight_layout()
    plt.show()

import numpy as np
from scipy.optimize import minimize
import concurrent.futures
from tqdm import tqdm
from scipy.signal import cont2discrete

'''
def dlqr(A, B, Q, R):
    from scipy.linalg import solve_discrete_are
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K, P

def mpc_fun(Ad, Bd, Q, R, x0, xref, N):
    u_min, u_max = -0.6, 0.6
    K, P = dlqr(Ad, Bd, Q, R)

    def cost_fn(u):
        u = np.array(u).reshape(-1, 1)
        x = x0.reshape(-1, 1)
        cost = 0.0
        for i in range(N):
            x = Ad @ x + Bd * u[i]
            cost += ((x - xref).T @ Q @ (x - xref))[0, 0] + (u[i].T @ R @ u[i])
        cost += ((x - xref).T @ P @ (x - xref))[0, 0]
        return cost

    bounds = [(u_min, u_max)] * N
    u0 = np.zeros(N)
    res = minimize(cost_fn, u0, bounds=bounds, method='SLSQP')
    if not res.success:
        raise RuntimeError("MPC optimization failed: " + res.message)
    return res.x

def run_mpc_trajectory(idx, Ad, Bd, Q, R, x0, xref, N):
    x = x0.copy()
    xset = []
    uset = []
    xset.append(x)
    for t in range(300):
        u = mpc_fun(Ad, Bd, Q, R, x, xref, N)
        x = Ad @ x + Bd * u[0]
        xset.append(x)
        uset.append(u[0])

    return idx, xset, uset

# Example setup

# Define continuous-time system
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


# generate xr between [1, 2] and [0,0]
xref = [np.random.uniform(1, 2, (N, 1)) for _ in range(10)]
x0_list = [np.random.randn(2, 1) for _ in range(10)]

results = []
with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(run_mpc_trajectory, i, Ad, Bd, Q, R, x0_list[i], xref[i][0], N)
        for i in range(len(x0_list))
    ]
    for f in tqdm(concurrent.futures.as_completed(futures), total=len(x0_list)):
        results.append(f.result())

print(results)'''


import numpy as np
import matplotlib.pyplot as plt

# Dynamics
def f(x, u, dt=0.1):
    """Double integrator dynamics"""
    x1, x2 = x
    x_next = np.array([
        x1 + dt * x2,
        x2 + dt * u
    ])
    return x_next

# Cost function
def cost_fn(x, u, x_ref=np.array([0, 0])):
    Q = np.diag([1.0, 0.1])
    R = 0.1
    dx = x - x_ref
    return dx.T @ Q @ dx + R * (u ** 2)


def cem_mpc_step(x0, f, cost_fn, N=20, K=256, Ke=20, 
                 u_min=-1.0, u_max=1.0, num_iters=5, sigma=0.8, alpha=0.9):
    m = 1  # control dimension
    mu = np.zeros((N, m))
    sigma_arr = np.ones((N, m)) * sigma**2

    for _ in range(num_iters):
        # Sample K control sequences: (K, N, m)
        std = np.sqrt(sigma_arr)[None, :, :]  # shape (1, N, m)
        mean = mu[None, :, :]                 # shape (1, N, m)

        U = np.random.randn(K, N, m) * std + mean
        U = np.clip(U, u_min, u_max)

        # Evaluate cost for each trajectory
        costs = np.zeros(K)
        for i in range(K):
            x = np.copy(x0)
            cost = 0
            for t in range(N):
                u = U[i, t, 0]
                cost += cost_fn(x, u)
                x = f(x, u)
            costs[i] = cost

        # Select top Ke elites
        elites_idx = np.argsort(costs)[:Ke]
        elites = U[elites_idx]  # shape (Ke, N, m)

        # Update distribution
        new_mu = np.mean(elites, axis=0)  # (N, m)
        new_sigma = np.var(elites, axis=0)  # (N, m)

        mu = alpha * new_mu + (1 - alpha) * mu
        sigma_arr = alpha * new_sigma + (1 - alpha) * sigma_arr

    return mu[0, 0], mu, sigma_arr

# Simulation setup
T = 50  # total time steps
x = np.array([2.0, 0.0])  # initial position and velocity
trajectory = [x]
controls = []

for t in range(T):
    u, mu, sigma = cem_mpc_step(x, f, cost_fn)
    controls.append(u)
    x = f(x, u)
    trajectory.append(x)

trajectory = np.array(trajectory)


plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(trajectory[:,0], label='Position')
plt.plot(trajectory[:,1], label='Velocity')
plt.axhline(0, color='k', linestyle='--')
plt.legend()
plt.title('State Trajectory')

plt.subplot(2,1,2)
plt.plot(controls, label='Control Input')
plt.axhline(0, color='k', linestyle='--')
plt.legend()
plt.xlabel('Time step')
plt.tight_layout()
plt.show()

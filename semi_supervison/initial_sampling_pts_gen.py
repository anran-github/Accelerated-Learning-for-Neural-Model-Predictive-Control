import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
from time import time
import json

def dlqr(A, B, Q, R):
    """Solve the discrete-time LQR controller."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K, P


def mpc_fun(Ad, Bd, Q, R, x0, xref, N, u_min=-0.6, u_max=0.6):
    """
    Simplified finite-horizon MPC (1-step receding horizon)
    Parameters
    ----------
    Ad, Bd : ndarray
        Discrete-time system matrices
    Q, R : ndarray
        State and input cost matrices
    x0 : ndarray
        Current state
    xref : ndarray
        Desired reference state
    N : int
        Prediction horizon
    u_min, u_max : float
        Input saturation limits

    Returns
    -------
    u_seq : ndarray
        Optimal input sequence [u(0), ..., u(N-1)]
    """
    # Compute LQR for terminal cost and constraint rollout
    Kcl, P = dlqr(Ad, Bd, Q, R)

    nx = Ad.shape[0]
    nu = Bd.shape[1]

    # Flattened optimization variable: u = [u0, u1, ..., uN-1]
    def cost_fn(u):
        u = np.array(u).reshape(N, nu)
        x = x0.reshape(nx, 1)
        cost = 0.0
        for k in range(N):
            u_k = u[k].reshape(nu, 1)
            x = Ad @ x + Bd @ u_k
            cost += float((x - xref).T @ Q @ (x - xref) + u_k.T @ R @ u_k)
        # Terminal cost
        cost += float((x - xref).T @ P @ (x - xref))
        return cost

    # Terminal feasibility constraints (20-step LQR rollout)
    def terminal_constraint(u):
        """Ensure terminal feasibility via 20-step LQR rollout"""
        u = np.array(u).reshape(N, nu)
        x = x0.reshape(nx, 1)
        for k in range(N):
            u_k = u[k].reshape(nu, 1)
            x = Ad @ x + Bd @ u_k

        # Start from x_N and simulate 20-step LQR rollout
        x_terminal = x.copy()
        for t in range(20):
            u_lqr = -Kcl @ (x_terminal - xref)
            # if any constraint is violated → constraint returns positive value
            if np.any(u_lqr < u_min) or np.any(u_lqr > u_max):
                return 1.0  # violation
            x_terminal = Ad @ x_terminal + Bd @ u_lqr
        return 0.0  # feasible

    # Bounds for each u_k
    bounds = [(u_min, u_max)] * (N * nu)

    # Nonlinear constraint wrapper
    nonlinear_constraints = {
        "type": "ineq",
        "fun": lambda u: -terminal_constraint(u)  # must be ≥0 → feasible means ≥0
    }

    u0 = np.zeros((N * nu,))
    res = minimize(
        cost_fn,
        u0,
        method="SLSQP",
        bounds=bounds,
        constraints=[nonlinear_constraints],
        options={"disp": False, "maxiter": 200},
    )

    if not res.success:
        print("Warning: MPC optimization did not fully converge:", res.message)

    return res.x

def mpc_fun_ori(Ad, Bd, Q, R, x0, xref, N, u_min=-0.6, u_max=0.6):
    """MPC optimization using scipy.optimize with box constraints"""
    # u_min, u_max = -0.6, 0.6
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

def sampling_boundary(sampling_num=16,x1_min=0.5, x1_max=2.5, x2_min=-1.0, x2_max=1.0):
    '''
    Sample points on the boundary of the state space

    **It is presumed that sampling_num is a multiple of 4**
    '''
    num_boundary_per_side = sampling_num // 4

    if sampling_num < 4:
        # return diagonals
        return np.array([[x1_min, x2_min],
                         [x1_max, x2_max]])
        
    x = np.linspace(x1_min, x1_max, 1 + num_boundary_per_side)
    y = np.linspace(x2_min, x2_max, 1 + num_boundary_per_side)

    # Get boundary points
    bottom = np.vstack((x, -1 * np.ones(1 + num_boundary_per_side)))
    top    = np.vstack((x,  1 * np.ones(1 + num_boundary_per_side)))
    left   = np.vstack((x1_min * np.ones(-1 + num_boundary_per_side), y[1:int(num_boundary_per_side)]))   # exclude corners
    right  = np.vstack((x1_max * np.ones(-1 + num_boundary_per_side), y[1:int(num_boundary_per_side)]))

    # Combine all and transpose to get shape (16, 2)
    X_boundary = np.hstack((bottom, top, left, right)).T

    return X_boundary


def sampling_center(sampling_num=16, xr=1.5):
    '''
    Sample points around the center (xr, 0) with a decaying distance.
    **It is presumed that sampling_num is a multiple of 4**
    '''
    if sampling_num < 4:
        # return diagonals
        return np.array([[xr - 0.5, -0.5],
                         [xr + 0.5,  0.5]])
    
    center_pts = np.array([xr, 0])
    distance_x1 = 0.4
    distance_x2 = 0.4
    sampling_pts = []
    
    num_pts_per_layer = 4
    decaying_factor = int(1 + sampling_num // num_pts_per_layer)
    for i in range(1,decaying_factor):
        distance_x1 /= 1.25
        distance_x2 /= 1.25
        angle = (i%2)*np.pi / 4 # either 45 or 0 degrees
        for j in range(num_pts_per_layer):
            angle_j = angle + j*np.pi / 2
            sampling_pts.append([center_pts[0] + distance_x1 * np.cos(angle_j),
                                center_pts[1] + distance_x2 * np.sin(angle_j)])
            
    return np.array(sampling_pts)
            

def sampling_data_generation(mode='uniform',sampling_num=16, xr=1.5):
    '''
    Generate initial sampling data for semi-supervised learning
    mode: 'uniform' or 'dense_boundary', 'dense_center'
    sampling_num: number of sampling points
    xr: reference state 2x1
    '''
    x1_min, x1_max = 0.5, 2.5
    x2_min, x2_max = -1.0, 1.0

    if mode == 'uniform':
        x1_samples = np.linspace(x1_min, x1_max, int(np.sqrt(sampling_num)))
        x2_samples = np.linspace(x2_min, x2_max, int(np.sqrt(sampling_num)))
        X1, X2 = np.meshgrid(x1_samples, x2_samples)
        initial_states = np.vstack([X1.ravel(), X2.ravel()]).T
    elif mode == 'dense_boundary':
        boundary_num = int(sampling_num * 0.8)
        center_num = sampling_num - boundary_num

        # Generate Boundary points
        X_boundary = sampling_boundary(sampling_num=boundary_num, x1_min=x1_min, x1_max=x1_max, x2_min=x2_min, x2_max=x2_max)
        
        # Generate Center points
        X_center = sampling_center(sampling_num=center_num, xr=xr)

        initial_states = np.vstack((X_boundary, X_center))
        
        
    elif mode == 'dense_center':
        center_num = int(sampling_num * 0.8)
        boundary_num = sampling_num - center_num

        # Boundary points
        X_boundary = sampling_boundary(sampling_num=boundary_num, x1_min=x1_min, x1_max=x1_max, x2_min=x2_min, x2_max=x2_max)

        # Combine points
        sampling_pts = sampling_center(sampling_num=center_num, xr=xr)
        X_center = np.array(sampling_pts)
        initial_states = np.vstack((X_boundary, X_center))


    
    # === Plotting ===
    plt.figure(figsize=(6, 6))
    plt.scatter(initial_states[:, 0], initial_states[:, 1], c='b', label='Initial States')
    plt.xlim(x1_min - 0.25, x1_max + 0.25)
    plt.ylim(x2_min - 0.25, x2_max + 0.5)
    plt.scatter(xr, 0, c='r', marker='*', s=200, label='Reference State')
    plt.xlabel(r'State $x_1$',fontsize=17)
    plt.ylabel(r'State $x_2$',fontsize=17)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    # plt.title(f'Initial Sampling States ({mode})',fontsize=16)
    plt.grid()
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(f'semi_supervison/DroneZ_MPC_weights/initial_sampling_{mode}_xr{xr}.png', dpi=300)
    plt.show()
    plt.close()


    # stack xr to each state
    initial_states = np.hstack((initial_states, xr * np.ones((initial_states.shape[0], 1))))
    # add equalibrium point
    initial_states = np.vstack((initial_states, np.array([[xr, 0.0, xr]])))
    return initial_states


'''

if __name__ == "__main__":

#     xr_list = np.linspace(1.0, 2.0, 11).tolist()
#     total_sampling_pts = []
#     for xr in xr_list:
#         init_pts = sampling_data_generation(mode='dense_center', sampling_num=10, xr=xr)
#         total_sampling_pts.append(init_pts)

#     total_sampling_pts = np.vstack(total_sampling_pts)
    # send to solve MPC problem




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
'''
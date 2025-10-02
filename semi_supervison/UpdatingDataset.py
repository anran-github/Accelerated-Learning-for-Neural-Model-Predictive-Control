import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import cont2discrete
from tqdm import tqdm



from initial_sampling_pts_gen import mpc_fun, sampling_data_generation


class UpdatingDataset(Dataset):
    '''
    ### Generate updating dataset for training.\n
    parameters:
        **mode**: 'uniform' or 'dense_boundary', 'dense_center'
        **sampling_num_per_xr**: number of sampling points per reference state xr
    '''
    def __init__(self,mode, sampling_num_per_xr):
        super(UpdatingDataset, self).__init__()

        self.mode = mode


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

        # Generate dataset
        # Generate total dataset first
        # Format: {[x1,x2,xr, u1,u2,...uN],...}
        x1_min, x1_max = 0.5, 2.5
        x2_min, x2_max = -1,  1
        xr_min, xr_max = 1.0, 2.0

        X1 = np.linspace(x1_min, x1_max, 101)
        X2 = np.linspace(x2_min, x2_max, 101)
        XR = np.linspace(xr_min, xr_max, 11)

        # combine to get all grid points
        self.data = np.meshgrid(X1, X2, XR)
        self.data = np.vstack(map(np.ravel, self.data)).T
        # add space for u_seq
        self.data = np.hstack((self.data, np.zeros((self.data.shape[0], N))))


        for xr in XR.tolist():
            init_pts = sampling_data_generation(mode=self.mode, sampling_num=sampling_num_per_xr, xr=xr)
            pts_process_bar = tqdm(init_pts)
            for state in pts_process_bar:
                x0 = state[:2]
                u_seq = mpc_fun(Ad, Bd, Q, R, x0, np.array([xr, 0]).reshape(-1, 1), N)
                
                # insert to dataset
                distance = np.linalg.norm(self.data[:,:3] - np.hstack((x0, xr)), axis=1)
                nearest_idx = np.argmin(distance)
                self.data[nearest_idx] = np.hstack((x0, xr, u_seq))
                pts_process_bar.set_description(f"Filling data for xr={xr:.2f}, current state=({x0[0]:.2f},{x0[1]:.2f}), nearest distance={distance[nearest_idx]:.4f}")

        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, u_seq = self.data[idx]
        return torch.tensor(state, dtype=torch.float32), torch.tensor(u_seq, dtype=torch.float32)
    


if __name__ == "__main__":
    dataset = UpdatingDataset(mode='dense_center', sampling_num_per_xr=10)
    print(len(dataset))
    print(dataset[0])
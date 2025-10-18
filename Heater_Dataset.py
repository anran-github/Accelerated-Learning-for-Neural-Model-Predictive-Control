import torch
import numpy as np
from torch.utils.data import Dataset
import control
from tqdm import tqdm
import os
import json
from torch.utils.data import Sampler
import math
import random
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.signal import cont2discrete, place_poles

from Objective_Formulations_mpc import ObjectiveFormulation
from semi_supervison.initial_sampling_pts_gen import mpc_fun




# Define continuous-time system
A = np.array([[0, -0.08091/159], [1, -15.34/159]])
B = np.array([[0.06545/159],[-0.72/159]])
C = np.array([[0,1]])
D = 0

dt = 5 # sampling time
ss = control.ss(A, B, C, D)
Gd = control.c2d(ss,dt, method='zoh') # 'zoh':assuming control input be constant between sampling intervals.

Ad, Bd, Cd, Dd = Gd.A, Gd.B, Gd.C, Gd.D

# MPC settings
N = 30;              # Prediction horizon
Q = torch.tensor([[0.1,0],[0.,100.]])
R = torch.tensor([0.001])
      

# Generate dataset
# Generate total dataset first
# Format: {[x1,x2,xr, u1,u2,...uN],...}
x1_min, x1_max = 0., 8.
x2_min, x2_max = 0.,  80.
xr_min, xr_max = 5.0, 70.0
u_min, u_max = 0., 100



class MPCDataset(Dataset):
    '''
    Generate Random NN Input Data for NN Training.\n

    '''
    def __init__(self, ):
        super(MPCDataset, self).__init__()

        # Generate Updating Dataset
        # Format: {[x1,x2,xr, u1,u2,...uN],...}
        # Generate Updating Dataset
        # Format: {[x1,x2,xr, u1,u2,...uN],...}
        X1 = np.linspace(x1_min, x1_max, 21)
        X2 = np.linspace(x2_min, x2_max, 101)
        XR = np.linspace(xr_min, xr_max, 101)

        # combine to get all grid points
        self.data = np.meshgrid(X1, X2, XR)
        self.data = np.vstack(list(map(np.ravel, self.data))).T



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        nn_input = self.data[idx,:3]
        return torch.tensor(nn_input, dtype=torch.float32)
    




class Data_Heater_Collected():
    '''
    Load Heater Dataset Collected from Solving MPC prolem
    '''
    def __init__(self,filename) -> None:

        # Load the CSV dataset
        df = pd.read_csv(filename, header=None)

        # Format:
        # x1 x2 r u1, u2, ..., uN

        # Assuming the first row is the header, adjust if needed
        self.data = df.values  # Transpose to have shape (2, n)
        
        # self.data_r = np.tile(xr,(len(self.data_u),1))
        print('----------DATA SUMMARY------------')
        print(f'There are {self.data.shape[0]} raw data points')

        # combine [x1,x2] and r as a single input:
        self.input_data = self.data[:,:3]  # first three columns: x1, x2, r

        self.label_data = self.data[:,3:]  # remaining columns: u1, u2, ..., uN
        # normalize the label data to [0,1]
        self.label_data = (self.label_data - u_min) / (u_max - u_min)

    def return_data(self,):
            return self.data
    
    def sample_data(self,sample_num_eachR, sample_mode='uniform'):
        # OUR PURPOSE: use limited samples to train the NN

        unique_r = np.unique(self.input_data[:,2])
        
        # sample_num_eachR = int(sample_num/len(unique_r))
        if sample_mode == 'uniform':
            idx = []
            for r in unique_r[:len(unique_r)-1]:
                idx_start = np.where(self.input_data[:,2]==r)[0]
                
                sampling_idx = np.random.uniform(idx_start[0], idx_start[-1], sample_num_eachR).astype(int)
                idx.extend(sampling_idx)

            idx = np.array(idx)
            sampled_input = self.input_data[idx,:]
            sampled_label = self.label_data[idx,:]

            return sampled_input, sampled_label
        



class UpdatingDataset(Dataset):
    '''
    ### Generate updating dataset for training.\n
    parameters:
        **mode**: 'uniform' or 'dense_boundary', 'dense_center'
        **sampling_num_per_xr**: number of sampling points per reference state xr
    '''
    def __init__(self, mode, sampling_num_per_xr,opt_dataset_path=None):
        super(UpdatingDataset, self).__init__()

        self.mode = mode


        # Generate Updating Dataset
        # Format: {[x1,x2,xr, u1,u2,...uN],...}
        X1 = np.linspace(x1_min, x1_max, 21)
        X2 = np.linspace(x2_min, x2_max, 101)
        XR = np.linspace(xr_min, xr_max, 101)

        # combine to get all grid points
        self.data = np.meshgrid(X1, X2, XR)
        self.data = np.vstack(list(map(np.ravel, self.data))).T
        # add space for u_seq
        self.data = np.hstack((self.data, np.zeros((self.data.shape[0], N))))


        # load optimal dataset
        opt_dataset = Data_Heater_Collected(opt_dataset_path)
        opt_input, opt_label = opt_dataset.sample_data(sampling_num_per_xr, self.mode)


        t_start = time.time()
        os.makedirs('semi_supervison/dataset', exist_ok=True)
        
        
        # solve mpc problem get optimal u*
        self.TruthData_Mask = []


        pts_process_bar = tqdm(opt_input)
        for idx, state in enumerate(pts_process_bar):
            # compute distance and find nearest point in self.data
            x0 = state[:2]
            xr = state[2]
            # insert to dataset
            distance = np.linalg.norm(self.data[:,:3] - np.hstack((x0, xr)), axis=1)

            if distance.min() > 0.05:
                continue

            nearest_idx = np.argmin(distance)
            self.data[nearest_idx] = np.hstack((x0, xr, opt_label[idx]))  # use optimal label

            # save the index
            self.TruthData_Mask.append(nearest_idx)


            pts_process_bar.set_description(f"Filling data for xr={xr:.2f}, current state=({x0[0]:.2f},{x0[1]:.2f}), nearest distance={distance[nearest_idx]:.4f}")

        t_end = time.time()
        print(f"Dataset generation completed in {t_end - t_start:.2f} seconds.")
        print(f"Total truth data points added: {len(self.TruthData_Mask)}")



        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        nn_input, u_seq = self.data[idx,:3], self.data[idx,3:]
        return torch.tensor(nn_input, dtype=torch.float32), torch.tensor(u_seq, dtype=torch.float32)
    
    def update_with_model(self, model, device):
        '''
        Update the dataset with model predictions.

        **Only update non-optimal section**
        '''
        model.eval()

        with torch.no_grad():

                nn_input = torch.tensor(self.data[:,:3], dtype=torch.float32).to(device)
                u_seq_pred = model(nn_input).cpu().numpy()
                # only update non-optimal section
                mask = np.ones(len(self.data), dtype=bool)
                mask[self.TruthData_Mask] = False


                # self.data[mask, 3:] = u_seq_pred[mask]

                self.data[mask, 3:] = 0.5*self.data[mask, 3:]+0.5*u_seq_pred[mask] # smooth the update
                print(f"Updated {np.sum(mask)} samples with model predictions.")
                
            

    def test_performance_index(self,model,device,model_path = None):
        """
        Test the performance index of the model given a reference trajectory xr.
        :param model: The trained neural network model.
        :param device: The device to run the model on (CPU or GPU).
        :return: The performance index calculated as the mean of the sum of norms
        """
        
        # ============ Model Argument Parsing ============

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load the trained model
        if model_path is not None:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print('----------Added previous weights: {}------------'.format(model_path))
        # model = P_Net(output_size=5).to(device)
        # trained_model_path = 'mathmatical_simulation/weights/test_model_weight_0.05_0.9_0.05.pth'
        # model.load_state_dict(torch.load(trained_model_path))
        # print('----------added previous weights: {}------------'.format(trained_model_path))

        model.eval()


        # =========== Input Data Preparation ============
        pts_count = 10000
        # Generate random initial states between -5 and 5:
        torch.manual_seed(42)  # For reproducibility
        
        # random select init state from dataset
        random_indices = np.random.choice(len(self.data), size=pts_count, replace=False)
        x0 = torch.tensor(self.data[random_indices,:2], dtype=torch.float32).to(device)  # Initial states
        x_r = torch.tensor(self.data[random_indices,2], dtype=torch.float32).to(device)


        x_upper_bound = 100
        x_lower_bound = -10

        # build observer gains:
        observer_poles = np.array([0.85, 0.9])
        L_res = place_poles(Ad.T,Cd.T,observer_poles)
        L = (L_res.gain_matrix).T





        # =========== Euler Method Simulation ============
        iterations = 300
        xset = torch.zeros((iterations,pts_count, 2))  # Initialize state set
        uset = torch.zeros((iterations,pts_count, 1))  # Initialize state set
        select_mask = torch.ones_like(x0[:,0], dtype=torch.bool)  # Mask to select valid points


        x = x0.clone()
        for i in range(iterations):

            # Prepare input for the neural network
            nn_input = torch.cat((x, x_r.unsqueeze(-1)), dim=1)  # Concatenate current state and reference trajectory
            # nn_input = nn_input.unsqueeze(1)  # Reshape to (batch_size, input_size)
            nn_input = nn_input.float()  # Ensure input is float type

            # Get the neural network output
            with torch.no_grad():
                nn_output = model(nn_input.to(device))
                
                # unnormalize the output
                nn_output = nn_output * (u_max - u_min) + u_min


            u = nn_output
            u0 = nn_output[:,0].unsqueeze(-1) # Control input
            xset[i,:, :] = x.cpu()
            uset[i,:, :] = u[:,0].unsqueeze(-1).cpu()

            # expand A and B
            Add = torch.tile(torch.tensor(Ad,dtype=torch.float32), (u.shape[0], 1, 1)).to(device)
            Bdd = torch.tile(torch.tensor(Bd,dtype=torch.float32), (u.shape[0], 1, 1)).to(device)
            x_prime = x.unsqueeze(1).transpose(1, 2).to(device)

            #  observer update
            Cdd = torch.tile(torch.tensor(Cd,dtype=torch.float32), (u.shape[0], 1, 1)).to(device)
            y = Cdd @ x_prime;  

            Ldd = torch.tile(torch.tensor(L,dtype=torch.float32), (u.shape[0], 1, 1)).to(device)
            x_observer = Add @ x_prime + Bdd @ u0.unsqueeze(1) + Ldd @ (y - Cdd @ x_prime)  
            
            
            x_new = Add @ x_prime + Bdd @ u0.unsqueeze(1)

            x1_new = x_observer[:,0,0]
            x2_new = x_new[:,1,0]


            inliters = (x1_new >= x_lower_bound) & (x1_new <= x_upper_bound) & \
                        (x2_new >= x_lower_bound) & (x2_new <= x_upper_bound)
            
            select_mask = select_mask & inliters # Update the mask to keep only inliers
            
            # remove x on outliers

            # Update state only for inliers
            x = torch.stack((x1_new, x2_new), dim=1)
            # print(x.shape)

            # =========== Objective Function Evaluation ============
            # obj_func_vals = criterion2.forward(nn_input.to(device),nn_output.to(device))



        # ========== Performance Index Calculation -- Consider ROI ============
        select_mask = select_mask.cpu()
        valid_trajectories = select_mask.sum()

        if valid_trajectories > 0:
            xset = xset[:,select_mask,:]  # Filter xset with the mask

            xr_tile = torch.tile(torch.stack((torch.zeros_like(x_r), x_r)).T,(xset.shape[0],1,1)).cpu()
            Performance_index = torch.norm(xset - xr_tile[:,select_mask, :], dim=2)
            Performance_index = Performance_index.sum(0).mean()
            # calculate PI for u:
            uset = uset[:,select_mask,:]
            Performance_index_u = torch.norm(uset, dim=2).sum(0).mean()
            # calculate u violation
            u_violation = torch.sum(torch.abs(uset) > u_max).item()
            
            print(f'****Valid trajectories: {valid_trajectories} | Violation: {u_violation}****')
            print(f'Performance Index ||x-xr||: {Performance_index:.3f} | Performance Index ||u||: {Performance_index_u:.3f}')
        else:
            print("No trajectories are in the range of [-5, 5]")
            print("Mission Failed!")
            Performance_index = 100000
            Performance_index_u = 100000
            u_violation = 100000

        # =========== Performance Index Calculation without ROI============
        # Performance_index = torch.norm(xset - x_r, dim=2)
        # print(f'Performance Index: {Performance_index.sum(0).mean()}')


        # =========== Plotting Results ============
        if model_path is not None:
            lines_display = 1000  # Number of points to display in the plot
            plt.figure(figsize=(10, 6))
            plt.scatter(xset[:, :lines_display, 0].flatten(), xset[:, :lines_display, 1].flatten(), s=1, label='State Points', alpha=0.5)
            # show the first 5 trajectories with red color
            plt.scatter(xset[:, :5, 0].flatten(), xset[:, :5, 1].flatten(), s=10, color='red', label='First 5 Trajectories', alpha=0.8)

            plt.title('State Points after Simulation')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.axhline(0, color='black', lw=0.5, ls='--')
            plt.axvline(0, color='black', lw=0.5, ls='--')
            plt.grid()
            plt.legend()
            plt.show()
            # print(f'Final Objective Function Value: {obj_func_vals.item()}')


        return Performance_index, Performance_index_u, u_violation

    def test_benchmark_mpc(self,):
        """
        Test the performance index of MPC given a reference trajectory xr.
        Same as test_performance_index but use MPC to get u.

        :param device: The device to run the model on (CPU or GPU).
        :return: The performance index calculated as the mean of the sum of norms
        """
        
        saved_jsonfile = f'semi_supervison/dataset/Benchmark-MPC-trajectory10000.json'
        
        # =========== Input Data Preparation ============
        pts_count = 10000
        # Generate random initial states between -5 and 5:
        torch.manual_seed(42)  # For reproducibility
        
        # random select init state from dataset
        random_indices = np.random.choice(len(self.data), size=pts_count, replace=False)
        x0 = torch.tensor(self.data[random_indices,:2], dtype=torch.float32)  # Initial states
        x_r = torch.tensor(self.data[random_indices,2], dtype=torch.float32)


        # =========== Euler Method Simulation ============
        iterations = 300

        if os.path.exists(saved_jsonfile):
            with open(saved_jsonfile, 'r') as f:
                trajectory_data = json.load(f)
            xset = np.array(trajectory_data['xset'])
            uset = np.array(trajectory_data['uset'])
            x_r_check = np.array(trajectory_data['xr'])

            # check if need to simulate more
            if x_r_check.shape[0]< pts_count:
                print(f"Existing trajectory has {x_r_check.shape[0]} points, need to simulate more to reach {pts_count} points.")
                x0 = x0[x_r_check.shape[0]:]
                x_r = x_r[x_r_check.shape[0]:]
            else:
                print(f"Existing trajectory has {x_r_check.shape[0]} points, no need to simulate more.")
      

                # ========== Performance Index Calculation -- Consider ROI ============
                xset = torch.tensor(xset, dtype=torch.float32)
                x_r = torch.zeros_like(xset)
                for i in range(x_r_check.shape[0]):
                    x_r[iterations*i:iterations*(i+1),0] = x_r_check[i]

                                
                Performance_index = torch.norm(xset - x_r)
                # calculate PI for u:
                uset = torch.tensor(uset, dtype=torch.float32)
                Performance_index_u = torch.norm(uset)
                # calculate u violation
                u_violation = torch.sum(torch.abs(uset) > u_max).item()
                print(f'Performance Index ||x-xr||: {Performance_index:.3f} | Performance Index ||u||: {Performance_index_u:.3f} | Violation: {u_violation}')
                
                return 0
            


        for j in tqdm(range(x0.shape[0])):
            x = (x0[j]).numpy().reshape(-1,1)
            xr = (x_r[j]).item()

            xset = np.zeros((iterations, 2))  # Initialize state set
            uset = np.zeros((iterations, 1))  # Initialize state set
            
            for i in range(iterations):

                u = mpc_fun(Ad, Bd, Q, R, x, np.array([xr, 0]).reshape(-1, 1), N)

                xset[i, :] = x.flatten()
                uset[i,:] = u[0]


                x_new = Ad @ x + Bd * u[0]

                # Update state only for inliers
                x = x_new


            if os.path.exists(saved_jsonfile):
                with open(saved_jsonfile, 'r') as f:
                    trajectory_data = json.load(f)
                xset_tmp = np.array(trajectory_data['xset'])
                uset_tmp = np.array(trajectory_data['uset'])
                x_r_tmp = np.array(trajectory_data['xr'])

                xset = np.vstack((xset_tmp, xset))
                uset = np.vstack((uset_tmp, uset))
                xrset = np.hstack((x_r_tmp, xr))
            else:
                xrset = np.array([xr])

            with open(saved_jsonfile, 'w') as f:
                json.dump({'xset': xset.tolist(),
                        'uset': uset.tolist(),
                        'xr': xrset.tolist()}, f)








class TruthAwareSampler(Sampler):
    '''
    Ensures each batch contains all truth data point.
    '''
    def __init__(self, dataset, batch_size, truth_indices):
        self.dataset = dataset
        self.batch_size = batch_size
        self.truth_indices = truth_indices
        self.all_indices = list(range(len(dataset)))
        
        # Non-truth indices
        self.other_indices = list(set(self.all_indices) - set(truth_indices))

        # print(f"Total samples: {len(self.all_indices)}, Truth samples: {len(self.truth_indices)}, Other samples: {len(self.other_indices)}")

    def __iter__(self):
        # Shuffle both sets
        random.shuffle(self.truth_indices)
        random.shuffle(self.other_indices)
        # Fill batches
        num_batches = math.ceil(len(self.all_indices) / self.batch_size)
        truth_cycle = iter(self.truth_indices)

        for b in range(num_batches):
            batch = []

            # 1. Always add one truth point (cycled if fewer than batches)
            try:
                batch.append(next(truth_cycle))
            except StopIteration:
                # Restart the cycle
                truth_cycle = iter(self.truth_indices)
                batch.append(next(truth_cycle))

            # 2. Fill rest of batch with random non-truth points
            while len(batch) < self.batch_size and self.other_indices:
                batch.append(self.other_indices.pop())

            yield batch

    def __len__(self):
        return math.ceil(len(self.all_indices) / self.batch_size)


'''
if __name__ == "__main__":
    ############ Example of using UpdatingDataset ############
    from torch.utils.data import DataLoader

    dataset = UpdatingDataset(mode='uniform', sampling_num_per_xr=20, opt_dataset_path='mpc_data_heater.csv')
    dataset.test_benchmark_mpc()
    # sampler = TruthAwareSampler(dataset, batch_size=1024, truth_indices=dataset.TruthData_Mask)
    # dataloader = DataLoader(dataset, batch_sampler=sampler)

    # for nn_input, u_seq in dataloader:
    #     print(nn_input.shape, u_seq.shape)
    #     break
'''
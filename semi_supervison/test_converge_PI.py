'''
This script tests the Performance Index (PI) of the simulation model
given a set of input x and the output from the trained neural network.
'''

import torch
import numpy as np
from network import P_Net
from Objective_Formulations_new import ObjectiveFormulation_ORI
from Objective_Formulations_mpc import ObjectiveFormulation
import argparse
import matplotlib.pyplot as plt
import control


dt = 0.1

# % read A,B,C,D matrices:
A = np.array([[0, 1], [0, -1.7873]])
B = np.array([[0],[-1.7382]])
C = np.array([[1,0]])
D = 0

ss = control.ss(A, B, C, D)
Gd = control.c2d(ss,dt, method='zoh') # 'zoh':assuming control input be constant between sampling intervals.

Ad = torch.tensor(Gd.A,dtype=torch.float32)
Bd = torch.tensor(Gd.B,dtype=torch.float32)
Cd = torch.tensor(Gd.C,dtype=torch.float32)
Dd = torch.tensor(Gd.D,dtype=torch.float32)


def test_performance_index(model,device,xr=0., model_path=None):
    """
    Test the performance index of the model given a reference trajectory xr.
    :param model: The trained neural network model.
    :param device: The device to run the model on (CPU or GPU).
    :param xr: The reference trajectory value.
    :return: The performance index calculated as the mean of the sum of norms
    """
    
    # ============ Model Argument Parsing ============

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load the trained model
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('----------added previous weights: {}------------'.format(model_path))
    # model = P_Net(output_size=5).to(device)
    # trained_model_path = 'mathmatical_simulation/weights/test_model_weight_0.05_0.9_0.05.pth'
    # model.load_state_dict(torch.load(trained_model_path))
    # print('----------added previous weights: {}------------'.format(trained_model_path))

    model.eval()


    # =========== Input Data Preparation ============
    pts_count = 10000
    # Generate random initial states between -5 and 5:
    torch.manual_seed(42)  # For reproducibility
    # x0 = (torch.rand(pts_count,2)*10-5)  # Random initial states
    x0 = (torch.rand(pts_count,2)*4-2)  # Random initial states

    x_upper_bound = 10
    x_lower_bound = -10

    x_r = torch.tensor([[xr]] * pts_count)  # Reference trajectory


    # =========== Euler Method Simulation ============
    criterion2 = ObjectiveFormulation_ORI(device=device)
    iterations = 300
    xset = torch.zeros((iterations,pts_count, 2))  # Initialize state set
    select_mask = torch.ones_like(x0[:,0], dtype=torch.bool)  # Mask to select valid points

    x = x0.clone()
    for i in range(iterations):

        # Prepare input for the neural network
        nn_input = torch.cat((x, torch.tensor([[0.]] * x.shape[0])), dim=1)  # Concatenate current state and reference trajectory
        # nn_input = nn_input.unsqueeze(1)  # Reshape to (batch_size, input_size)
        nn_input = nn_input.float()  # Ensure input is float type

        # Get the neural network output
        with torch.no_grad():
            nn_output = model(nn_input.to(device))
            # nn_output order: # [p1, p2, p3, u, theta]

        nn_output = nn_output.cpu()  # Move output to CPU for further processing

        u = nn_output[:, 3]  # Control input
        theta = nn_output[:, 4]  # Theta parameter

        xset[i,:, :] = x

        x1_new = x[:, 0] + dt * x[:, 1]
        x2_new = x[:, 1] + dt * (x[:, 0]**3+(x[:, 1]**2+1)*u)



        inliters = (x1_new >= x_lower_bound) & (x1_new <= x_upper_bound) & \
                    (x2_new >= x_lower_bound) & (x2_new <= x_upper_bound)
        
        select_mask = select_mask & inliters  # Update the mask to keep only inliers
        
        # remove x on outliers

        # Update state only for inliers
        x = torch.stack((x1_new, x2_new), dim=1)
        # print(x.shape)

        # =========== Objective Function Evaluation ============
        # obj_func_vals = criterion2.forward(nn_input.to(device),nn_output.to(device))



    # ========== Performance Index Calculation -- Consider ROI ============
    valid_trajectories = select_mask.sum()

    if valid_trajectories > 0:
        xset = xset[:,select_mask,:]  # Filter xset with the mask

        Performance_index = torch.norm(xset - x_r[:valid_trajectories, :], dim=2)
        print(f'Performance Index: {Performance_index.sum(0).mean()} | Trajectories in range: {valid_trajectories} | Ratio: {Performance_index.sum(0).mean()/valid_trajectories}')
    else:
        print("No trajectories are in the range of [-5, 5]")
        print("Mission Failed!")
        Performance_index = torch.tensor([1e5])

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


    return Performance_index.sum(0).mean(), valid_trajectories


def trajectory_plot(true_data,model_predict_data):

    true_data = np.array(true_data)  # Ensure true_data is in the correct shape
    output_data = np.array(model_predict_data)  # Ensure output_data is in the correct shape

    plt.figure(figsize=(10, 6))

    for i in range(true_data.shape[0]):
        plt.plot(true_data[i,:,0],true_data[i,:,1],linestyle='-',color='blue', marker='*', markersize=5)
        plt.plot(output_data[i,:,0],output_data[i,:,1],linestyle='--',color='red', marker='o', markersize=5)

    # plt.plot(output_data[:, 0], output_data[:, 1], label='Output Trajectory', color='red', linestyle='--')
    plt.title('Trajectory Comparison')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.grid()
    plt.legend(['Optimization', 'Trained NN'])
    plt.tight_layout()
    plt.savefig('semi_supervison/DroneZ_MPC_weights/trajectory_compare.png')
    plt.show()
    plt.close()



def test_trajectory(model,val_trajectory,device):

    label_x, label_vals = val_trajectory

    steps = len(label_x[0])  # Number of steps in the trajectory
    

    init_pts = np.array(label_x)[:,0,:]  # Initial points for the trajectories

    init_pts_in = torch.tensor(init_pts, dtype=torch.float32).to(device)
    
    # Initialize the state set
    xset = torch.zeros((init_pts_in.shape[0], steps, 2)).to(device)
    nn_output_set = torch.zeros((init_pts_in.shape[0], steps, 5)).to(device)

    for i in range(steps):
        with torch.no_grad():
            model.eval()
            nn_output = model(init_pts_in)
            # nn_output order: # [p1, p2, p3, u, theta] 

            # save state
            xset[:, i, :] = init_pts_in[:,:2]
            nn_output_set[:, i, :] = nn_output

            # update state
            u = nn_output[:, 3]
            x1_new = init_pts_in[:, 0] + dt * init_pts_in[:, 1]
            x2_new = init_pts_in[:, 1] + dt * (init_pts_in[:, 0]**3 + (init_pts_in[:, 1]**2 + 1) * u)
            init_pts_in = torch.stack((x1_new, x2_new, torch.zeros_like(x1_new)), dim=1)


    trajectory_plot(label_x, xset.cpu().numpy())

    # compare other paramters with true label
    label_vals = np.array(label_vals)  # Ensure label_vals is in the correct shape
    objective_formulation = ObjectiveFormulation_ORI()
    # Forward pass through the objective formulation
    
    xset = xset.reshape(-1,2) 
    nn_output_set = nn_output_set.reshape(-1,5)  # Reshape to match the expected input shape
    obj_nn_output = objective_formulation.forward_any(torch.tensor(xset).cpu(), torch.tensor(nn_output_set).cpu())
    
    label_x = torch.tensor(label_x,dtype=torch.float32)[:,:,:2].reshape(-1,2) 
    label_vals = torch.tensor(label_vals,dtype=torch.float32).reshape(-1,5) 
    
    obj_label_vals = objective_formulation.forward_any(label_x,label_vals)
    # plot two values:
    plt.figure(figsize=(10, 6))
    plt.plot(obj_nn_output.cpu().numpy(), label='NN Output J+Ci', color='red')
    plt.plot(obj_label_vals.cpu().numpy(), label='Optimization J+Ci', color='blue')
    # display average values as text blocks on the figure:
    avg_nn_output = obj_nn_output.mean().item()
    avg_label_vals = obj_label_vals.mean().item()
    plt.text(200, 150, f'Avg NN Output: {avg_nn_output:.2f}', color='red', fontsize=15)
    plt.text(200, 140, f'Avg Opt Output: {avg_label_vals:.2f}', color='blue', fontsize=15)
    plt.title('Objective Function Values +Constraints Comparison')
    plt.xlabel('Steps')
    plt.ylabel('Objective Function Value + Constraints')
    plt.grid(linestyle='--', color='gray')
    plt.legend()
    plt.tight_layout()
    plt.savefig('semi_supervison/weights/trajectory_obj_compare.png')
    plt.show()
    plt.close()


def test_trajectory_MPC(model,val_trajectory,device):

    label_x, label_vals = val_trajectory

    steps = len(label_x[0])  # Number of steps in the trajectory
    

    init_pts = np.array(label_x)[:,0,:]  # Initial points for the trajectories

    init_pts_in = torch.tensor(init_pts, dtype=torch.float32).to(device)
    
    # Initialize the state set
    xset = torch.zeros((init_pts_in.shape[0], steps, 2)).to(device)
    uset = torch.zeros((init_pts_in.shape[0], steps, 1)).to(device)
    nn_output_set = torch.zeros((init_pts_in.shape[0], steps, 50)).to(device)

    for i in range(steps):
        with torch.no_grad():
            model.eval()
            nn_output = model(init_pts_in)
            # nn_output order: # [p1, p2, p3, u, theta] 

            # save state
            xset[:, i, :] = init_pts_in[:,:2]
            nn_output_set[:, i, :] = nn_output

            # update state
            u = nn_output[:, 0].unsqueeze(1)
            uset[:, i, :] = u

            Add = torch.tile(Ad, (u.shape[0], 1, 1)).to(device)
            Bdd = torch.tile(Bd, (u.shape[0], 1, 1)).to(device)
            x_prime = init_pts_in[:, :2].unsqueeze(1).transpose(1, 2)
            x_new = Add @ x_prime + Bdd @ u.unsqueeze(1)
            init_pts_in = torch.cat((x_new.squeeze(-1), init_pts_in[:,2].unsqueeze(-1)), dim=1)


    trajectory_plot(label_x, xset.cpu().numpy())

    # plot x1,x2,u with three sub plot
    label_x = np.array(label_x)[:,:,:2] #(16,100,2)
    xset = xset.cpu().numpy()
    diff_x = np.mean(label_x-xset,axis=0)
    diff_x1 = diff_x[:,0]
    diff_x2 = diff_x[:,1]
    diff_x = np.mean(diff_x,axis=1)

    uset = uset.cpu().numpy().squeeze(-1)
    label_u = np.array(label_vals)[:,:,0]
    diff_u = np.mean(label_u-uset,axis=0)


    x = np.arange(diff_x.shape[0])
    plt.figure(figsize=(10, 6))
    plt.subplot(411)
    plt.plot(x,diff_x,linestyle='-',color='blue', marker='*', markersize=3)
    plt.grid(linestyle='--', color='gray')    
    plt.ylabel('Diff of mean |x|')
    plt.legend()
    plt.title("Difference between MPC Control and NN Control (MPC Out-NN Out)")

    plt.subplot(412)
    plt.plot(x,diff_x1,linestyle='-',color='blue', marker='*', markersize=3)
    plt.grid(linestyle='--', color='gray')    
    plt.ylabel('Diff of mean |x1|')
    plt.legend()
    
    plt.subplot(413)
    plt.plot(x,diff_x2,linestyle='-',color='blue', marker='*', markersize=3)
    plt.grid(linestyle='--', color='gray')    
    plt.ylabel('Diff of mean |x2|')
    plt.legend()

    plt.subplot(414)
    plt.plot(x,diff_u,linestyle='-',color='blue', marker='*', markersize=3)
    plt.grid(linestyle='--', color='gray')    
    plt.ylabel('Diff of mean |u|')
    plt.xlabel('Steps')
    plt.legend()

    plt.savefig('semi_supervison/DroneZ_MPC_weights/diff_x1x2u_compare.png')
    plt.show()
    plt.close()
    

    '''
    # compare other paramters with true label
    label_vals = np.array(label_vals)  # Ensure label_vals is in the correct shape
    objective_formulation = ObjectiveFormulation()
    # Forward pass through the objective formulation
    
    xset = xset.reshape(-1,2) 
    nn_output_set = nn_output_set.reshape(-1,5)  # Reshape to match the expected input shape
    obj_nn_output = objective_formulation.forward_any(torch.tensor(xset).cpu(), torch.tensor(nn_output_set).cpu())
    
    label_x = torch.tensor(label_x,dtype=torch.float32)[:,:,:2].reshape(-1,2) 
    label_vals = torch.tensor(label_vals,dtype=torch.float32).reshape(-1,5) 
    
    obj_label_vals = objective_formulation.forward_any(label_x,label_vals)
    # plot two values:
    plt.figure(figsize=(10, 6))
    plt.plot(obj_nn_output.cpu().numpy(), label='NN Output J+Ci', color='red')
    plt.plot(obj_label_vals.cpu().numpy(), label='Optimization J+Ci', color='blue')
    # display average values as text blocks on the figure:
    avg_nn_output = obj_nn_output.mean().item()
    avg_label_vals = obj_label_vals.mean().item()
    plt.text(200, 150, f'Avg NN Output: {avg_nn_output:.2f}', color='red', fontsize=15)
    plt.text(200, 140, f'Avg Opt Output: {avg_label_vals:.2f}', color='blue', fontsize=15)
    plt.title('Objective Function Values +Constraints Comparison')
    plt.xlabel('Steps')
    plt.ylabel('Objective Function Value + Constraints')
    plt.grid(linestyle='--', color='gray')
    plt.legend()
    plt.tight_layout()
    plt.savefig('semi_supervison/weights/trajectory_obj_compare.png')
    plt.show()
    plt.close()
    '''




# '''
if __name__ == "__main__":

    # # test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = P_Net(output_size=50).to(device)
    ckpt_path = 'semi_supervison/DroneZ_MPC_weights/weight_best.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    PI, valid_cnt = test_performance_index(model, device=device, xr=0.0, model_path=ckpt_path)
    # print(f'Performance Index: {PI} | Valid Trajectories Count: {valid_cnt}')

    from dataprocessing import dataloading_MPC
    x_trajectory, trajectories_in_val, trajectories_label_val = dataloading_MPC('DroneZ_MPC/dataset/droneZ_MPC_16trajectory.csv')
    test_trajectory_MPC(model, (x_trajectory, trajectories_label_val), device)

# '''

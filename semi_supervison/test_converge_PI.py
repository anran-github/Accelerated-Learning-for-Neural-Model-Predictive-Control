'''
This script tests the Performance Index (PI) of the simulation model
given a set of input x and the output from the trained neural network.
'''

import torch
import numpy as np
from network import P_Net
from Objective_Formulations_new import ObjectiveFormulation
import argparse
import matplotlib.pyplot as plt



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

    dt = 0.1
    x_upper_bound = 10
    x_lower_bound = -10

    x_r = torch.tensor([[xr]] * pts_count)  # Reference trajectory


    # =========== Euler Method Simulation ============
    criterion2 = ObjectiveFormulation(device=device)
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
        x2_new = x[:, 1] + dt * dt*(x[:, 0]**3+(x[:, 1]**2+1)*u)



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

# # test
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = P_Net(output_size=5).to(device)
# PI, valid_cnt = test_performance_index(model, device=device, xr=0.0, model_path='mathmatical_simulation/weights/no_noise_weight_1_0.0_0.0 copy.pth')
# print(f'Performance Index: {PI} | Valid Trajectories Count: {valid_cnt}')
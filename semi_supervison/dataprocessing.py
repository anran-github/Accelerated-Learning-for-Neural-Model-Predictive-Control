import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Data_Purify():
    def __init__(self,filename,xr,p_max,u_max) -> None:

        # Load the CSV dataset
        df = pd.read_csv(filename, header=None)

        # Format:
        # x1 p1 p2 u
        # x2 p2 p3 theta
        self.purify_factor_p = p_max
        self.purify_factor_u = u_max

        # Assuming the first row is the header, adjust if needed
        data = df.values  # Transpose to have shape (2, n)


        self.data_x = []
        self.data_u = []
        self.data_p = []
        self.data_theta = []
        instance = 2
        for row in range(data.shape[0]//instance):
            self.data_x.extend([data[instance*row:instance*row+instance,0]])
            self.data_u.extend([data[instance*row,3]])
            self.data_p.extend([data[instance*row:instance*row+instance,1:3]])
            self.data_theta.extend([data[instance*row+1,3]])

        
        # self.data_theta = np.tile(xr,(len(self.data_u),1))
        print('----------DATA SUMMARY------------')
        print(f'There are {len(self.data_u)} raw data points')

        # combine [x1,x2] and r as a single input:
        # self.input_data_combine = [np.append(self.data_x[i],self.data_theta[i]) for i in range(len(self.data_x))]
        self.input_data_combine = [np.append(self.data_x[i],0) for i in range(len(self.data_x))]

        # now we only need 4: p1,p2,p3,..., p10, u, theta
        self.label_data = [[self.data_p[i][0,0],self.data_p[i][0,1],self.data_p[i][1,1],
                            self.data_u[i], self.data_theta[i]] for i in range(len(self.data_u))]

    def return_data(self,):
            return self.input_data_combine, self.label_data



    def purified_data(self):

        data_p1 = [x[0,0] for x in self.data_p]
        data_p2 = [x[0,1] for x in self.data_p]
        data_p3 = [x[1,1] for x in self.data_p]

        data_p1 = np.array(data_p1)
        data_p2 = np.array(data_p2)
        data_p3 = np.array(data_p3)

        self.data_u = np.array(self.data_u)
        mask_outlier1 = np.abs(data_p1) < (self.purify_factor_p)
        mask_outlier3 = np.abs(data_p2) < (self.purify_factor_p)
        mask_outlier4 = np.abs(data_p3) < (self.purify_factor_p)
        mask_outlier2 = np.abs(self.data_u) < (self.purify_factor_u)
        self.mask_outlier = np.logical_and(
            np.logical_and(mask_outlier1,mask_outlier2),
            np.logical_and(mask_outlier3,mask_outlier4)
        )

        # for outliers
        self.outlier_mask = np.logical_not(self.mask_outlier)
        self.outliers = np.array(self.data_x)[self.outlier_mask]
        # use mask get rid of outliers
        self.data_p = np.array(self.data_p)[self.mask_outlier]
        self.data_theta = np.array(self.data_theta)[self.mask_outlier]
        self.data_x = np.array(self.data_x)[self.mask_outlier]
        self.data_u = np.array(self.data_u)[self.mask_outlier]
        print(f'Dataset is purified! Now there are {len(self.data_theta)} data points available.')
        print('--------------------------------')

        self.len_data = len(self.data_u)

        # Final Data for feeding NN.

        # combine [x1,x2] and r as a single input:
        self.input_data_combine = [np.append(self.data_x[i],0) for i in range(len(self.data_x))]

        # now we only need 5: p1,p2,p3,u, theta
        self.label_data = [[self.data_p[i][0,0],self.data_p[i][0,1],
                            self.data_p[i][1,1],self.data_u[i],self.data_theta[i]] for i in range(len(self.data_u))]

        return self.input_data_combine, self.label_data


    def draw_data(self, data_u, data_p):

        plt.subplot(221)
        plt.plot(list(range(len(data_u))),data_u)
        # plt.show()
        plt.title('Parameter u(t)')
        # plt.close()

        plt.subplot(222)
        data_p1 = [x[0,0] for x in data_p]
        plt.plot(list(range(len(data_p1))),data_p1)
        # plt.show()
        plt.title('Parameter p(1)')

        plt.subplot(223)
        data_p2 = [x[0,1] for x in data_p]
        plt.plot(list(range(len(data_p2))),data_p2)
        # plt.show()
        plt.title('Parameter p2')

        plt.subplot(224)
        data_p3 = [x[1,1] for x in data_p]
        plt.plot(list(range(len(data_p3))),data_p3)
        plt.show()
        plt.title('Parameter p3')
        plt.close()


def dataloading(filename):
    '''
    Load initial data point and corresponding trajectories.
    '''
    # Load the CSV dataset
    df = pd.read_csv(filename, header=None)

    # Data Format:
    # x1 p1 p2 u
    # x2 p2 p3 theta

    # Assuming the first row is the header, adjust if needed
    data = df.values  # Transpose to have shape (2, n)

    
    init_input = [] # shape (n, 3): # [x1, x2, xr], n = number of data points
    init_label = []     # shape (n, 5): [p1, p2, p3, u, theta]
    trajectories_init_input = [] # shape (n, 3,steps) 
    trajectories_init_label = [] # shape (n, 5,steps)
    instance = 2

    for row in range(data.shape[0]//instance):

        init_input.append([data[instance*row,0],data[instance*row+1,0],0])
        u = data[instance*row,3]
        p1 = data[instance*row,1]
        p2 = data[instance*row,2]
        p3 = data[instance*row+1,2]
        theta = data[instance*row+1,3]

        init_label.extend([[p1, p2, p3, u, theta]])

        temp_x = []
        temp_label = []
        for col in range(data.shape[1]//4):
            # collect trajectories:
            temp_x.append([data[instance*row,4*col],data[instance*row+1,4*col],0])
            
            u = data[instance*row,4*col+3]
            p1 = data[instance*row,4*col+1]
            p2 = data[instance*row,4*col+2]
            p3 = data[instance*row+1,4*col+2]
            theta = data[instance*row+1,4*col+3]

            temp_label.extend([[p1, p2, p3, u, theta]])

        trajectories_init_input.append(temp_x)
        trajectories_init_label.append(temp_label)

    # norm the input label:
    # init_label_max = np.max(np.array(init_label), axis=0)
    # init_label = np.array(init_label) / init_label_max
    # init_label = init_label.tolist()


    # data_theta = np.tile(xr,(len(data_u),1))
    print('----------DATA SUMMARY------------')
    print(f'Number of initial data points: {len(init_input)}')
    print(f'Number of trajectories: {len(trajectories_init_input)}')
    print('----------------------------------')

    return init_input, init_label, trajectories_init_input, trajectories_init_label


def dataloading_MPC(filename):
    '''
    Load initial data point and corresponding trajectories.
    '''
    # Load the CSV dataset
    df = pd.read_csv(filename, header=None)

    step_length = 100

    # Data Format:
    # x1 x2,xr, u

    # Assuming the first row is the header, adjust if needed
    data = df.values  # Transpose to have shape (2, n)

    x_trajectory = []
    trajectories_init_input, trajectories_step_label = [], []
    label_tmp = []
    for row in range(data.shape[0]//step_length):

        for step in range(step_length):
            if step == 0:
                trajectories_init_input.extend([data[row*step_length+step,:3]])

        label_tmp = data[row*step_length:row*step_length+step+1,3:]
        trajectories_step_label.append(label_tmp)

        x_trajectory.append(data[row*step_length:row*step_length+step+1,:3])

            

    return x_trajectory, trajectories_init_input, trajectories_step_label



# Data loading for Drone-Z
class DroneZ_Data_Purify():
    def __init__(self,filename,u_max) -> None:

        # Load the CSV dataset
        df = pd.read_csv(filename, header=None)

        # Format:
        # x1 x2 xr u
        self.purify_factor_u = u_max

        # Assuming the first row is the header, adjust if needed
        data = df.values 


        self.data_x = []
        self.data_u = []
        self.data_xr = []
        self.input_data_combine = []

        for row in range(data.shape[0]):
            self.data_x.extend([data[row,:2]])
            self.data_u.extend([data[row,3:]])
            self.data_xr.extend([data[row,2]])

            self.input_data_combine.append([data[row,0], data[row,1], data[row,2]])

        
        # self.data_theta = np.tile(xr,(len(self.data_u),1))
        print('----------DATA SUMMARY------------')
        print(f'There are {len(self.data_u)} raw data points')


        # now we only need 1: u
        self.label_data = self.data_u.copy()

    def return_data(self,):
            return self.input_data_combine, self.label_data



    def purified_data(self):

        data_p1 = [x[0,0] for x in self.data_p]
        data_p2 = [x[0,1] for x in self.data_p]
        data_p3 = [x[1,1] for x in self.data_p]

        data_p1 = np.array(data_p1)
        data_p2 = np.array(data_p2)
        data_p3 = np.array(data_p3)

        self.data_u = np.array(self.data_u)
        mask_outlier1 = np.abs(data_p1) < (self.purify_factor_p)
        mask_outlier3 = np.abs(data_p2) < (self.purify_factor_p)
        mask_outlier4 = np.abs(data_p3) < (self.purify_factor_p)
        mask_outlier2 = np.abs(self.data_u) < (self.purify_factor_u)
        self.mask_outlier = np.logical_and(
            np.logical_and(mask_outlier1,mask_outlier2),
            np.logical_and(mask_outlier3,mask_outlier4)
        )

        # for outliers
        self.outlier_mask = np.logical_not(self.mask_outlier)
        self.outliers = np.array(self.data_x)[self.outlier_mask]
        # use mask get rid of outliers
        self.data_p = np.array(self.data_p)[self.mask_outlier]
        self.data_theta = np.array(self.data_theta)[self.mask_outlier]
        self.data_x = np.array(self.data_x)[self.mask_outlier]
        self.data_u = np.array(self.data_u)[self.mask_outlier]
        print(f'Dataset is purified! Now there are {len(self.data_theta)} data points available.')
        print('--------------------------------')

        self.len_data = len(self.data_u)

        # Final Data for feeding NN.

        # combine [x1,x2] and r as a single input:
        self.input_data_combine = [np.append(self.data_x[i],0) for i in range(len(self.data_x))]

        # now we only need 5: p1,p2,p3,u, theta
        self.label_data = [[self.data_p[i][0,0],self.data_p[i][0,1],
                            self.data_p[i][1,1],self.data_u[i],self.data_theta[i]] for i in range(len(self.data_u))]

        return self.input_data_combine, self.label_data


    def draw_data(self,):

        # Convert to grid
        x1_unique = np.sort(np.unique(np.round(np.array(self.data_x)[:,0],3)))
        x2_unique = np.sort(np.unique(np.round(np.array(self.data_x)[:,1],3)))

        X, Y = np.meshgrid(x1_unique, x2_unique)
        Z = np.zeros_like(X, dtype=float)

        # Fill Z values
        for i in range(len(self.data_x)):
            xi = round(self.data_x[i][0],3)
            yi = round(self.data_x[i][1],3)
            vi = self.data_u[i]

            x_idx = x1_unique.tolist().index(xi)
            y_idx = x2_unique.tolist().index(yi)
            Z[y_idx, x_idx] = vi


        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')

        # Labels
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('U Value')
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
        plt.close()


'''
# data_tmp = DroneZ_Data_Purify('DroneZ_MPC/dataset/drone_mpc_z.csv', u_max=1e5)
# data_tmp.draw_data()
from test_converge_PI import test_performance_index, test_trajectory_MPC, trajectory_plot
from network import P_Net
import torch
x_trajectory, trajectories_in_val, trajectories_label_val = dataloading_MPC('DroneZ_MPC/dataset/droneZ_MPC_16trajectory.csv')
model = P_Net(output_size=50).cuda()
model.load_state_dict(torch.load('semi_supervison/DroneZ_MPC_weights/exp1/weight_best.pth'))
test_trajectory_MPC(model, (x_trajectory, trajectories_label_val), device='cuda:0')
'''
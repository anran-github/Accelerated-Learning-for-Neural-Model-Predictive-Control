import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import lr_scheduler

from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse
import time

# import NN structure:
from network import P_Net

from Objective_Formulations_mpc import ObjectiveFormulation
from test_converge_PI import test_performance_index, test_trajectory_MPC, trajectory_plot
from dataprocessing import DroneZ_Data_Purify, dataloading_MPC




# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")


# Parse the command-line arguments

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='DroneZ_MPC/dataset/drone_mpc_z.csv', help='corresponding theta dataset')
parser.add_argument('--reference',type=float,default=1.5, help='reference of dataset')
parser.add_argument('--lr',type=float, default=5e-4, help='learning rate')
parser.add_argument('--init_pts_mode', type=str, default='uniform',help='input your pretrained weight path if you want')
parser.add_argument('--pre_trained', type=str, default='',help='input your pretrained weight path if you want')
args = parser.parse_args()
print(args)

save_path = f'semi_supervison/DroneZ_MPC_weights'
os.makedirs(save_path, exist_ok=True)
x_r = np.array([args.reference])
data_tmp = DroneZ_Data_Purify(args.dataset,u_max=1e5)


# ============== Import and Purify Data ==============
# # import dataset 
# input_data_valid,label_data_valid = data_tmp.purified_data()
input_data_valid,label_data_valid = data_tmp.return_data()



if args.init_pts_mode == 'uniform':
    # METHOD 1: select 16 data points uniformly:
    x1 = np.linspace(0.5, 2.5, 4)
    x2 = np.linspace(-1, 1, 4)
    X_selected = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
elif args.init_pts_mode == 'boundary':
    # METHOD 2: select 16 data points on the operation boundary
    # 4 points along each edge
    x = np.linspace(0.5, 2.5, 5)
    y = np.linspace(-1, 1, 5)

    # Get boundary points
    bottom = np.vstack((x, -1 * np.ones(5)))
    top    = np.vstack((x,  1 * np.ones(5)))
    left   = np.vstack((0.5 * np.ones(3), y[1:4]))   # exclude corners
    right  = np.vstack((2.5 * np.ones(3), y[1:4]))

    # Combine all and transpose to get shape (16, 2)
    X_selected = np.hstack((bottom, top, left, right)).T
    
elif args.init_pts_mode == 'bias':
    # METHOD 3: select 16 data points on a specified region.
    np.random.seed(0)

    # Generate 12 points near the origin (e.g., from a normal distribution with small std)
    near_origin = np.random.normal(loc=-0.3, scale=0.4, size=(12, 2))

    # Generate 4 points near the boundary [-5, 5]
    boundary_radius = 5
    angles = np.linspace(0, 2 * np.pi, 5)[:-1]  # 4 angles
    boundary_points = np.stack([
        boundary_radius * np.cos(angles),
        boundary_radius * np.sin(angles)
    ], axis=1)

    # Combine points
    X_selected = np.vstack([near_origin, boundary_points])


plt.plot(X_selected[:,0],X_selected[:,1],'*', markersize=10, color='red', label='Selected Points')
plt.grid(linestyle='--')
plt.xlabel('x1')
plt.ylabel('x')
plt.title('Selected Points for Initial Training')
plt.tight_layout()
plt.savefig(os.path.join('semi_supervison/DroneZ_MPC_weights', f'selected_points_{args.init_pts_mode}.png'))
plt.show()
plt.close()

# find the closest points in input_data_valid
input_data_valid = np.array(input_data_valid)
closest_indices = []
for point in X_selected:
    distances = np.linalg.norm(input_data_valid[:, :2] - point, axis=1)
    closest_index = np.argmin(distances)
    closest_indices.append(closest_index)

# select the corresponding label_data_valid
data_opt_set = input_data_valid[closest_indices].tolist()
label_opt_set = np.array(label_data_valid)[closest_indices].tolist()



T_start =  time.time()
# ================== NN Zero Training ==============

# convert to tensor
X_train_tensor = torch.Tensor(data_opt_set)
y_train_tensor = torch.Tensor(label_opt_set)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = P_Net(output_size=50).to(device)

lr_zero = 1e-3
epoch_NN0 = 50

optimizer = torch.optim.RMSprop(model.parameters(), lr=lr_zero)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_NN0, eta_min=lr_zero*0.07)

criterion = nn.MSELoss()

model.train()
for i in range(epoch_NN0):

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()



# ================= Generate another dataset ==============
global true_label_location, true_label_values
true_label_values = []
true_label_location = []

def data_update_set_function(model,device):
    x1 = np.linspace(0.5, 2.5, 200)
    x2 = np.linspace(-1, 1, 200)
    X = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
    # generate input data: [x1,x2,xr]: shape (1000000, 3)
    data_update_set = np.concatenate((X, np.full((X.shape[0], 1), x_r)), axis=1)
    data_update_set = torch.Tensor(data_update_set).to(device)


    model.eval()
    with torch.no_grad():
        # output order: [p1, p2, p3, u, theta]
        label_update_set = model(data_update_set).cpu().numpy()


    # update label with 16 data points, other points are set to zero
    # remember the location for future update
    global true_label_location, true_label_values
    if len(true_label_location) == 0:
        for i, point in enumerate(X):
            close_flag = np.all(np.isclose(point, X_selected,atol=0.01), axis=1)
            if np.any(close_flag):
                index = np.where(close_flag)[0][0]
                label_update_set[i] = label_opt_set[index]
                true_label_location.append(True)  # Store the index of the point
                true_label_values.append(label_opt_set[index])  # Store the label value

            else:
                true_label_location.append(False)  # Store the index of the point

    else:
        # use current true_label_location to update label_update_set
        label_update_set[true_label_location] = true_label_values

    return data_update_set, torch.tensor(label_update_set)


global diff_currentdata_optdata,w1_set, w2_set

w1_set = []
w2_set = []
diff_currentdata_optdata = []

def currentdataset_vs_optdataset(model,device):
    # give input x from opt dataset, find nn output:
    model.eval()
    with torch.no_grad():
        input_nn = torch.tensor(input_data_valid,dtype=torch.float32, device=device)
        nn_output = model(input_nn)

    opt_label = torch.tensor(label_data_valid,dtype=torch.float32, device=device)
    # loss = F.mse_loss(nn_output,opt_label,reduction='none')
    loss = torch.abs(nn_output-opt_label)

    # l_mean, l_std = torch.std_mean(loss,dim=0, keepdim=True)
    # loss_selected = loss[loss[:,3]<(l_mean[:,3]+0.5*l_std[:,3])]
    # loss_mean = loss_selected.mean(dim=0)

    # for MPC problem, we only consider the first u value:
    loss_mean = loss.mean(dim=0)[0]

    global diff_currentdata_optdata
    diff_currentdata_optdata.append(loss_mean.cpu().tolist())

    if len(diff_currentdata_optdata) != 0:
        if np.min(np.array(diff_currentdata_optdata)) == loss_mean.cpu().tolist():
            # save model
            torch.save(model.state_dict(), os.path.join('semi_supervison/DroneZ_MPC_weights','weight_best.pth'))



# ================= NN Training with above mixture Data ==============
# Note: Above data set will be updated every eopch

#  Define Loss Function and Optimizer
criterion = nn.MSELoss()
criterion2 = ObjectiveFormulation(device=device)


# Define a running average class to normalize losses
class RunningAverage:
    def __init__(self):
        self.avg = None

    def update(self, value):
        if self.avg is None:
            self.avg = value.detach()

        elif self.avg < value.detach():
            self.avg = value.detach()
        # else:
        
        #     self.avg = 0.9 * self.avg + 0.1 * value.detach()

        return value / (self.avg + 1e-8)



running_avg1 = RunningAverage()
running_avg2 = RunningAverage()

def loss_function(w1,w2,nn_inputs, outputs, targets,init_restart):
    loss1_1 = criterion(outputs[:,:3], targets[:,:3].to(device))
    loss1_2 = criterion(outputs[:,3], targets[:,3].to(device))
    loss1_3 = criterion(outputs[:,4], targets[:,4].to(device))
    loss1 = loss1_1 + 10*loss1_2 + loss1_3

    loss2 = criterion2.forward(nn_inputs.to(device), outputs.to(device))
    loss2 = loss2.mean()  # Ensure loss2 is a scalar
    # loss3 = criterion3_nonlinear_function(nn_inputs.to(device), outputs.to(device))


    # if init_restart==True:
    #     running_avg1.avg = loss1.detach()
    #     running_avg2.avg = loss2.detach()

    loss1 = running_avg1.update(loss1)
    loss2 = running_avg2.update(loss2)

    

    # Then clip if necessary
    # loss1 = torch.clamp(loss1, max=1.0)
    # loss2 = torch.clamp(loss2, max=1.0)
    # loss3 = torch.clamp(loss3, max=1.0)

    return w1*loss1, w2*loss2

    # return loss1, loss2, loss3


def test(model, test_loader,epoch,w1_tensor,w2_tensor):
    test_loss = 0.0
    u_losses = 0
    model.eval()
    init_restart = True
    with torch.no_grad():
        loop = tqdm(test_loader)
        for inputs, targets in loop:
            outputs = model(inputs.to(device))


            loss1, loss2 = loss_function(w1_tensor,w2_tensor,inputs, outputs,
                                          targets.to(device),init_restart)
            init_restart = False
            loss = loss1 + loss2


            if not 'cuda' in device.type:
                test_loss += loss.item() * inputs.size(0)
            else:
                test_loss += loss.cpu().item() * inputs.size(0)


            # loop.set_postfix(loss=f"{loss_set[-1]:.4f}", refresh=True)
    test_loss /= len(test_loader.dataset)

    # compare each element with dataset:
    currentdataset_vs_optdataset(model,device)

    global Loss_init, delta_Loss, Loss_old, w1_set, w2_set

    w1_set.append(w1_tensor.item())
    w2_set.append(w2_tensor.item())

    if epoch == 0:
        Loss_init = test_loss
        Loss_old = test_loss
        delta_Loss = 1
        print(f'Test Loss: {test_loss:.4f}| Delta Loss: {delta_Loss:.4f}')

    
    else:
        delta_Loss = abs(test_loss - Loss_old)/Loss_init
        print(f'Test Loss: {test_loss:.4f}| Delta Loss: {delta_Loss:.4f}')
        Loss_old = test_loss


    return delta_Loss, test_loss
    



    # # test PI
    # # test_PI, num_in = test_performance_index(model, device=device, xr=0.0)




batch_size = 4096
num_epochs = 10
total_iterations = 30



# define w1, w2 change values:
# w1 = np.linspace(0.1, 0.9, total_iterations)
# w1 = 0.125*np.linspace(0.5,2.7,total_iterations)**2
# w1 = 0.025*np.linspace(1,6.,10)**2
w1 = (0.6*torch.erf(torch.linspace(-3,0.001,10))+0.8).tolist()
w1.extend(10*[w1[-1]])
[w1.insert(0,w1[0]) for _ in range(10)]
w1 = np.array(w1)

# w1 = np.exp(np.linspace(-2.5,-0.01,total_iterations))

w2 = 1 - w1
train_loss_set = []
train_loss_set_separate = []
test_loss_set = []

for i in range(total_iterations):

    print(f'Iteration {i+1}/{total_iterations}, w1={w1[i]}, w2={w2[i]}')
    w1_tensor = torch.tensor(w1[i], dtype=torch.float32, device=device)
    w2_tensor = torch.tensor(w2[i], dtype=torch.float32, device=device)

    # update the data_update_set and label_update_set
    input_data_generated, label_generated = data_update_set_function(model, device)

    # split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(input_data_generated, label_generated, test_size=0.2, random_state=42)

    # create dataset and dataloader
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    # reset lr and optimizer
    # if i == 0:
    #     lr_iteration = lr_zero 
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr_iteration)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-1, eta_min=lr_iteration*0.9)



    epoch = 0

    # define terminal condition:
    global Loss_init, delta_Loss, Loss_old


    test_delta_Loss = 1
    init_restart = True

    # ============= run epochs given fixed w1, w2 =============
    while epoch<num_epochs:
    # while test_delta_Loss > 0.01 and epoch<num_epochs:

        model.train()



        loss_avg = [] # Reset loss_avg for each epoch
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            # output order: [p1, p2, p3, u, theta]
            outputs = model(inputs.to(device))
            
            # Calculate losses
            loss1, loss2 = loss_function(w1_tensor,w2_tensor,inputs, 
                                         outputs, targets.to(device),init_restart)
            init_restart = False

            loss = loss1 + loss2 

            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=100)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            # update lr and gradient
            optimizer.step()
            scheduler.step()
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}')


            loss_avg.append([loss1.item(), loss2.item()])  # Store the loss value for plotting

        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        train_loss_set_separate.append(np.sum(loss_avg,axis=0)*np.array([1/w1[i], 1/w2[i]]))
        train_loss_set.append(np.sum(loss_avg))
        

        # validation:
        test_delta_Loss,testloss = test(model, test_loader, epoch, w1_tensor,w2_tensor)
        test_loss_set.append(testloss)
        epoch += 1


    # update lr for new iterations
    # lr_iteration = scheduler.get_last_lr()[0]

T_end = time.time() - T_start
print(f'--------------Total Cost: {T_end}s----------------')
# plot training loss and testing loss
train_loss_set = np.array(train_loss_set)
test_loss_set = np.array(test_loss_set)
train_loss_set_separate = np.array(train_loss_set_separate)
x = np.arange(train_loss_set.shape[0])
plt.figure(figsize=(10, 6))

plt.plot(x, train_loss_set, label='Train Loss', color='red',marker='o',linewidth=2)
plt.plot(x, train_loss_set_separate[:,0], label='Train Loss1',linestyle='dashed', color='green')
plt.plot(x, train_loss_set_separate[:,1], label='Train Loss2',linestyle='dashed', color='orange')
plt.plot(x, test_loss_set, label='Test Loss', color='blue',marker='*',linewidth=2)
plt.grid(linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.savefig(os.path.join('semi_supervison/DroneZ_MPC_weights', 'train_test_loss.png'))
plt.show()
plt.close()

data_plot = np.array(diff_currentdata_optdata)
x = np.arange(data_plot.shape[0])
plt.subplot(311)
plt.plot(x,data_plot, label='error u', color='red',marker='o',linewidth=2)
plt.title('Difference between updating data and optimization data')
plt.ylabel('Diff')
plt.grid(linestyle='--')
plt.subplot(312)
plt.plot(x,np.array(w1_set), label='w1', color='blue',marker='*',linewidth=2)
plt.ylabel('w1')
plt.grid(linestyle='--')
plt.subplot(313)
plt.plot(x,np.array(w2_set), label='w2', color='green',marker='*',linewidth=2)
plt.ylabel('w2')
plt.grid(linestyle='--')

plt.xlabel('Iterations')
plt.legend()
plt.savefig(os.path.join('semi_supervison/DroneZ_MPC_weights', 'data_Diff.png'))
plt.show()
plt.close()


# test_PI, num_in = test_performance_index(model, device=device, xr=0.0, model_path=os.path.join('semi_supervison/DroneZ_MPC_weights','weight_best.pth'))


# model.load_state_dict(torch.load('semi_supervison/DroneZ_MPC_weights/weight_0.9_0.1.pth', map_location=device))
# PI, valid_cnt = test_performance_index(model, device=device, xr=0.0, model_path='semi_supervison/DroneZ_MPC_weights/weight_0.9_0.1.pth')
# print(f'Performance Index: {PI} | Valid Trajectories Count: {valid_cnt}')

data_opt_set, label_opt_set, trajectories_in_val, trajectories_label_val = dataloading_MPC('DroneZ_MPC/dataset/droneZ_MPC_16trajectory.csv')
test_trajectory_MPC(model, (trajectories_in_val, trajectories_label_val), device)


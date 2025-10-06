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
from test_converge_PI import  test_trajectory_MPC
from dataprocessing import DroneZ_Data_Purify, dataloading_MPC
from UpdatingDataset import UpdatingDataset, TruthAwareSampler
from loss_function import RunningAverage


# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

running_avg1 = RunningAverage()
running_avg2 = RunningAverage()
running_avg2.avg = torch.tensor([350]).to(device)



parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='DroneZ_MPC/dataset/drone_mpc_z.csv', help='corresponding theta dataset')
parser.add_argument('--lr',type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=4096, help='input batch size for training')
parser.add_argument('--sampling_mode', type=str, default='dense_center',help='uniform, dense_center, dense_boundary')
parser.add_argument('--total_iterations', type=int, default=10,help='total iterations for updating dataset and training')
parser.add_argument('--num_epochs', type=int, default=5,help='number of epochs for each iteration')
parser.add_argument('--pre_trained', type=str, default='',help='input your pretrained weight path if you want')
args = parser.parse_args()
print(args)


# specify save path
save_path = f'semi_supervison/DroneZ_MPC_weights'
os.makedirs(save_path, exist_ok=True)


# ============== Load Updating Dataset ==============
dataset = UpdatingDataset(mode=args.sampling_mode, sampling_num_per_xr=10)

data_opt_set, label_opt_set = dataset[dataset.TruthData_Mask]


T_start =  time.time()
# ================== NN Zero Training ==============

train_dataset_opt = TensorDataset(data_opt_set, label_opt_set)
train_loader_opt = DataLoader(train_dataset_opt, batch_size=32, shuffle=True)

model = P_Net(output_size=50).to(device)

lr_zero = args.lr
epoch_NN0 = 50

optimizer = torch.optim.RMSprop(model.parameters(), lr=lr_zero)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr_zero,momentum=0.9)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_NN0, eta_min=lr_zero*0.07)

criterion = nn.MSELoss()

model.train()
print('================== NN Zero Training ==================')
nn0_training_bar = tqdm(range(epoch_NN0))
for i in nn0_training_bar:

    for inputs, targets in train_loader_opt:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        nn0_training_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])


# update Updating Dataset with trained model (Except the optimal data points):
dataset.update_with_model(model, device=device)



# Iterative training with updated dataset
#  Define Loss Function and Optimizer
criterion = nn.MSELoss()
criterion2 = ObjectiveFormulation(device=device)


total_iterations = args.total_iterations

# define w1, w2 change values:
w1 = (-0.3*torch.erf(torch.linspace(-2.5,2.5,total_iterations))+0.5).tolist()
# w2.extend(10*[w2[-1]])
# [w2.insert(0,w2[0]) for _ in range(10)]
w1 = np.array(w1)

# # plot w1 change
# plt.figure()
# plt.plot(np.arange(len(w1)),w1)
# plt.xlabel('Iteration')
# plt.ylabel('w1 value')
# plt.title('w1 value change over iterations')
# plt.grid(linestyle='--')
# plt.show()
# plt.close()
w2 = 1 - w1

# set of losses without timing omega:
loss1_set = []
loss2_set = []

train_loss_set = []
train_loss_set_separate = []
test_loss_set = []
lr_set = []


# ================ Iteration Start ================
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr*0.95)

PI_x_bext = np.inf
for i in range(total_iterations):

    sampler = TruthAwareSampler(dataset, batch_size=args.batch_size, truth_indices=dataset.TruthData_Mask)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    
    print(f'Iteration {i+1}/{total_iterations}, w1={w1[i]:.2f}, w2={w2[i]:.2f}')
    w1_tensor = torch.tensor(w1[i], dtype=torch.float32, device=device)
    w2_tensor = torch.tensor(w2[i], dtype=torch.float32, device=device)


    # ============= run epochs given fixed w1, w2 =============
    train_bar = tqdm(range(args.num_epochs))
    for epoch in train_bar:
    # while test_delta_Loss > 0.01 and epoch<num_epochs:

        model.train()

        loss_avg = [] # Reset loss_avg for each epoch
        for inputs, targets in dataloader:

            optimizer.zero_grad()
            # output order: [p1, p2, p3, u, theta]
            outputs = model(inputs.to(device))


            loss1 = criterion(outputs, targets.to(device))

            loss2 = criterion2.forward(inputs.to(device), outputs.to(device))
            loss2 = loss2.mean()  # Ensure loss2 is a scalar

            # loss1 = w1[i]*running_avg1.update(loss1)
            # loss2 = w2[i]*running_avg2.update(loss2)
            # loss1 = w1[i]*(loss1/loss1.detach())
            # loss2 = w2[i]*(loss2/loss2.detach())

            loss =  w1[i]*(loss1/loss1.detach()) + w2[i]*(loss2/loss2.detach()) 

            loss.backward()
            optimizer.step()

            loss_avg.append([loss1.item(), loss2.item()])  # Store the loss value for plotting

            scheduler.step()

        lr_set.append(scheduler.get_last_lr()[0])
        # average loss1 and loss2 before timing omega:
        train_loss_set_separate.append((np.sum(loss_avg,axis=0)/len(loss_avg)))
        train_loss_set.append(np.sum(np.sum(loss_avg,axis=0)/len(loss_avg)))

    
        train_bar.set_postfix(train_loss=f"{train_loss_set_separate[-1][1]:.4f}", lr=scheduler.get_last_lr()[0])
       
        # validation: if the loss2 decreases, save models

    # update lr for new iterations
    # lr_iteration = scheduler.get_last_lr()[0]

    # test PI after each iteration
    PI_x, PI_u = dataset.test_performance_index(model, device=device, model_path=None)
    if PI_x < PI_x_bext:
        PI_x_bext = PI_x
        torch.save(model.state_dict(), os.path.join(save_path, f'weight_X{PI_x:.2f}_U{PI_u:.2f}.pth'))
        print(f'New best model saved with PI_x: {PI_x_bext}')

    # update the data_update_set and label_update_set
    dataset.update_with_model(model, device=device)

T_end = time.time() - T_start
print(f'--------------Total Cost: {T_end}s----------------')


# plot training loss and testing loss
train_loss_set = np.array(train_loss_set)
train_loss_set_separate = np.array(train_loss_set_separate)
x = np.arange(train_loss_set.shape[0])
loss1_set = train_loss_set_separate[:,0]
loss2_set = train_loss_set_separate[:,1]
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(x, loss1_set, label='Loss1 (MSE)', color='red',marker='o',linewidth=2)
plt.ylim(0, np.mean(loss1_set)+2*np.std(loss1_set))
plt.grid(linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Loss1')
plt.title('Training Loss1 (MSE)')
plt.legend()
plt.subplot(212)
plt.plot(x, loss2_set, label='Loss2 (MPC)', color='blue',marker='*',linewidth=2)
plt.ylim(0, np.mean(loss2_set)+2*np.std(loss2_set))
plt.grid(linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Loss2')
plt.title('Training Loss2 (MPC)')
plt.legend()
plt.savefig(os.path.join('semi_supervison/DroneZ_MPC_weights', 'train_loss.png'))
plt.show()
plt.close()















'''

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

    # ensure the optimal data points are included in the training set
    in_excluded = data_update_set[torch.logical_not(torch.tensor(true_label_location))]
    label_excluded = label_update_set[torch.logical_not(torch.tensor(true_label_location))]
    # split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(in_excluded, torch.tensor(label_excluded), test_size=0.2, random_state=42)

    # ensure the optimal data points are included in the training set
    X_train =torch.cat((X_train, torch.tensor(data_opt_set).to(device)), dim=0)
    y_train = np.concatenate((y_train, np.array(label_opt_set)), axis=0)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)

    return X_train, X_test, y_train, y_test


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
        
        self.memory_length = 30
        self.avg = torch.tensor([]).to(device)

    def update(self, value):

        if self.avg.shape[0]==0:
            self.avg = value.detach()
            self.avg = self.avg.reshape(1,1)
            self.alpha = 1.

        elif self.avg<value:
            # self.avg = torch.cat((self.avg,value.detach().reshape(1,1)),dim=0)
            self.avg = value.detach().reshape(1,1)

        # elif len(self.avg) == self.memory_length:
        #     self.avg,_ = torch.sort(self.avg,descending=True)
        #     # self.avg = self.avg[1:,:]

        # mean_val = torch.mean(self.avg)

        return value / (self.avg + 1e-8)



running_avg1 = RunningAverage()
running_avg2 = RunningAverage()

def loss_function(w1,w2,nn_inputs, outputs, targets,init_restart,mode='train'):
    loss1 = criterion(outputs, targets.to(device))
    # loss1_2 = criterion(outputs[:,3], targets[:,3].to(device))
    # loss1_3 = criterion(outputs[:,4], targets[:,4].to(device))
    # loss1 = loss1_1 + 10*loss1_2 + loss1_3

    loss2 = criterion2.forward(nn_inputs.to(device), outputs.to(device))
    loss2 = loss2.mean()  # Ensure loss2 is a scalar
    # loss3 = criterion3_nonlinear_function(nn_inputs.to(device), outputs.to(device))


    # if init_restart==True:
    #     running_avg1.avg = torch.tensor([]).to(device)
    #     running_avg2.avg = torch.tensor([]).to(device)

    if mode == 'train':
        loss1 = running_avg1.update(loss1)
        loss2 = running_avg2.update(loss2)

    elif mode=='test':
        # do not update in test
        loss1 /= torch.mean(running_avg1.avg)
        loss2 /= torch.mean(running_avg2.avg)

    

    # Then clip if necessary
    # loss1 = torch.clamp(loss1, max=1.0)
    # loss2 = torch.clamp(loss2, max=1.0)
    # loss3 = torch.clamp(loss3, max=1.0)

    return w1*loss1, running_avg2.alpha*w2*loss2

    # return loss1, loss2, loss3


def test(model, test_loader,epoch,w1_tensor,w2_tensor):
    test_loss = 0.0
    u_losses = 0
    model.eval()
    init_restart = False
    with torch.no_grad():
        loop = tqdm(test_loader)
        for inputs, targets in loop:
            outputs = model(inputs.to(device))


            loss1, loss2 = loss_function(w1_tensor,w2_tensor,inputs, outputs,
                                          targets.to(device),init_restart,mode='test')
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
num_epochs = 15
total_iterations = 30



# define w1, w2 change values:
w1 = (0.6*torch.erf(torch.linspace(-3,0.001,10))+0.8).tolist()
# w1 = (0.6*torch.erf(torch.linspace(-3,0.478,10))+0.69).tolist()
# w1 = (torch.erf(torch.linspace(-1.82,1.82,10))).tolist()
w1.extend(10*[w1[-1]])
[w1.insert(0,w1[0]) for _ in range(10)]
w1 = np.array(w1)


# Time range
# t = np.linspace(0, 30, total_iterations)
# # Sine function
# w1 = 0.3 * np.sin(2 * np.pi * t / 30 - np.pi / 2) + 0.5
# # w1 = np.exp(np.linspace(-2.5,-0.01,total_iterations))

w2 = 1 - w1
train_loss_set = []
train_loss_set_separate = []
test_loss_set = []
lr_set = []

gradient1_set1 = []
gradient1_set2 = []

for i in range(total_iterations):

    print(f'Iteration {i+1}/{total_iterations}, w1={w1[i]}, w2={w2[i]}')
    w1_tensor = torch.tensor(w1[i], dtype=torch.float32, device=device)
    w2_tensor = torch.tensor(w2[i], dtype=torch.float32, device=device)

    # update the data_update_set and label_update_set
    X_train, X_test, y_train, y_test = data_update_set_function(model, device)

    # create dataset and dataloader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    # reset lr and optimizer
    # if i == 0:
        # lr_iteration = lr_zero 

    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr_iteration)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_iteration*0.95)



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
            
            # add optimal data into mini-batch
            inputs = torch.cat((inputs, X_train_tensor_opt.to(device)), dim=0)
            targets = torch.cat((targets, y_train_tensor_opt.to(device)), dim=0)
            # shuffle the inputs and targets
            indices = torch.randperm(inputs.size(0))
            inputs = inputs[indices]
            targets = targets[indices]
            # print(f'inputs shape: {inputs.shape}, targets shape: {targets.shape}')


            optimizer.zero_grad()
            # output order: [p1, p2, p3, u, theta]
            outputs = model(inputs.to(device))
            
            # Calculate losses
            loss1, loss2 = loss_function(w1_tensor,w2_tensor,inputs, 
                                         outputs, targets.to(device),init_restart)
            init_restart = False

            loss = loss1 + loss2 


            # Measure gradients from loss1 (MSE)
            optimizer.zero_grad()
            loss1.backward(retain_graph=True)  # retain_graph only if you need the graph for loss2
            grad_norm1 = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)

            # Store norm for analysis/logging
            # print(f'Grad norm from Loss1 (MSE): {grad_norm1:.4f}')

            # Reset gradients before computing loss2 gradients
            optimizer.zero_grad()
            loss2.backward()
            grad_norm2 = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
            # print(f'Grad norm from Loss2 (MPC): {grad_norm2:.4f}')

            # (Optional) Now do the actual training step, combining both losses
            # optimizer.zero_grad()
            # combined_loss = loss1 +  loss2
            # combined_loss.backward()
            optimizer.step()
            # running_avg2.alpha = grad_norm1/grad_norm2

            # loss.backward()
            # # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=100)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            # # update lr and gradient
            # optimizer.step()
            # # print(f'Epoch [{epoch+1}/{num_epochs}], Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}')


            loss_avg.append([loss1.item(), loss2.item()])  # Store the loss value for plotting

            scheduler.step()

        lr_set.append(scheduler.get_last_lr()[0])
        gradient1_set1.append(grad_norm1.item())
        gradient1_set2.append(grad_norm2.item())
        # average loss1 and loss2 before timing omega:
        train_loss_set_separate.append((np.sum(loss_avg,axis=0)/len(loss_avg))*np.array([1/w1[i], 1/w2[i]]))
        train_loss_set.append(np.sum(np.sum(loss_avg,axis=0)/len(loss_avg)))
        

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

plt.figure(figsize=(10, 16))
data_plot = np.array(diff_currentdata_optdata)
x = np.arange(data_plot.shape[0])
plt.subplot(611)
plt.plot(x,data_plot, label='error u', color='red',marker='o',linewidth=2)
plt.title('Difference between updating data and optimization data')
plt.ylabel('Diff')
plt.grid(linestyle='--')
plt.subplot(612)
plt.plot(x,np.array(w1_set), label='w1', color='blue',marker='*',linewidth=2)
plt.ylabel('w1')
plt.grid(linestyle='--')
plt.subplot(613)
plt.plot(x,np.array(w2_set), label='w2', color='green',marker='*',linewidth=2)
plt.ylabel('w2')
plt.grid(linestyle='--')
plt.legend()


plt.subplot(614)
x = range(len(gradient1_set1))
plt.plot(np.array(x),np.array(gradient1_set1), label='gradient1', color='purple',linewidth=2)
plt.ylabel('grad1')
plt.grid(linestyle='--')
plt.legend()

plt.subplot(615)
plt.plot(np.array(x),np.array(gradient1_set2), label='gradient2', color='orange',linewidth=2)
plt.ylabel('grad2')
plt.grid(linestyle='--')
plt.legend()




plt.subplot(616)
x = range(len(lr_set))
plt.plot(np.array(x),np.array(lr_set),label='learning rate', color='red',linewidth=2)
plt.grid(linestyle='--')
plt.ylabel('lr')
plt.xlabel('Iterations')
plt.legend()
plt.savefig(os.path.join('semi_supervison/DroneZ_MPC_weights', 'data_Diff.png'))
plt.show()
plt.close()


# test_PI, num_in = test_performance_index(model, device=device, xr=0.0, model_path=os.path.join('semi_supervison/DroneZ_MPC_weights','weight_best.pth'))


# model.load_state_dict(torch.load('semi_supervison/DroneZ_MPC_weights/weight_0.9_0.1.pth', map_location=device))
# PI, valid_cnt = test_performance_index(model, device=device, xr=0.0, model_path='semi_supervison/DroneZ_MPC_weights/weight_0.9_0.1.pth')
# print(f'Performance Index: {PI} | Valid Trajectories Count: {valid_cnt}')

x_trajectory, trajectories_in_val, trajectories_label_val = dataloading_MPC('DroneZ_MPC/dataset/droneZ_MPC_16trajectory.csv')
test_trajectory_MPC(model, (x_trajectory, trajectories_label_val), device)
'''
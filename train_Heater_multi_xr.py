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
from Heater_Dataset import Data_Heater_Collected, UpdatingDataset, TruthAwareSampler



# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")




parser = argparse.ArgumentParser()
parser.add_argument('--opt_dataset',type=str,default='mpc_data_heater.csv', help='corresponding theta dataset')
parser.add_argument('--lr',type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=4096, help='input batch size for training')
parser.add_argument('--sampling_num_per_xr', type=int, default=50,help='number of samples per reference trajectory for updating dataset')
parser.add_argument('--sampling_mode', type=str, default='uniform',help='uniform, dense_center, dense_boundary')
parser.add_argument('--omega_mode',type=str, default='vshape', help='omega changing mode:constant, linear, erf, vshape')
parser.add_argument('--total_iterations', type=int, default=40,help='total iterations for updating dataset and training')
parser.add_argument('--num_epochs', type=int, default=10,help='number of epochs for each iteration')
parser.add_argument('--pre_trained', type=str, default='',help='input your pretrained weight path if you want')
args = parser.parse_args()
print(args)


# specify save path
save_path = f'Heater_Results'
os.makedirs(save_path, exist_ok=True)

# ====== Load Modle and Optimal Dataset========
model = P_Net(output_size=30).to(device)

# data_tmp = Data_Heater_Collected(args.opt_dataset)
# opt_dataset = data_tmp.return_data()


# ============== Load Updating Dataset ==============
dataset = UpdatingDataset(mode=args.sampling_mode, 
                          sampling_num_per_xr=args.sampling_num_per_xr,
                          opt_dataset_path=args.opt_dataset)

data_opt_set, label_opt_set = dataset[dataset.TruthData_Mask]


# Evaluate Saved model
if args.pre_trained != '':
    model.load_state_dict(torch.load(args.pre_trained, map_location=device,weights_only=True))
    print(f'Pre-trained model {args.pre_trained} loaded!')

    # test PI after each iteration
    PI_x, PI_u, u_violation = dataset.test_performance_index(model, device=device, model_path=None)
    
    # validate current dataset vs optimal dataset
    # diff_opt = currentdataset_vs_optdataset(model,device,opt_dataset)
    # print(f'Current dataset vs Optimal dataset MSE: {diff_opt}')
    # print(f'Pre-trained model PI_x: {PI_x}, PI_u: {PI_u}')
    # print('---------------------End-------------------------')
    # exit()



T_start =  time.time()
# ================== NN Zero Training ==============

train_dataset_opt = TensorDataset(data_opt_set, label_opt_set)
train_loader_opt = DataLoader(train_dataset_opt, batch_size=32, shuffle=True)


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
if args.omega_mode == 'linear':
    # Method 1: linear change
    w1 = np.linspace(1.0, 0., total_iterations)
elif args.omega_mode == 'erf':
    # Method 2: erf change
    w1 = (-0.5*torch.erf(torch.linspace(-2.5,2.5,total_iterations))+0.5).tolist()
elif args.omega_mode == 'vshape':
    # Method 3: vshape change
    half_iter = total_iterations // 2
    w1_half = np.linspace(1.0, 0., half_iter)
    if total_iterations % 2 == 0:
        w1 = np.concatenate((w1_half, w1_half[::-1]))
    else:
        w1 = np.concatenate((w1_half, [0.5], w1_half[::-1]))
elif args.omega_mode == 'constant':
    # Method 4: constant change
    w1 = 1.0 * np.ones(total_iterations)
# # w2.extend(10*[w2[-1]])
# # [w2.insert(0,w2[0]) for _ in range(10)]
w1 = np.array(w1)

w2 = 1 - w1

# # plot w1 change
# plt.subplot(211)
# plt.plot(np.arange(w1.shape[0]),w1, label='w1',marker='o',linewidth=2)
# plt.xlabel('Iteration')
# plt.ylabel(r'$\omega_1$ value')
# plt.title(r'$\omega$ value change over iterations')
# plt.grid(linestyle='--')
# # plot w2 change
# plt.subplot(212)
# plt.plot(np.arange(w2.shape[0]),w2, label='w2',color='orange', marker='*',linewidth=2)
# plt.xlabel('Iteration')
# plt.ylabel(r'$\omega_2$ value')
# plt.grid(linestyle='--')
# plt.tight_layout()
# plt.savefig(os.path.join(save_path, f'omega_change_{args.omega_mode}.png'), dpi=300)
# plt.show()
# plt.close()

# set of losses without timing omega:
loss1_set = []
loss2_set = []

train_loss_set = []
train_loss_set_separate = []
test_loss_set = []
lr_set = []


# ================ Iteration Start ================
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(total_iterations*args.num_epochs), eta_min=args.lr*0.1)

PI_x_bext = np.inf
u_violation_best = np.inf
PI_u_bext = np.inf
diff_optdata_nnout = []
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


            loss1_1  = 10 * criterion(outputs[:,0], targets[:,0].to(device)) 
            
            loss1_2 = criterion(outputs[:,1:], targets[:,1:].to(device))
            loss1 = loss1_1 + loss1_2

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
    PI_x, PI_u, u_violation = dataset.test_performance_index(model, device=device, model_path=None)
    



    # validate current dataset vs optimal dataset
    # diff_opt = currentdataset_vs_optdataset(model,device,opt_dataset)
    # print(f'Current dataset vs Optimal dataset MSE: {diff_opt}')
    # diff_optdata_nnout.append(diff_opt)

    
    if PI_x < PI_x_bext:
        PI_x_bext = PI_x
        PI_u_bext = PI_u
        u_violation_best = u_violation
        torch.save(model.state_dict(), 
                   os.path.join(save_path, f'{args.sampling_mode}_S{args.sampling_num_per_xr}_{args.omega_mode}_Iter{args.total_iterations}_Epoch{args.num_epochs}.pth'))
        print(f'New best model saved with PI_x: {PI_x_bext}, PI_u: {PI_u_bext} at iteration {i+1}')

    # update the data_update_set and label_update_set
    dataset.update_with_model(model, device=device)

T_end = time.time() - T_start
print(f'--------------Total Cost: {T_end}s----------------')
print(f'The best PI_x: {PI_x_bext:.3f}, PI_u: {PI_u_bext:.3f}, U violation: {u_violation_best:.3f}')
print('---------------Training Finished-------------------')

# plot training loss and testing loss
train_loss_set = np.array(train_loss_set)
train_loss_set_separate = np.array(train_loss_set_separate)
x = np.arange(train_loss_set.shape[0])
loss1_set = train_loss_set_separate[:,0]
loss2_set = train_loss_set_separate[:,1]
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(x, loss1_set, label='Loss1 (MSE)', color='red',marker='o',linewidth=2)
plt.ylim(0, np.mean(loss1_set)+3*np.std(loss1_set))
plt.grid(linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Loss1')
plt.title('Training Loss1 (MSE)')
plt.legend()
plt.subplot(212)
plt.plot(x, loss2_set, label='Loss2 (MPC)', color='blue',marker='*',linewidth=2)
plt.ylim(150, np.mean(loss2_set)+3*np.std(loss2_set))
plt.grid(linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Loss2')
plt.title('Training Loss2 (MPC)')
plt.legend()
plt.savefig(os.path.join(save_path, 'train_loss.png'))
plt.show()
plt.close()

# plot difference between current dataset and optimal dataset
plt.figure()
plt.plot(np.arange(len(diff_optdata_nnout)), diff_optdata_nnout, marker='o', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Difference')
plt.title('Difference between Current Dataset and Optimal Dataset')
plt.grid(linestyle='--')
plt.savefig(os.path.join(save_path, 'diff_current_optimal_dataset.png'), dpi=300)















# '''


# # test_PI, num_in = test_performance_index(model, device=device, xr=0.0, model_path=os.path.join('semi_supervison/DroneZ_MPC_weights','weight_best.pth'))


# # model.load_state_dict(torch.load('semi_supervison/DroneZ_MPC_weights/weight_0.9_0.1.pth', map_location=device))
# # PI, valid_cnt = test_performance_index(model, device=device, xr=0.0, model_path='semi_supervison/DroneZ_MPC_weights/weight_0.9_0.1.pth')
# # print(f'Performance Index: {PI} | Valid Trajectories Count: {valid_cnt}')

# x_trajectory, trajectories_in_val, trajectories_label_val = dataloading_MPC('DroneZ_MPC/dataset/droneZ_MPC_16trajectory.csv')
# test_trajectory_MPC(model, (x_trajectory, trajectories_label_val), device)
# '''
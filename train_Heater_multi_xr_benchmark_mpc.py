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
from Heater_Dataset import MPCDataset, UpdatingDataset


# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")



parser = argparse.ArgumentParser()
parser.add_argument('--lr',type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=4096, help='input batch size for training')
parser.add_argument('--num_epochs', type=int, default=1000,help='number of epochs for each iteration')
parser.add_argument('--pre_trained', type=str, default='',help='input your pretrained weight path if you want')
args = parser.parse_args()
print(args)


# specify save path
save_path = f'Heater_Results'
os.makedirs(save_path, exist_ok=True)


# ============== Load Updating Dataset ==============
dataset = MPCDataset()
testdataset = UpdatingDataset(mode='uniform', 
                              sampling_num_per_xr=10,
                              opt_dataset_path='mpc_data_heater.csv')

T_start =  time.time()
# ================== NN Training ==============

train_loader_opt = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

model = P_Net(output_size=30).to(device)

lr_zero = args.lr
epoch_NN0 = args.num_epochs

optimizer = torch.optim.RMSprop(model.parameters(), lr=lr_zero)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr_zero,momentum=0.9)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_NN0, eta_min=lr_zero*0.07)

criterion = ObjectiveFormulation(device=device)


model.train()
print('================== NN Zero Training ==================')
nn0_training_bar = tqdm(range(epoch_NN0))
train_loss_set = []
PI_x_bext = 1e10
for i in nn0_training_bar:
    loss_tmp = []
    if i==0:
        lowest_loss = 1e10
    for inputs in train_loader_opt:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        loss = criterion.forward(inputs.to(device), outputs.to(device))
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        nn0_training_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        loss_tmp.append(loss.item())

    # # save model when loss is small
    # if i>3 and np.mean(loss_tmp)<lowest_loss:
    #     lowest_loss = np.mean(loss_tmp)
    #     torch.save(model.state_dict(), f'{save_path}/NN_benchmark_MPCLoss.pth')
    #     print(f"Model saved at epoch {i}, loss: {lowest_loss:.6f}")


    # test PI and save model
    PI_x, PI_u, u_violation = testdataset.test_performance_index(model, device=device, model_path=None)
    if PI_x < PI_x_bext:
        PI_x_bext = PI_x
        PI_u_bext = PI_u
        torch.save(model.state_dict(), os.path.join(save_path, f'Benchmark-MPC.pth'))
        print(f'New best model saved with PI_x: {PI_x_bext:.3f}, PI_u: {PI_u:.3f}, violation: {u_violation} at epoch {i}')
    
    train_loss_set.append(np.mean(loss_tmp))

T_end = time.time()
print(f"NN training time: {T_end - T_start:.2f} seconds")
print(f'The lowest training loss: {min(train_loss_set):.3f}')
print(f'The best PI_x: {PI_x_bext:.3f}, PI_u: {PI_u_bext:.3f}')

# plot loss curve
plt.figure()
plt.semilogy(np.arange(len(train_loss_set)), train_loss_set)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Curve')
plt.grid()
plt.savefig(f'{save_path}/training_loss_curve_MPC.png', dpi=300)
plt.show()

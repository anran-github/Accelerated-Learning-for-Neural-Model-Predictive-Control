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
from UpdatingDataset import MPCDataset, UpdatingDataset
from loss_function import RunningAverage

running_avg1 = RunningAverage()
running_avg2 = RunningAverage()



# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")



parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='semi_supervison/dataset/drone_mpc_z_multi_ref.csv', help='corresponding theta dataset')
parser.add_argument('--lr',type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=4096, help='input batch size for training')
parser.add_argument('--num_epochs', type=int, default=100,help='number of epochs for each iteration')
parser.add_argument('--pre_trained', type=str, default='',help='input your pretrained weight path if you want')
args = parser.parse_args()
print(args)


# specify save path
save_path = f'semi_supervison/DroneZ_MPC_weights'
os.makedirs(save_path, exist_ok=True)


# ============== Load Updating Dataset ==============
with open(args.dataset, 'r') as f:
    dataset = pd.read_csv(f)

input_data, label_data = dataset.iloc[:,:3].values, dataset.iloc[:,3:].values
training_dataset = TensorDataset(torch.tensor(input_data, dtype=torch.float32), torch.tensor(label_data, dtype=torch.float32))
train_loader_opt = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)



testdataset = UpdatingDataset(mode='dense_center', sampling_num_per_xr=10)



T_start =  time.time()
# ================== NN Training ==============


model = P_Net(output_size=50).to(device)

lr_zero = args.lr
epoch_NN0 = args.num_epochs

optimizer = torch.optim.RMSprop(model.parameters(), lr=lr_zero)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr_zero,momentum=0.9)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_NN0, eta_min=lr_zero*0.07)

criterion = nn.MSELoss()


model.train()
print('================== NN Zero Training ==================')
nn0_training_bar = tqdm(range(epoch_NN0))
train_loss_set = []
PI_x_bext = 1e10
for i in nn0_training_bar:
    loss_tmp = []
    if i==0:
        lowest_loss = 1e10
    for inputs, labels in train_loader_opt:

        optimizer.zero_grad()

        outputs = model(inputs.to(device))
        loss = criterion.forward(outputs.to(device), labels.to(device))

        loss.backward()
        optimizer.step()
        scheduler.step()
        nn0_training_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        loss_tmp.append(loss.item())



    # test PI and save model
    PI_x, PI_u = testdataset.test_performance_index(model, device=device, model_path=None)
    if PI_x < PI_x_bext:
        PI_x_bext = PI_x
        PI_u_bext = PI_u
        torch.save(model.state_dict(), os.path.join(save_path, f'Benchmark-MSE.pth'))
        print(f'New best model saved with PI_x: {PI_x_bext:.3f}, PI_u: {PI_u:.3f} at epoch {i}')
    
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
plt.savefig(f'{save_path}/training_loss_curve.png', dpi=300)
plt.show()

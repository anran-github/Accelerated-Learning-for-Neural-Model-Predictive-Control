import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.optim import lr_scheduler

from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse
import ast

# import NN structure:
from network import P_Net

from Objective_Formulations_new import ObjectiveFormulation
from test_converge_PI import test_performance_index
from dataprocessing import Data_Purify


# define a nonlinear function
def criterion3_nonlinear_function(x, data):
    # data: [p1, p2, p3, u, theta]
    x1 = x[:, 0]
    x2 = x[:, 1]
    
    u = data[:, 3]
    
    dt = 0.1

    x1_new = x1 + dt * x2
    x2_new = x2 + dt * (x1**3+(x2**2+1)*u)
    x = torch.stack((x1_new, x2_new), dim=1)

    return torch.norm(x, dim=1).mean()  # Return the norm of the new state vector


def test_visualization(model, test_loader):
    '''
    Test the model and visualize the results.
    :param model: The trained model.
    :param test_loader: DataLoader for the test dataset.
    :return: None
    '''
    model.eval()
    data_input = []
    data_output = []
    data_label = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(device))
            data_input.append(inputs.numpy())
            data_output.append(outputs.cpu().numpy())
            data_label.append(targets.numpy())
    data_input = np.concatenate(data_input, axis=0)
    data_output = np.concatenate(data_output, axis=0)
    data_label = np.concatenate(data_label, axis=0)

    # Plot the results
    x = data_input[:, 0]
    bar_width = 0.08  # small width to prevent overlap
    alpha = 0.5       # transparency

    plt.figure(figsize=(12, 8))

    # p1
    plt.subplot(221)
    plt.bar(x , data_label[:, 0], width=bar_width, color='r', alpha=alpha, label='Label p1')
    plt.bar(x , data_output[:, 0], width=bar_width, color='b', alpha=alpha, label='Output p1')
    plt.title('p1 vs Input x1')
    plt.xlabel('Input x1')
    plt.ylabel('p1')
    plt.legend()

    # p2
    plt.subplot(222)
    plt.bar(x , data_label[:, 1], width=bar_width, color='r', alpha=alpha, label='Label p2')
    plt.bar(x , data_output[:, 1], width=bar_width, color='b', alpha=alpha, label='Output p2')
    plt.title('p2 vs Input x1')
    plt.xlabel('Input x1')
    plt.ylabel('p2')
    plt.legend()

    # p3
    plt.subplot(223)
    plt.bar(x , data_label[:, 2], width=bar_width, color='r', alpha=alpha, label='Label p3')
    plt.bar(x , data_output[:, 2], width=bar_width, color='b', alpha=alpha, label='Output p3')
    plt.title('p3 vs Input x1')
    plt.xlabel('Input x1')
    plt.ylabel('p3')
    plt.legend()

    # u
    plt.subplot(224)
    plt.bar(x , data_label[:, 3], width=bar_width, color='r', alpha=alpha, label='Label u')
    plt.bar(x , data_output[:, 3], width=bar_width, color='b', alpha=alpha, label='Output u')
    plt.title('u vs Input x1')
    plt.xlabel('Input x1')
    plt.ylabel('u')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'test_visualization.png'))
    plt.show()    








# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")


# Parse the command-line arguments

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='semi_supervison/dataset/MM_DiffSys_dataset.csv', help='corresponding theta dataset')
parser.add_argument('--reference',type=float,default=0, help='reference of dataset')
parser.add_argument('--lr',type=float, default=5e-4, help='learning rate')
parser.add_argument('--pre_trained', type=str, default='',help='input your pretrained weight path if you want')
args = parser.parse_args()
print(args)

save_path = f'./weights'
os.makedirs(save_path, exist_ok=True)
x_r = np.array([args.reference])
data_tmp = Data_Purify(args.dataset,x_r,p_max=10,u_max=1e5)


# ============== Import and Purify Data ==============
# # import dataset 
input_data_valid,label_data_valid = data_tmp.purified_data()


# select 16 data points uniformly:
x1 = np.linspace(-5, 5, 4)
x2 = np.linspace(-5, 5, 4)
X_selected = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
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




# ================== NN Zero Training ==============

# update input_data_valid and label_data_valid
input_data_valid = data_opt_set
label_data_valid = label_opt_set

# convert to tensor
X_train_tensor = torch.Tensor(input_data_valid)
y_train_tensor = torch.Tensor(label_data_valid)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = P_Net(output_size=5).to(device)

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
global true_label_location
true_label_location = []

def data_update_set_function(model,device):
    x1 = np.linspace(-5, 5, 200)
    x2 = np.linspace(-5, 5, 200)
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
    global true_label_location
    if len(true_label_location) == 0:
        for i, point in enumerate(X):
            close_flag = np.all(np.isclose(point, X_selected,atol=0.02), axis=1)
            if np.any(close_flag):
                index = np.where(close_flag)[0][0]
                label_update_set[i] = label_opt_set[index]
                true_label_location.append(True)  # Store the index of the point
            else:
                true_label_location.append(False)  # Store the index of the point

    else:
        # use current true_label_location to update label_update_set
        label_update_set[true_label_location] = np.array([label_opt_set])

    return data_update_set, torch.tensor(label_update_set)




# ================= NN Training with above mixture Data ==============
# Note: Above data set will be updated every eopch

#  Define Loss Function and Optimizer
criterion = nn.MSELoss()
criterion2 = ObjectiveFormulation(x_r=x_r,device=device)


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

    global Loss_init, delta_Loss, Loss_old
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
num_epochs = 20
total_iterations = 10



# define w1, w2 change values:
w1 = np.linspace(0.1, 0.9, total_iterations)
w2 = np.linspace(0.9, 0.1, total_iterations)
train_loss_set = []
train_loss_set_separate = []
test_loss_set = []

for i in range(total_iterations):

    print(f'Iteration {i+1}/{total_iterations}, w1={w1[i]}, w2={w2[i]}')
    w1_tensor = torch.tensor(w1[i], dtype=torch.float32, device=device)
    w2_tensor = torch.tensor(w2[i], dtype=torch.float32, device=device)

    # update the data_update_set and label_update_set
    input_data_valid, label_data_valid = data_update_set_function(model, device)

    # split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(input_data_valid, label_data_valid, test_size=0.2, random_state=42)

    # create dataset and dataloader
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}')


            loss_avg.append([loss1.item(), loss2.item()])  # Store the loss value for plotting

        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        train_loss_set_separate.append(np.sum(loss_avg,axis=0)*np.array([1/w1[i], 1/w2[i]]))
        train_loss_set.append(np.sum(loss_avg))
        

        # validation:
        test_delta_Loss,testloss = test(model, test_loader, epoch, w1_tensor,w2_tensor)
        test_loss_set.append(testloss)
        epoch += 1


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
plt.savefig(os.path.join('semi_supervison/weights', 'train_test_loss.png'))
plt.show()
plt.close()


# save model
torch.save(model.state_dict(), os.path.join('semi_supervison/weights','weight_{}_{}.pth'.format(w1[i], w2[i])))
test_PI, num_in = test_performance_index(model, device=device, xr=0.0, model_path=os.path.join('semi_supervison/weights','weight_{}_{}.pth'.format(w1[i], w2[i])))



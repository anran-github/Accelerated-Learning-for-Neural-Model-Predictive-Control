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


# define a nonlinear function
def criterion3_nonlinear_function(x, data):
    # data: [p1, p2, p3, u, theta]
    x1 = x[:, 0]
    x2 = x[:, 1]
    
    u = data[:, 3]
    
    dt = 0.1

    x1_new = x1 + dt * x2
    x2_new = x2 + dt * dt*(x1**3+(x2**2+1)*u)
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




# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")


# Parse the command-line arguments

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='./dataset/MM_DiffSys_dataset.csv', help='corresponding theta dataset')
parser.add_argument('--reference',type=float,default=0, help='reference of dataset')
parser.add_argument('--lr',type=float, default=5e-4, help='learning rate')
parser.add_argument('--seting_weight',type=list, default=[1,0.,0.0], help='weight for loss function, [MSE, Objective, Nonlinear]')
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



# =============== Add Noise to Label Data ============== 
# input_data_valid,label_data_tmp = data_tmp.purified_data()


# middle noise:


# low noise: 2% uniform noise
# np.random.seed(42)  # For reproducibility
# noise = np.random.uniform(-0.02, 0.02, size=(len(label_data_tmp), 5)) 
# label_data_valid = np.array(label_data_tmp)*noise + np.array(label_data_tmp)

# label_data_valid  = label_data_valid.tolist()



# data_tmp.draw_data(data_tmp.data_u,data_tmp.data_p)




X_train, X_test, y_train, y_test = train_test_split(input_data_valid, label_data_valid, test_size=0.2, random_state=42)

X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 4096
num_epochs = 600

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





model = P_Net(output_size=5).to(device)

if len(args.pre_trained):
    model.load_state_dict(torch.load(args.pre_trained))
    model.eval()
    print('----------added previous weights: {}------------'.format(args.pre_trained))



# ======== Define Loss Function and Optimizer ==============
designed_weights = args.seting_weight  # MSE, Objective, Nonlinear

criterion = nn.MSELoss()
criterion2 = ObjectiveFormulation(x_r=x_r,device=device)


# Define a running average class to normalize losses
class RunningAverage:
    def __init__(self):
        self.avg = None

    def update(self, value):
        if self.avg is None:
            self.avg = value.detach()

        # elif self.avg < value.detach():
        #     self.avg = value.detach()
        # else:
        
        #     self.avg = 0.9 * self.avg + 0.1 * value.detach()
        return value / (self.avg + 1e-8)

running_avg1 = RunningAverage()
running_avg2 = RunningAverage()
running_avg3 = RunningAverage()

def loss_function(nn_inputs, outputs, targets):
    loss1_1 = criterion(outputs[:,:3], targets[:,:3].to(device))
    loss1_2 = criterion(outputs[:,3], targets[:,3].to(device))
    loss1_3 = criterion(outputs[:,4], targets[:,4].to(device))
    loss1 = loss1_1 + 10*loss1_2 + loss1_3

    loss2 = criterion2.forward(nn_inputs.to(device), outputs.to(device))
    loss3 = criterion3_nonlinear_function(nn_inputs.to(device), outputs.to(device))

    # Normalize losses using running averages
    loss1 = running_avg1.update(loss1)
    loss2 = running_avg2.update(loss2)
    # if loss2 < 0:
    #     loss2 = torch.abs(loss2)
        
    loss3 = running_avg3.update(loss3)



    # Then clip if necessary
    # loss1 = torch.clamp(loss1, max=1.0)
    # loss2 = torch.clamp(loss2, max=1.0)
    # loss3 = torch.clamp(loss3, max=1.0)

    return designed_weights[0]*loss1, designed_weights[1]*loss2, designed_weights[2]*loss3 

    # return loss1, loss2, loss3



optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
# optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs//20, eta_min=args.lr*0.07)

# '''


global lowest_loss, lowest_PI, highest_in, lowest_PI_num_ratio
lowest_loss = np.inf
lowest_PI   = np.inf
highest_in  = 0
lowest_PI_num_ratio = 1000

def test(model, test_loader,epoch):
    test_loss = 0.0
    u_losses = 0
    model.eval()
    with torch.no_grad():
        loop = tqdm(test_loader)
        for inputs, targets in loop:
            outputs = model(inputs.to(device))


            loss1, loss2, loss3 = loss_function(inputs, outputs, targets)
            loss = loss1 + loss2 + loss3


            if not 'cuda' in device.type:
                test_loss += loss.item() * inputs.size(0)
            else:
                test_loss += loss.cpu().item() * inputs.size(0)

            # loop.set_postfix(loss=f"{loss_set[-1]:.4f}", refresh=True)
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')


    # test PI
    test_PI, num_in = test_performance_index(model, device=device, xr=0.0)

    # save model
    # Save the model
    global lowest_loss, lowest_PI, highest_in, lowest_PI_num_ratio
    # if lowest_loss>test_loss:
    #     lowest_loss = test_loss
    lowest_PI_num_ratio 

    if lowest_PI_num_ratio > (test_PI/num_in):

        
        print('Model Saved!')
        # torch.save(model.state_dict(), os.path.join(
        #     save_path,'no_noise.pth'))
        torch.save(model.state_dict(), os.path.join(
            save_path,'no_noise_weight_{}_{}_{}.pth'.format(
                designed_weights[0], designed_weights[1], designed_weights[2])))
        
        lowest_PI = test_PI
        highest_in = num_in
        lowest_PI_num_ratio = test_PI/num_in


# test(model, test_loader,epoch=0)


# '''


losses = []

model.train()


init_epoch = 0

for epoch in range(init_epoch,int(init_epoch+num_epochs)):
    loss_avg = [] # Reset loss_avg for each epoch
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        # output order: [p1, p2, p3, u, theta]
        outputs = model(inputs.to(device))
        
        # Calculate losses
        loss1, loss2, loss3 = loss_function(inputs, outputs, targets.to(device))

        loss = loss1 + loss2 + loss3 

        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=100)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        # update lr and gradient
        optimizer.step()
        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, Loss3: {loss3.item():.4f}')


        loss_avg.append([loss1.item(), loss2.item(), loss3.item()])  # Store the loss value for plotting

    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    losses.append(np.sum(loss_avg,axis=0))

    if epoch >= 1 and epoch%2==0:
        test(model, test_loader,epoch)


# torch.save(model.state_dict(), os.path.join(
#             save_path,'model_epoch_last.pth'))


# plot three losses:
plt.figure(figsize=(12, 8))
plt.clf()  # Clear previous plot
plt.subplot(2, 2, 1)
plt.plot(np.array(losses)[:, 0], label='MSE LOSS')
plt.grid(linestyle='--', color='gray')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(np.array(losses)[:, 1], label='Objective Loss')
plt.grid(linestyle='--', color='gray')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(np.array(losses)[:, 2], label='Nonlinear Loss')
plt.grid(linestyle='--', color='gray')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(np.sum(np.array(losses), axis=1), label='Total Loss')
plt.grid(linestyle='--', color='gray')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_path,'training_weight_{}_{}_{}.png'.format(designed_weights[0], designed_weights[1], designed_weights[2])))
plt.show()


# test_visualization(model, test_loader)
test_performance_index(model, device=device, xr=0.0, model_path=os.path.join(save_path,'no_noise_weight_{}_{}_{}.pth'.format(designed_weights[0], designed_weights[1], designed_weights[2])))





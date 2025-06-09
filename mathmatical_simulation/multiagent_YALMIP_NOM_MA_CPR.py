'''
This file does:
 !! Generate TABLE 1 
Given 101x101 fix points:
Check the voilateion rate, control effort:
NOM & YALMIP
'''



import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time 
import ast
# from network_ori import P_Net
from network import P_Net



# ===============IMPORT MATLAB DATA===============
        # % =========DATA FORMAT===========
        # % x1 p1 p2 u 
        # % x2 p2 p3 problem
        # % ===============================

# show label data

# EXCHANGE BETWEEN YALMP & MULTI-AGENT
df_ma = pd.read_csv('dataset/MM_DiffSys_dataset.csv', header=None)
df_yalmip = pd.read_csv('dataset/DiffSys_01theta_10201points_compare_YALMIP.csv', header=None)

def read_csvdata(df):
    # Assuming the first row is the header, adjust if needed
    data = df.values  


    data_x1_set = []
    data_x2_set = []
    data_p1_set = []
    data_p2_set = []
    data_p3_set = []
    data_u_set = []
    data_problem_set = []

    for row in range(data.shape[0]//2):
        tmp_x1 = [data[2*row,i] for i in range(data.shape[1]) if i%4==0]
        tmp_x2 = [data[2*row+1,i] for i in range(data.shape[1]) if i%4==0]
        tmp_data_p1 = [data[2*row,i+1] for i in range(data.shape[1]) if i%4==0]
        tmp_data_p2 = [data[2*row,i+2] for i in range(data.shape[1]) if i%4==0]
        tmp_data_p3 = [data[2*row+1,i+2] for i in range(data.shape[1]) if i%4==0]
        tmp_data_u = [data[2*row,i+3] for i in range(data.shape[1]) if i%4==0]
        tmp_data_problem = [data[2*row+1,i+3] for i in range(data.shape[1]) if i%4==0]

        # tmp_data_u = np.array(data_u)
        data_x1_set.extend(tmp_x1)
        data_x2_set.extend(tmp_x2)
        data_u_set.extend(tmp_data_u)
        data_p1_set.extend(tmp_data_p1)
        data_p2_set.extend(tmp_data_p2)
        data_p3_set.extend(tmp_data_p3)
        data_problem_set.extend(tmp_data_problem)


    # keep format:
    # file order: x1, x2, u, p1, p2, p3, r
    data_yalmip = torch.stack([torch.tensor(data_x1_set),torch.tensor(data_x2_set),torch.tensor(data_u_set),
                            torch.tensor(data_p1_set),torch.tensor(data_p2_set),
                            torch.tensor(data_p3_set),torch.tensor(data_problem_set)],dim=1)

    return data_yalmip


data_ma = read_csvdata(df_ma)
data_yalmip = read_csvdata(df_yalmip)

# ===============IMPORT NOM DATA===============
path = 'dataset/DifSYS_NOM_Dataset_0.1.txt'

# import dataset and purify outliers    
with open(path, 'r') as f:
    # file order: x1, x2, u, p1, p2, p3, r
    data = [ast.literal_eval(x) for x in f.readlines()]
    data_nom = torch.tensor(data).squeeze(1)


# first comparison: violation rate --- only available on YALMIP data
print(f'\n The YALIMP Infeasible rate is: {100*torch.sum(data_yalmip[:,-1] == 1)/data_yalmip.shape[0]:.2f}%')
print(f' The YALIMP Voilation rate is: {100*torch.sum(data_yalmip[:,-1] == 3)/data_yalmip.shape[0]:.2f}%\n')




# second comparison: performance:
# for those valid datapoints, compare the value of obj. fun. values.

device = torch.device("cpu")
# % read A,B,C,D matrices:
dt = 0.1
Ad = torch.tensor([[1, dt], [0, 1]]).to(device)
Bd=np.array([[0],[0]])

Bd = np.float32(Bd)
Bd = torch.tensor(Bd).to(device)

Q = torch.tensor([[2,0],[0.,2]]).to(device)
R = torch.tensor([0.1]).to(device)



x_r = torch.tensor([0.,0.]).reshape(2,1).to(device)
theta_fix=torch.tensor(0.1).to(device)


#============== DEFINE: OBJECTIVE FUNCTION AND CONSTRAINS:=====================
def obj_fun(data,theta=None,dataset='NOM'):
    # expend for objective values.
    P = torch.stack([data[:, 3:5], data[:, 4:6]], dim=1)

    x = data[:,:2].reshape(data.shape[0],2,1)
    Add = torch.tile(Ad, (data.shape[0], 1, 1))
    Add[:,1,0] = 3*dt*x[:,0,0]**2

    Bdd = torch.tile(Bd, (data.shape[0], 1, 1))
    Bdd[:,1,0] = dt*(x[:,1,0]**2+1)

    
    u = data[:,2].reshape(data.shape[0],1,1)
    if dataset != 'NOM':
        # theta = torch.expand_copy(theta,(u.shape[0],1,1))
        theta = torch.reshape(theta,(u.shape[0],1,1))
    # x_rr = data[:,6].reshape(data.shape[0],1,1)
    x_rr = torch.tile(x_r,(data.shape[0], 1, 1))
    QQ = torch.tile(Q, (data.shape[0], 1, 1))
    x_dag = torch.permute(Add@x+Bdd@u-x_rr, (0, 2, 1))
    # OBJECTIVE FUNCTION
    y = R*u**2 + x_dag @ QQ @ (Add@x+Bdd@u-x_rr)+ (x-x_rr).permute(0,2,1)@P@(x-x_rr) + torch.exp(-theta)
    y = y.squeeze(-1)
    
    return y        




# mask = data_yalmip[:,-1] == 0
# data_yalmip_valid = data_yalmip[mask]
# data_nom_cpr = data_nom[mask]

obj_values_nom = obj_fun(data_nom.type(torch.float32),theta=theta_fix)


# EXCHANGE BETWEEN YALMP & MULTI-AGENT
obj_values_yalmip = obj_fun(data_yalmip.type(torch.float32), theta=theta_fix)
obj_values_ma = obj_fun(data_ma.type(torch.float32), theta=data_ma[:,-1].type(torch.float32),dataset='YALMIP')
# print(f'\n There are {100*torch.sum(obj_values_nom<obj_values_yalmip)/obj_values_yalmip.shape[0]:.2f}% higher than YALMIP.\n')
print(f'The Avg. Cost of YALMIP: {torch.mean(obj_values_yalmip)/torch.mean(obj_values_yalmip)} \n The Avg. Cost of NOM: {torch.mean(obj_values_nom)/torch.mean(obj_values_yalmip)} \n The Avg. Cost of Multi-Agent: {torch.mean(obj_values_ma)/torch.mean(obj_values_yalmip)}.')

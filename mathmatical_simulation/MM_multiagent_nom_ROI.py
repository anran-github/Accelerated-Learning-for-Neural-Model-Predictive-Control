'''
This file does:
 !! Generate TABLE 1 
Given 101x101 fix points:
Check the voilateion rate, control effort:
NOM & YALMIP
'''



import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

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
df1 = pd.read_csv('dataset/MM_DiffSys_dataset.csv', header=None)
# df1 = pd.read_csv('/home/anranli/code/Neural-Optimization-Machine-NOM-main/dataset/DiffSys_01theta_10201points_compare_YALMIP.csv', header=None)

# Assuming the first row is the header, adjust if needed
data = df1.values  


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
data_multi = torch.stack([torch.tensor(data_x1_set),torch.tensor(data_x2_set),torch.tensor(data_u_set),
                           torch.tensor(data_p1_set),torch.tensor(data_p2_set),
                           torch.tensor(data_p3_set),torch.tensor(data_problem_set)],dim=1)

# ===============IMPORT NOM DATA===============
path = 'dataset/DifSYS_NOM_Dataset_0.1.txt'

# import dataset and purify outliers    
with open(path, 'r') as f:
    # file order: x1, x2, u, p1, p2, p3, r
    data = [ast.literal_eval(x) for x in f.readlines()]
    data_nom = torch.tensor(data).squeeze(1)



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
def dynamic_model(data,theta=None,dataset='NOM'):
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
    x_dag = torch.permute(Add@x+Bdd@u, (0, 2, 1))
    # OBJECTIVE FUNCTION
    # y = R*u**2 + x_dag @ QQ @ (Add@x+Bdd@u-x_rr)+ (x-x_rr).permute(0,2,1)@P@(x-x_rr) + torch.exp(-theta)
    # y = y.squeeze(-1)
    
    return x_dag        




# mask = data_multi[:,-1] == 0
# data_multi_valid = data_multi[mask]
# data_nom_cpr = data_nom[mask]

obj_values_nom = dynamic_model(data_nom.type(torch.float32),theta=theta_fix)
roi_mask_nom = torch.zeros_like(obj_values_nom)
# build roi mask:
roi_mask_nom[:,0,0] = torch.logical_and(obj_values_nom[:,0,0]>=-5, obj_values_nom[:,0,0]<=5)
roi_mask_nom[:,0,1] = torch.logical_and(obj_values_nom[:,0,1]>=-5, obj_values_nom[:,0,1]<=5)
roi_nom = torch.logical_and(roi_mask_nom[:,0,0],roi_mask_nom[:,0,1])


# Convert to numpy array for plotting
mask_nom = roi_nom.reshape(101,101)

# EXCHANGE BETWEEN YALMP & MULTI-AGENT
# obj_values_yalmip = dynamic_model(data_multi.type(torch.float32), theta=theta_fix)
obj_values_multi = dynamic_model(data_multi.type(torch.float32), theta=data_multi[:,-1].type(torch.float32),dataset='YALMIP')
roi_mask_multi = torch.zeros_like(obj_values_multi)
# build roi mask:
roi_mask_multi[:,0,0] = torch.logical_and(obj_values_multi[:,0,0]>=-5, obj_values_multi[:,0,0]<=5)
roi_mask_multi[:,0,1] = torch.logical_and(obj_values_multi[:,0,1]>=-5, obj_values_multi[:,0,1]<=5)
roi_multi = torch.logical_and(roi_mask_multi[:,0,0],roi_mask_multi[:,0,1])

# Convert to numpy array for plotting
mask_multi = roi_multi.reshape(101,101)

overlap_mask_np = torch.logical_xor(roi_multi,roi_nom).reshape(101,101)


# ================ Smooth Boundary =======================
# Increase mask resolution by scaling up by 2x
scale_factor = 10
mask_nom_high_res = F.interpolate(mask_nom.float().unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode='bilinear').squeeze().numpy() > 0.5
mask_multi_high_res = F.interpolate(mask_multi.float().unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode='bilinear').squeeze().numpy() > 0.5
overlap_mask_high_res = np.logical_xor(mask_nom_high_res, mask_multi_high_res)

# Apply Gaussian filter to smooth the edges
mask_nom_smooth = gaussian_filter(mask_nom_high_res.astype(float), sigma=10)
mask_multi_smooth = gaussian_filter(mask_multi_high_res.astype(float), sigma=10)
overlap_mask_smooth = gaussian_filter(overlap_mask_high_res.astype(float), sigma=10)


# ============================ ROI PLOT ==================================
# introduce latex
plt.rcParams['text.usetex'] = True

# control number font size on axis.
plt.rcParams.update({'font.size': 25})
plt.rcParams["font.family"] = "Times New Roman"

plt.figure(figsize=(10, 8))  # Adjust width and height as needed
# Set up the color map: orange for True, blue for False
cmap1 = colors.ListedColormap(['white', 'orange'])  # First mask: blue (False), orange (True)
cmap2 = colors.ListedColormap(['none', 'orange'])   # Second mask: transparent (False), green (True)
cmap_overlap = colors.ListedColormap(['none', 'brown'])  # Overlap mask: transparent (False), red (True)

# Plot the mask
# Plot the first mask
plt.imshow(mask_nom_smooth, cmap=cmap1, extent=[-5, 5, -5, 5], origin='lower')

# Plot the second mask
plt.imshow(mask_multi_smooth, cmap=cmap2, extent=[-5, 5, -5, 5], origin='lower', alpha=0.7)

# Plot the overlap mask with red color
plt.imshow(overlap_mask_smooth, cmap=cmap_overlap, extent=[-5, 5, -5, 5], origin='lower', alpha=0.7)
# plt.colorbar(ticks=[0, 1], label='Mask Value')

# Create custom legend
# Create custom legend
legend_patches = [
    Patch(color='orange', label=r'RoA with $\theta=0.1$'),
    # Patch(color='blue', label='Operation Region'),
    # Patch(color='green', label='new 2'),
    Patch(color='brown', label=r'Improvements from introducing $\theta$ as a decision variable')  # Label for the overlap
]
# plt.legend(handles=legend_patches, loc='lower right')
# Add the legend outside the graph, with centered text
legend = plt.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1, frameon=False)

# Center-align the legend text
for text in legend.get_texts():
    text.set_ha('center')  # Center alignment
    
plt.yticks(np.arange(-5, 6, step=1))
plt.xticks(np.arange(-5, 6, step=1))
# plt.title('Region of Attraction for different methods')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.tight_layout()
plt.savefig('ROA_MultiAgent.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
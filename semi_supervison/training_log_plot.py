# plot saved training log

import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.ndimage import gaussian_filter1d

# Paper Figure: Neural MPC output VS MPC

log_file_path = 'semi_supervison/DroneZ_MPC_weights/training_logs_dense_center_S40_linear_Iter20_Epoch10.npz'
with open(log_file_path, 'rb') as f:
    log_data = np.load(f, allow_pickle=True)
    log_data = log_data.tolist()

loss1_linear = log_data['Loss1']
loss2_linear = log_data['Loss2']
Difference_linear = log_data['Difference']

log_file_path = 'semi_supervison/DroneZ_MPC_weights/training_logs_dense_center_S40_erf_Iter20_Epoch10.npz'
with open(log_file_path, 'rb') as f:
    log_data = np.load(f, allow_pickle=True)
    log_data = log_data.tolist()

loss1_erf = log_data['Loss1']
loss2_erf = log_data['Loss2']
Difference_erf = log_data['Difference']

log_file_path = 'semi_supervison/DroneZ_MPC_weights/training_logs_dense_center_S40_vshape_Iter20_Epoch10.npz'
with open(log_file_path, 'rb') as f:
    log_data = np.load(f, allow_pickle=True)
    log_data = log_data.tolist()

loss1_vshape = log_data['Loss1']
loss2_vshape = log_data['Loss2']
Difference_vshape = log_data['Difference']


log_file_path = 'semi_supervison/DroneZ_MPC_weights/training_logs_dense_center_S40_constant_Iter20_Epoch10.npz'
with open(log_file_path, 'rb') as f:
    log_data = np.load(f, allow_pickle=True)
    log_data = log_data.tolist()

loss1_constant = log_data['Loss1']
loss2_constant = log_data['Loss2']
Difference_constant = log_data['Difference']

lr = log_data['Learning_Rate']
epochs = range(1, len(Difference_constant) + 1)

# only plotting difference
# Smoothing the difference data
smooth_difference_linear = gaussian_filter1d(Difference_linear, sigma=2)
smooth_difference_erf = gaussian_filter1d(Difference_erf, sigma=2)
smooth_difference_vshape = gaussian_filter1d(Difference_vshape, sigma=2)
# smooth_difference_constant = gaussian_filter1d(Difference_constant, sigma=2)

plt.figure(figsize=(9,5))

# Plotting raw data with transparent color
# plt.plot(epochs, Difference_linear, label='Linear (raw)', linewidth=1, alpha=0.3, color='blue')
# plt.plot(epochs, Difference_erf, label='Erf (raw)', linewidth=1, alpha=0.3, color='orange')
# plt.plot(epochs, Difference_vshape, label='Up-Down (raw)', linewidth=1, alpha=0.3, color='green')
# plt.plot(epochs, Difference_constant, label='Constant (raw)', linewidth=1, alpha=0.3, color='red')

# Plotting smoothed data
plt.plot(epochs, smooth_difference_linear, label='Linear', linewidth=2, color='blue')
plt.plot(epochs, smooth_difference_erf, label='Erf', linewidth=2, color='orange')
plt.plot(epochs, smooth_difference_vshape, label='Up-Down', linewidth=2, color='green')
# plt.plot(epochs, smooth_difference_constant, label='Constant', linewidth=2, color='red')

plt.xlim(1, len(epochs))
# display x-axis as integers only
plt.xticks(np.arange(1, len(epochs)+1, step=2))
# make font size bigger
plt.xlabel('Iterations', fontsize=17)
plt.ylabel('Difference', fontsize=17)
plt.title('Neural MPC Output Difference over Training Epochs', fontsize=15)
plt.grid(linestyle='--')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('semi_supervison/DroneZ_MPC_weights/training_difference_comparison.png', dpi=300)
plt.show()
plt.close()





# =======================================================================================
# Paper Figure: Loss Comparison

log_file_path = 'semi_supervison/DroneZ_MPC_weights/training_logs_dense_center_S40_linear_Iter10_Epoch5.npz'
with open(log_file_path, 'rb') as f:
    log_data = np.load(f, allow_pickle=True)
    log_data = log_data.tolist()

loss1_epoch5 = log_data['Loss1']
loss2_epoch5 = log_data['Loss2']
Difference_epoch5 = log_data['Difference']


log_file_path = 'semi_supervison/DroneZ_MPC_weights/training_logs_dense_center_S40_linear_Iter10_Epoch10.npz'
with open(log_file_path, 'rb') as f:
    log_data = np.load(f, allow_pickle=True)
    log_data = log_data.tolist()

loss1_epoch10 = log_data['Loss1']
loss2_epoch10 = log_data['Loss2']
Difference_epoch10 = log_data['Difference']

log_file_path = 'semi_supervison/DroneZ_MPC_weights/training_logs_dense_center_S40_linear_Iter10_Epoch15.npz'
with open(log_file_path, 'rb') as f:
    log_data = np.load(f, allow_pickle=True)
    log_data = log_data.tolist()

loss1_epoch15 = log_data['Loss1']
loss2_epoch15 = log_data['Loss2']
Difference_epoch15 = log_data['Difference']



lr = log_data['Learning_Rate']
# epochs = range(1, len(loss1_epoch5) + 1)

# compute mean for each iteration:
iteration = 10
smooth_loss1_ep5 = [np.mean(loss1_epoch5[5*i:int(i*5+5)]) for i in range(10)]
smooth_loss1_ep10 = [np.mean(loss1_epoch10[10*i:int(i*10+10)]) for i in range(10)]
smooth_loss1_ep15 = [np.mean(loss1_epoch15[15*i:int(i*15+15)]) for i in range(10)]
smooth_loss2_ep5 = [np.mean(loss2_epoch5[5*i:int(i*5+5)]) for i in range(10)]
smooth_loss2_ep10 = [np.mean(loss2_epoch10[10*i:int(i*10+10)]) for i in range(10)]
smooth_loss2_ep15 = [np.mean(loss2_epoch15[15*i:int(i*15+15)]) for i in range(10)]


# Smoothing the difference data
smooth_loss1_ep5 = gaussian_filter1d(smooth_loss1_ep5, sigma=1.5)
smooth_loss1_ep10 = gaussian_filter1d(smooth_loss1_ep10, sigma=1.5)
smooth_loss1_ep15 = gaussian_filter1d(smooth_loss1_ep15, sigma=1.5)

smooth_loss2_ep5 = gaussian_filter1d(smooth_loss2_ep5, sigma=1.5)
smooth_loss2_ep10 = gaussian_filter1d(smooth_loss2_ep10, sigma=1.5)
smooth_loss2_ep15 = gaussian_filter1d(smooth_loss2_ep15, sigma=1.5)




plt.figure(figsize=(9,5))
plt.subplot(1,2,1)
# Plotting raw data with transparent color
plt.plot(range(1, iteration + 1), smooth_loss1_ep5, label='5 Epochs', linewidth=2, color='blue')
plt.plot(range(1, iteration + 1), smooth_loss1_ep10, label='10 Epochs', linewidth=2, color='orange')
plt.plot(range(1, iteration + 1), smooth_loss1_ep15, label='15 Epochs', linewidth=2, color='green')
plt.xlim(1, iteration)
# display x-axis as integers only
# plt.xticks(np.arange(1, len(epochs)+1, step=2))
# make font size bigger
plt.xlabel('Iterations', fontsize=17)
plt.ylabel('MSE Loss', fontsize=17)
# plt.title('Neural MPC Output Difference over Training Epochs', fontsize=15)
plt.grid(linestyle='--')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)

plt.subplot(1,2,2)
# Plotting smoothed data
plt.plot(range(1, iteration + 1), smooth_loss2_ep5, label='5 Epochs', linewidth=2, color='blue')
plt.plot(range(1, iteration + 1), smooth_loss2_ep10, label='10 Epochs', linewidth=2, color='orange')
plt.plot(range(1, iteration + 1), smooth_loss2_ep15, label='15 Epochs', linewidth=2, color='green')


plt.xlim(1,iteration)
# display x-axis as integers only
# plt.xticks(np.arange(1, len(epochs)+1, step=2))
# make font size bigger
plt.xlabel('Iterations', fontsize=17)
plt.ylabel('MPC Loss', fontsize=17)
plt.grid(linestyle='--')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('semi_supervison/DroneZ_MPC_weights/training_LossandEpoch.png', dpi=300)
plt.show()


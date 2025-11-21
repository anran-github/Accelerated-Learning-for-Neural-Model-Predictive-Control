import scipy.io
import matplotlib.pyplot as plt
import numpy as np


# control number font size on axis.
# introduce latex
# plt.rcParams['text.usetex'] = True
# plt.rcParams.update({'font.size': 18})
# # make font thicker
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams["font.family"] = "Times New Roman"



# Load the .mat file
mat_MPC = scipy.io.loadmat('DroneZ_MPC/Real_Drone_Experiments/results/MPC_STEP.MAT')
mat_NN = scipy.io.loadmat('DroneZ_MPC/Real_Drone_Experiments/results/MPC_NN_STEP.MAT')

# Access the data from the loaded .mat file
# Assuming the data is stored in a variable named 'data'
SS_MPC = mat_MPC['SS']
SS_NN = mat_NN['SS']

t_MPC = mat_MPC['Time'][1:,0] - mat_MPC['Time'][0,0]
t_NN = mat_NN['Time'][1:,0] - mat_NN['Time'][0,0]


# Plot the data

# Step Reference Range
start = 120
start_NN = 600
end = 546
end_NN = 3090

reference = []
for t in t_MPC:
    if t < 10:
        reference.append(1.0)
    elif t>= 10 and t < 20:
        reference.append(2.0)
    elif t >=20 and t < 30:
        reference.append(1.2)
    else:
        reference.append(1.7)

# summary PI and tracking error
tracking_error_mpc, tracking_error_nn = 0, 0
PI_mpc, PI_nn = 0, 0
CI_mpc, CI_nn = [], []
for i in range(start,end):
    t = t_MPC[i]
    z_ref = reference[i]
    # find the closest time in NN data
    idx_nn = (np.abs(t_NN - t)).argmin()
    z_mpc = SS_MPC[2,i]
    z_nn = SS_NN[2,idx_nn]

    tracking_error_mpc += np.linalg.norm(z_ref - z_mpc)
    tracking_error_nn += np.linalg.norm(z_ref - z_nn)

    PI_mpc += np.linalg.norm(mat_MPC['CI'][2,i])
    PI_nn += np.linalg.norm(mat_NN['CI'][2,idx_nn])
    CI_mpc.append(mat_MPC['CI'][2,i])
    CI_nn.append(mat_NN['CI'][2,idx_nn])



plt.figure(figsize=(9,6))
plt.subplot(211)
plt.plot(t_MPC[start:end],SS_MPC[2,start:end],label='MPC',linewidth=3)
plt.plot(t_NN[start_NN:end_NN],SS_NN[2,start_NN:end_NN],label='Neural MPC',linewidth=3)
# plot reference
plt.plot(t_MPC[start:end],reference[start:end],'--',label='Reference',linewidth=3,color='0.5')
plt.legend(fontsize=18,loc='lower right')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Time [s]',fontsize=20)
plt.ylabel('Z Position [m]',fontsize=20)
plt.grid(linestyle = '--')

# # control input
plt.subplot(212)
# plt.plot(t_MPC[start:end],mat_MPC['CI'][2,start:end],label='MPC')
# plt.plot(t_NN[start_NN:end_NN],mat_NN['CI'][2,start_NN:end_NN],label='Neural MPC')
plt.plot(t_MPC[start:end],CI_mpc,label='MPC')
plt.plot(t_MPC[start:end],CI_nn,label='Neural MPC')
plt.legend(fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Time [s]',fontsize=20)
plt.ylabel('Control Input',fontsize=20)
plt.grid(linestyle = '--')
plt.tight_layout()
# plt.savefig('DroneZ_MPC/Real_Drone_Experiments/dataplot_NN_LQR_Step.png',dpi=500)
plt.show()


print(f'MPC Tracking Error: {tracking_error_mpc:.4f}, NMPC Tracking Error: {tracking_error_nn:.4f}')
print(f'MPC Control Effort (PI): {PI_mpc:.4f}, NMPC Control Effort (PI): {PI_nn:.4f}')
print('-----------------------------------')


# =======================Sin wave reference===================================

mat_MPC = scipy.io.loadmat('DroneZ_MPC/Real_Drone_Experiments/results/MPC_Sin.MAT')
mat_NN = scipy.io.loadmat('DroneZ_MPC/Real_Drone_Experiments/results/MPC_NN_Sin.MAT')

# Access the data from the loaded .mat file
# Assuming the data is stored in a variable named 'data'
SS_MPC = mat_MPC['SS']
SS_NN = mat_NN['SS']

t_MPC = mat_MPC['Time'][1:,0] - mat_MPC['Time'][0,0]
t_NN = mat_NN['Time'][1:,0] - mat_NN['Time'][0,0]



# Sin Reference
# start = 85
# start_NN = 450
# end = 820
# end_NN = 5000
start = 370
start_NN = 2000
end = 820
end_NN = 5000

reference = [0.5*np.sin(0.2*x) + 1.5 for x in t_MPC]


# summary PI and tracking error
tracking_error_mpc, tracking_error_nn = 0, 0
PI_mpc, PI_nn = 0, 0
CI_mpc, CI_nn = [], []
for i in range(start,end):
    t = t_MPC[i]
    z_ref = reference[i]
    # find the closest time in NN data
    idx_nn = (np.abs(t_NN - t)).argmin()
    z_mpc = SS_MPC[2,i]
    z_nn = SS_NN[2,idx_nn]

    tracking_error_mpc += np.linalg.norm(z_ref - z_mpc)
    tracking_error_nn += np.linalg.norm(z_ref - z_nn)

    PI_mpc += np.linalg.norm(mat_MPC['CI'][2,i])
    PI_nn += np.linalg.norm(mat_NN['CI'][2,idx_nn])
    CI_mpc.append(mat_MPC['CI'][2,i])
    CI_nn.append(mat_NN['CI'][2,idx_nn]-0.006) # offset for better plot




plt.figure(figsize=(9,6))
plt.subplot(211)
plt.plot(t_MPC[start:end],SS_MPC[2,start:end],label='MPC',linewidth=3)
plt.plot(t_NN[start_NN:end_NN],SS_NN[2,start_NN:end_NN],label='Neural MPC',linewidth=3)
# plot reference
plt.plot(t_MPC[start:end],reference[start:end],'--',label='Reference',linewidth=3, color='0.5')
plt.legend(fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Time [s]',fontsize=20)
plt.ylabel('Z Position [m]',fontsize=20)
plt.grid(linestyle = '--')

# control input
plt.subplot(212)
# plt.plot(t_MPC[start:end],mat_MPC['CI'][2,start:end],label='MPC')
# plt.plot(t_NN[start_NN:end_NN],mat_NN['CI'][2,start_NN:end_NN],label='Neural MPC')
plt.plot(t_MPC[start:end],CI_mpc,label='MPC')
plt.plot(t_MPC[start:end],CI_nn,label='Neural MPC')
plt.legend(fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Time [s]',fontsize=20)
plt.ylabel('Control Input',fontsize=20)
plt.grid(linestyle = '--')
plt.tight_layout()
plt.savefig('DroneZ_MPC/Real_Drone_Experiments/dataplot_NN_LQR_Sin.png',dpi=500)
plt.show()





print(f'MPC Tracking Error: {tracking_error_mpc:.4f}, NMPC Tracking Error: {tracking_error_nn:.4f}')
print(f'MPC Control Effort (PI): {PI_mpc:.4f}, NMPC Control Effort (PI): {PI_nn:.4f}')
print('-----------------------------------')


# # Z direction 
# plt.subplot(211)
# plt.plot(t_MPC[start:end]-t_MPC[start],SS_MPC[2,start:end],label='MPC')
# plt.plot(t_NN[start_NN:end_NN]-t_NN[start_NN],SS_NN[2,start_NN:end_NN],label='NMPC')
# plt.legend()
# plt.xlim(0,14)
# plt.xlabel('Time [s]')
# plt.ylabel('Z Position [m]')
# plt.grid(linestyle = '--')

# # Plot control inputs
# plt.subplot(212)
# plt.plot(t_MPC[start:end]-t_MPC[start],mat_MPC['CI'][2,start:end],label='MPC')
# plt.plot(t_NN[start_NN:end_NN]-t_NN[start_NN],mat_NN['CI'][2,start_NN:end_NN],label='NMPC')
# plt.legend()
# plt.xlim(0,14)
# plt.xlabel('Time [s]')
# plt.ylabel('Control Input')
# plt.grid(linestyle = '--')

# plt.tight_layout()
# # plt.savefig('Drone_Experiments/Drone_NN_LQR.png',dpi=500)
# plt.show()


# # Analysis control effort.
# control_MPC = mat_MPC['CI']
# control_nn = mat_NN['CI']

# for i in range(3):
#     if i == 2:
#         i += 1
#     # print(i)
#     # control_MPC = controls_lqr[i,:]
#     # control_nn = controls_nn[i,:]
#     mpc_ctrl_sum = np.sum(np.abs(control_MPC[i,start:end]))
#     nn_ctrl_sum = np.sum(np.abs(control_nn[i,start:end]))

#     plt.subplot(211)
#     plt.plot(t_MPC[start:end]-t_MPC[start],control_MPC[i,start:end],label='LQR')
#     _,x = plt.xlim()
#     y,_ = plt.ylim()
#     plt.annotate(f'Integrate Value:{mpc_ctrl_sum:.1f}',[x/3,y+0.01],bbox=dict(boxstyle="round", fc="none", ec="gray"))
#     plt.legend()
#     plt.ylabel('LQR Input')
#     plt.grid()
#     plt.title('Control Signal Changes with time')
#     plt.subplot(212)
#     plt.plot(t_MPC[start:end]-t_MPC[start],control_nn[i,start:end],label='NN')
#     _,x = plt.xlim()
#     y,_ = plt.ylim()
#     plt.annotate(f'Integrate Value:{nn_ctrl_sum:.1f}',[x/3,y+0.01],bbox=dict(boxstyle="round", fc="none", ec="gray"))
#     plt.grid()
#     plt.ylabel('NN Input')
#     plt.xlabel('Time [s]')
#     plt.legend()
#     plt.show()
#     plt.close()

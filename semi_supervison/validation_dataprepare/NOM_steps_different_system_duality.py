
# ==========================================================
    # Use Objtive Fun. and NOM for IP problems
# ==========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse


import os
from torch.optim import lr_scheduler
from tqdm import tqdm


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Objective_Formulations_new import ObjectiveFormulation



# MODEL FOR IP PROBLEM
class NOM_IPP(nn.Module):
    def __init__(self, obj_fun=0, constrains=[]):
        super(NOM_IPP, self).__init__()

        self.input_u = nn.Linear(1, 1)
        self.input_p1 = nn.Linear(1, 1)
        self.input_p2 = nn.Linear(1, 1)
        self.input_p3 = nn.Linear(1, 1)
        self.input_theta = nn.Linear(1, 1)
        self.obj_fun = obj_fun
        self.constrains = constrains
    

    def forward(self, x):
        # input order: x1,x2,u,p1,p2,p3,theta,xr
        x1,x2,u = x[:,0].unsqueeze(-1),x[:,1].unsqueeze(-1),x[:,2].unsqueeze(-1)
        p1,p2,p3 = x[:,3].unsqueeze(-1),x[:,4].unsqueeze(-1),x[:,5].unsqueeze(-1)
        theta = x[:,6].unsqueeze(-1)
        xr = 0.*x[:,-1].unsqueeze(-1)
        in_x1 = x1
        in_x2 = x2
        # trainable weights:
        in_u = self.input_u(u)


        in_p1 = self.input_p1(p1)
        in_p2 = self.input_p2(p2)
        in_p3 = self.input_p3(p3)
        in_theta = self.input_theta(theta)
        # xr = torch.ones_like(in_u)

        layer_1 = torch.cat((in_x1,in_x2,in_u,in_p1,in_p2,in_p3,in_theta,xr),dim=1)


        output_obj = self.obj_fun(layer_1)
        value_cons = [constraint(layer_1) for constraint in self.constrains]

        output = output_obj + sum(value_cons)


        return output




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


#---------------- IP Variables Setting-----------------

# % read A,B,C,D matrices:
dt = 0.2
Ad = torch.tensor([[1, dt], [0, 1]]).to(device)
Bd=np.array([[0],[0]])
Bd = np.float32(Bd)
Bd = torch.tensor(Bd).to(device)



# import reference and theta
parser = argparse.ArgumentParser(description="Process some input parameters.")
parser.add_argument('--theta', type=float, default=0.1, help="Theta: [0.1, 0.01, 0.001]")
parser.add_argument('--reference', type=float, default=0., help="Reference: range between [-1,1]")
parser.add_argument('--resume', type=bool, default=False, help="Reference: range between [-1,1]")

args = parser.parse_args()

x_r = torch.tensor([args.reference,0.]).reshape(2,1).to(device)
# theta=torch.tensor(args.theta).to(device)


#============== DEFINE: OBJECTIVE FUNCTION AND CONSTRAINS:=====================
# MODEL FOR IP PROBLEM
class NOM(nn.Module):
    def __init__(self,):
        super(NOM, self).__init__()

        self.input_u = nn.Linear(1, 1)
        self.input_p1 = nn.Linear(1, 1)
        self.input_p2 = nn.Linear(1, 1)
        self.input_p3 = nn.Linear(1, 1)
        self.input_theta = nn.Linear(1, 1)

        formulation = ObjectiveFormulation(device=device)
        self.obj_fun = formulation.obj_fun
        self.constrains = [formulation.constrain1, formulation.constrain2,formulation.constrain3]
    

    def forward(self, x):
        # input shape: (bs,6): arg order: x1, x2, u, p1, p2, p3, theta
        x1,x2,u = x[:,0].unsqueeze(-1),x[:,1].unsqueeze(-1),x[:,2].unsqueeze(-1)
        p1,p2,p3 = x[:,3].unsqueeze(-1),x[:,4].unsqueeze(-1),x[:,5].unsqueeze(-1)
        theta = x[:,6].unsqueeze(-1)
        xr = 0.*x[:,6].unsqueeze(-1)
        in_x1 = x1
        in_x2 = x2
        # trainable weights:
        in_u = self.input_u(u)
        in_p1 = self.input_p1(p1)
        in_p2 = self.input_p2(p2)
        in_p3 = self.input_p3(p3)
        in_theta = self.input_theta(theta)
        # xr = torch.ones_like(in_u)

        layer_1 = torch.cat((in_x1,in_x2,in_u,in_p1,in_p2,in_p3,in_theta,xr),dim=1)


        output_obj = self.obj_fun(layer_1)
        # value_cons = [constraint(layer_1) for constraint in self.constrains]
        # value_cons = torch.stack(value_cons,dim=1).squeeze(-1)
        # output = output_obj + torch.sum(value_cons,dim=1).unsqueeze(-1)
        output = output_obj 
        # sum up obj function and constraints.
        for f_constrain in self.constrains:

            constraint_value = f_constrain(layer_1)
            if constraint_value.dim() == 3:
                constraint_value = constraint_value.squeeze(-1)

            output += constraint_value

        return output


    # ======================MODEL LOADING===========================



def model_init(offset=True):
    # device = torch.device("cpu")
    model = NOM().to(device)
    
    # freeze WHOLE weights
    for param in model.parameters():
        param.requires_grad = False

    model.input_u.weight.requires_grad = True
    model.input_p1.weight.requires_grad = True
    model.input_p2.weight.requires_grad = True
    model.input_p3.weight.requires_grad = True
    model.input_theta.weight.requires_grad = True
    model.input_u.bias.requires_grad = True
    model.input_p1.bias.requires_grad = True
    model.input_p2.bias.requires_grad = True
    model.input_p3.bias.requires_grad = True
    model.input_theta.bias.requires_grad = True

    # or:
    # model.input_u.requires_grad_(True)
    # model.input_p1.requires_grad_(True)
    # model.input_p2.requires_grad_(True)
    # model.input_p3.requires_grad_(True)


    if offset:
        offset_w  = (torch.rand(1).item()-0.5)*0.001
        offset_alpha = (torch.rand(1).item()-0.5)*0.001
    else:
        offset_w  = 0
        offset_alpha = 0

    nn.init.constant_(model.input_u.weight, 1+offset_w)
    nn.init.constant_(model.input_p1.weight, 1)
    nn.init.constant_(model.input_p2.weight, 1)
    nn.init.constant_(model.input_p3.weight, 1)
    nn.init.constant_(model.input_theta.weight, 1)

    nn.init.constant_(model.input_u.bias, 0+offset_alpha)
    nn.init.constant_(model.input_p1.bias, 0)
    nn.init.constant_(model.input_p2.bias, 0)
    nn.init.constant_(model.input_p3.bias, 0)
    nn.init.constant_(model.input_theta.bias, 0)

    return model, offset_alpha, offset_w



def main(epochs=2000, lr=0.0035,start_point=[]):
    # device = torch.device("cpu")
    net,offset_alpha, offset_w = model_init(offset=True)
    criterion = nn.MSELoss()
    # results = {}
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs//50, eta_min=lr*0.07)
            
    net.train()
        


    loss_list = []
    bar = tqdm(range(epochs),leave=False)
    num_cpr = 50
    inputs = start_point.unsqueeze(0)
    labels = torch.zeros(1,1).to(device)

    for epoch in range(epochs):

        optimizer.zero_grad()
        if torch.cuda.is_available():
            outputs = net(inputs.to(device))
        else:
            outputs = net(inputs)

        # loss = criterion(outputs, labels.reshape(-1,1).to(device))
        loss = outputs
        
        if epoch == 0:
            loss_opt= loss.item()
            weight_opt = [net.input_u.weight.data.item(),
                          net.input_p1.weight.data.item(),
                            net.input_p2.weight.data.item(),
                            net.input_p3.weight.data.item(),
                            net.input_theta.weight.data.item()]
            bias_opt = [net.input_u.bias.data.item(),
                        net.input_p1.bias.data.item(),
                            net.input_p2.bias.data.item(),
                            net.input_p3.bias.data.item(),
                            net.input_theta.bias.data.item()]
            best_vals = [loss_opt, weight_opt,bias_opt]


        # save weights and bias for the lowest loss 
        if loss.item() < best_vals[0]:
            best_vals[0] = loss.item()
            best_vals[1] = [net.input_u.weight.data.item(),
                            net.input_p1.weight.data.item(),
                            net.input_p2.weight.data.item(),net.input_p3.weight.data.item(),
                            net.input_theta.weight.data.item()]
            best_vals[2] = [net.input_u.bias.data.item(),
                            net.input_p1.bias.data.item(),
                            net.input_p2.bias.data.item(),net.input_p3.bias.data.item(),
                            net.input_theta.bias.data.item()]


        loss.backward()
        # gradient clip to avoid gradient explosion.
        torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1)

        # Update the model parameters
        optimizer.step()  
        # Step the scheduler
        scheduler.step()



        # check loss still decreases or not.
        # loss_list.append(loss.item())
        # if len(loss_list) == num_cpr:
        #     avg = round(sum(loss_list)/len(loss_list),3)
        #     if avg == round(loss.item(),3):
        #         # it means the loss keeps the same.
        #         break
        #     loss_list.pop(0)

        if torch.cuda.is_available():
            bar.set_postfix(epoch=epoch,val=outputs.cpu().item(),loss=loss.cpu().item(),lrs=np.round(scheduler.get_last_lr(),5).tolist())
        else:
            bar.set_postfix(epoch=epoch,val=outputs.item(),loss=loss.item(),lrs=np.round(scheduler.get_last_lr(),5).tolist())


    # use weights and bias to get optimal values:
    # net.eval()
    
    with torch.no_grad():
        optimal_u = best_vals[1][0]*(start_point[2]*(1+offset_w) + offset_alpha) + best_vals[2][0]
        optimal_p1 = best_vals[1][1]*start_point[3] + best_vals[2][1]
        optimal_p2 = best_vals[1][2]*start_point[4] + best_vals[2][2]
        optimal_p3 = best_vals[1][3]*start_point[5] + best_vals[2][3]
        optimal_theta = best_vals[1][4]*start_point[6] + best_vals[2][4]
    
    optimal_point = torch.tensor([start_point[0],start_point[1],optimal_u,optimal_p1,optimal_p2,
                                    optimal_p3,optimal_theta,start_point[7]],dtype=torch.float32).unsqueeze(0).to(device)

    # results['objvalue'] = optimal_y
    # results['optpoint'] = optimal_point
    # results['eig'] = eig

    return optimal_point
        


def worker(lr, param, return_dict, index):
    result = main(lr=lr,start_point=param)
    # IMPORTANT: detach from gpu.
    return_dict[index] = result.cpu()






if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)

    # ======================== OPTIMAL VERIFY==========================
    x0 = torch.tensor([2, -1]).reshape(1,2).to(device)
    iterations = 30
    x1_set = []
    x2_set = []
    v_dot_set = []
    delta_x_set = []
    u_set = []
    obj_values = {'optimal':{'cons':[],'obj':[]},'init':{'cons':[],'obj':[]}}

    topk = 10

    # instead of using dataset, set u and P random points and find the topk lowest

    min_x, max_x = -1,1
    min_u, max_u = -15,15
    min_p1, max_p1 = 0.001,0.01
    min_p2, max_p2 = 0.001,0.01
    min_p3, max_p3 = 0.001,0.01
    num = 1
    u = np.linspace(min_u,max_u, num*16)
    p1 = np.linspace(min_p1,max_p1, num*6)
    p2 = np.linspace(min_p2,max_p2, num*5)
    p3 = np.linspace(min_p3,max_p3, num*6)
    theta = np.linspace(1e-4,1, num*6)
    # U, P1, P2, P3 = np.meshgrid(u, p1, p2, p3)
    # data = np.column_stack((U.ravel(), P1.ravel(), P2.ravel(), P3.ravel()))
    # data = torch.tensor(data,dtype=torch.float32).to(device)
    X1, X2, U, P1, P2, P3,THETA, XR = np.meshgrid(0, 0, u, p1, p2, p3,theta, 0)
    data = np.column_stack((X1.ravel(), X2.ravel(), U.ravel(), P1.ravel(), P2.ravel(), P3.ravel(),THETA.ravel(), XR.ravel()))
    data = torch.tensor(data,dtype=torch.float32).to(device)

    t_start = time()
    # start loop
    optimal_save = []
    # this model is for choosing the optimal point
    NOM_valid,_,_ = model_init(offset=False)
    NOM_valid.eval()
    i_bar = tqdm(range(iterations))
    for t in i_bar:
        if t==0:
            xt = x0.numpy()
        # if t != 0 and torch.sum(torch.round(xt-x0,decimals=2)) == 0:
        #     print('target reached!')
        #     break
        data[:,0],data[:,1] = xt[0,0].item(), xt[0,1].item()


        # use OBJ NN to generate an order set
        with torch.no_grad():
            # y_pred = obj_fun(available_data) + constrain1(available_data) + constrain3(available_data) +constrain4(available_data)
            y_pred = NOM_valid(data)
            values_st,idx = torch.topk(y_pred.flatten(), topk,largest=False)
        # consider top 5 from lowest to highest

        # ===================== train NOM for each starting points =========================
        starting_pts = data[idx]

        # MULTI PROCESSING ...... 
        with mp.Manager() as manager:
            return_dict = manager.dict()

            # List to keep track of process objects
            processes = []

            # Create and start subprocesses
            for ii, param in enumerate(starting_pts):
                # CHANGING learning rate:
                lr = values_st[ii]*(0.00025 + 0.00005 * torch.rand(1).item())  # smaller step when close to zero.
                if lr.item() >= 1:
                    lr = torch.tensor(0.001)
                lr = torch.tensor(0.01)
                p = mp.Process(target=worker, args=(lr.item(), param, return_dict, ii))
                processes.append(p)
                p.start()

            # Wait for all subprocesses to finish
            for p in processes:
                p.join()

            # Summarize results
            results = [return_dict[i] for i in range(len(starting_pts))]
            # print("Results:", results)

        # only consider non-nan values:
        optimal_vals = [x for x in results if torch.isnan(x).sum()==0]
        # in case all optimal values are nan: choose the lowest starting point
        if len(optimal_vals) == 0:
            optimal_vals =  [starting_pts[0]]
        # convert to tensor
        optimal_vals = torch.stack(optimal_vals).squeeze(1).to(device)
        with torch.no_grad():
            y_final = NOM_valid(optimal_vals.to(device))
            # y_obj = obj_fun(optimal_vals) 
            values,id = torch.min(y_final,0)
        optimal_best = optimal_vals[id]
        # optimal order: x1,x2,u,p1,p2,p3,theta,xr
        P = torch.tensor([[optimal_best[0,3],optimal_best[0,4]],
                          [optimal_best[0,4],optimal_best[0,5]]]).reshape(2,2)
        eig,_ = torch.linalg.eigh(P)

        optimal_save.append(optimal_best.tolist())
        input_data = starting_pts[id]
        # cons_init = (constrain1(input_data) + constrain3(input_data)+ constrain4(input_data)).item()
        # cons_final = (constrain1(optimal_best) + constrain3(optimal_best)+ constrain4(optimal_best)).item()
        print('---------------------------------------------------')
        print('Current state [{:.3f},{:.3f},{:.3f}]. optimal u: {:.3e}, eigen values: {}'.format(
            optimal_best[0,0],optimal_best[0,1],optimal_best[0,6], optimal_best[0,2].item(),eig))
        # note that the starting point objective value is not necessary the same as this NN init:
        # baceuse y_pred[idx[0]] is the min value of starting point.
        print('Optimal Val: {:.5f} | NN init: {:.5f}'.format(y_final[id].item(), y_pred[idx[0]].item()))
        # print(f'Constrains: {cons_final:} | {cons_init}')


        # UPDATE STATE
        xt = xt.T
        x1_tt = xt[0,0] + dt*xt[1,0]
        x2_tt= xt[1,0] + dt*(xt[0,0]**3) + dt*(xt[1,0]**2+1) * optimal_best[0,2].item()
        xtt =  np.array([x1_tt.item(),x2_tt.item()]).reshape(xt.shape[0],xt.shape[1])

        x1_set.append(xt[0,0].item())
        x2_set.append(xt[1,0].item())
        # our previous V

        # v_dot = np.sqrt(np.linalg.norm(xtt.T @ P @ xtt ,axis=1)) - np.sqrt(np.linalg.norm(xt.T @ P @ xt,axis=1))
        # v_dot_set.append(v_dot)
        u_set.append(eig.min().item())
        ##======REMEMBER TO CHEKCK THETA HERE (0.01)======
        # arrow_trajectory.append(np.linalg.norm(xt))
        delta_x_set.append((xtt-xt))
        xt = xtt.T
        p_tt = P



    # # save optimal solutions
    # with open('NOM_Trajectory.txt','a') as f:
    #     [f.writelines(str(x)+ '\n') for x in optimal_save]
    # optimal_save = [] # clear 




# '''
    t_consume = time() - t_start
    print(f'total time: {t_consume:.2f} s')
    x1_set = np.array(x1_set).T

    times = np.array(range(len(u_set)))*dt
    plt.subplot(311)

    plt.plot(times,x1_set,marker='o')
    # plt.plot(x1_set[-1],x2_set[-1],'r+')

    plt.title('State/Control input change versus time with NOM')
    plt.ylabel('X1')
    plt.grid()

    plt.subplot(312)

    plt.plot(times,x2_set,marker='o')
    # plt.plot(x1_set[-1],x2_set[-1],'r+')

    plt.ylabel('X2')
    plt.grid()

    plt.subplot(313)
    plt.plot(times,u_set,marker='o')

    plt.xlabel('Time (s)')
    plt.ylabel('Minimum Eigen Value of P')
    plt.grid()

    plt.tight_layout()
    plt.savefig('image_NOM.png')
    plt.show()
    plt.close()

    '''
    plt.subplot(211)
    # show constrains compare
    plt.plot(range(iterations),obj_values['init']['cons'],label='Starting')
    plt.plot(range(iterations),obj_values['optimal']['cons'],'r+-',label='Optimal')
    plt.text(0.5,11.,f'total time: {t_consume:.2f} s')
    plt.legend()
    plt.title('Constraint Values')
    plt.grid()
    plt.subplot(212)
    # show constrains compare
    plt.plot(range(iterations),obj_values['init']['obj'],label='Starting')
    plt.plot(range(iterations),obj_values['optimal']['obj'],'r+-',label='Optimal')
    plt.legend()
    plt.title('Objective Function Values')
    plt.xlabel('Iterations')
    plt.grid()
    plt.savefig('values.png')
    plt.show()
    plt.close()
    '''
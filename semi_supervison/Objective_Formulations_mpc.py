import torch 
import numpy as np
import control




#============== DEFINE: OBJECTIVE FUNCTION AND CONSTRAINS:=====================

class ObjectiveFormulation():
    """
    Objective function and constraints for the optimization problem.
    """
    def __init__(self,device='cpu'):

        # Ad: system matrix
        # Bd: control matrix
        # dt: time step
        # Q: state cost matrix
        # R: control cost matrix
        # x_r: reference trajectory
        # theta: parameter for constraint


        # % read A,B,C,D matrices:
        A = np.array([[0, 1], [0, -1.7873]])
        B = np.array([[0],[-1.7382]])
        C = np.array([[1,0]])
        D = 0
        
        self.dt = 0.2 # sampling time
        ss = control.ss(A, B, C, D)
        Gd = control.c2d(ss,self.dt, method='zoh') # 'zoh':assuming control input be constant between sampling intervals.
        
        self.Ad = torch.tensor(Gd.A,dtype=torch.float32).to(device)
        self.Bd = torch.tensor(Gd.B,dtype=torch.float32).to(device)
        self.Cd = torch.tensor(Gd.C,dtype=torch.float32).to(device)
        self.Dd = torch.tensor(Gd.D,dtype=torch.float32).to(device)


        # MPC parameters
        self.N = 10;              # Prediction horizon
        self.Q = torch.tensor([[2.,0],[0.,1.]]).to(device)
        self.R = torch.tensor([1]).to(device)

        self.u_min = -0.6
        self.u_max = 0.6


        # self.x_r = torch.tensor([[1.5],[0]],dtype=torch.float32).to(device)
        

        self.c = torch.tensor(1e2).to(device)
        self.acti_fun = torch.nn.Sequential(torch.nn.Tanh(),torch.nn.ReLU())


    def forward(self, nn_input, nn_output):
        """
        Forward pass for the objective function and constraints.
        :param data: input data, shape (batch_size, 6)
        :return: objective values and constraints
        """
        # nn_input: [x1,x2,xr]
        # nn_output: [u]
        
        # data order: x1, x2, ui,...,uN
        data_total = torch.cat((nn_input[:,:2], nn_output), dim=1)
        # update reference trajectory
        self.xr_input   = torch.zeros((nn_input.shape[0],2,1), dtype=torch.float32).to(nn_input.device)
        self.xr_input[:,0,0] = nn_input[:,2]  # update x1, change it based on your model

        # ================================ MPC:==================================
        for i in range(self.N):

            # calculate cost for each iteration
            # data format: (bs,3): [x1, x2, ui]
            if i == 0:
                data = data_total[:,:3]
            else:
                # reshape to (bs, 3): [x1, x2, ui]
                data = torch.cat((xt_plus, data_total[:,2+i].reshape(data_total.shape[0],1,1)), dim=2)
                data = data.squeeze(1)  # remove the second dimension
            # Objective values
            obj_values,xt_plus = self.obj_fun(data)
        
            # Constraints
            constraints_i = self.constrain1(data) 
            
            # sum up the objective values and constraints
            if i == 0:
                obj_values_total = obj_values
                constraints_total = constraints_i
            else:
                obj_values_total += obj_values
                constraints_total += constraints_i


        # terminal condition
        K, P, _ = control.dlqr(self.Ad.cpu().numpy(), self.Bd.cpu().numpy(),
                                self.Q.cpu().numpy(), self.R.cpu().numpy())
        K, P    = torch.tensor(K, dtype=torch.float32).to(data_total.device), torch.tensor(P, dtype=torch.float32).to(data_total.device)

        # add terminal cost
        xt_plus = data[:,:2].reshape(data.shape[0],2,1)  # reshape to (bs, 2, 1)
        boj_values_terminal =  (xt_plus-self.xr_input).permute(0,2,1)@P@(xt_plus-self.xr_input)
        obj_values_total    += boj_values_terminal.squeeze(-1)

        # add terminal constraint
        Add = torch.tile(self.Ad, (data.shape[0], 1, 1))
        Bdd = torch.tile(self.Bd, (data.shape[0], 1, 1))
        x_terminal = xt_plus
        for i in range(20):
            u =  - K@(x_terminal-self.xr_input)
            # add constraint
            constraints_total += self.constrain1(torch.cat((x_terminal.reshape(data.shape[0],2), u.reshape(data.shape[0],1)), dim=1))
            x_terminal = Add@x_terminal + Bdd@u 

        # ==================================================================


        return obj_values_total + constraints_total
    
    def forward_any(self, nn_input, nn_output):
        """
        Forward pass for the objective function and constraints.
        :param data: input data, shape (batch_size, 6)
        :return: objective values and constraints
        """
        # input x: [x1,x2]
        # nn_output: [u]
        
        # data order: x1, x2, ui,...,uN
        data_total = torch.cat((nn_input[:,:2], nn_output), dim=1)
        
        # ================================ MPC:==================================
        for i in range(self.N):

            # calculate cost for each iteration
            # data format: (bs,3): [x1, x2, ui]
            if i == 0:
                data = data_total[:,:3]
            else:
                # reshape to (bs, 3): [x1, x2, ui]
                data = torch.cat((xt_plus, data_total[:,2+i].reshape(data_total.shape[0],1,1)), dim=2)
                data = data.squeeze(1)  # remove the second dimension
            # Objective values
            obj_values,xt_plus = self.obj_fun(data)
        
            # Constraints
            constraints_i = self.constrain1(data) 
            
            # sum up the objective values and constraints
            if i == 0:
                obj_values_total = obj_values
                constraints_total = constraints_i
            else:
                obj_values_total += obj_values
                constraints_total += constraints_i


        # terminal condition
        K, P, _ = control.dlqr(self.Ad.cpu().numpy(), self.Bd.cpu().numpy(),
                                self.Q.cpu().numpy(), self.R.cpu().numpy())
        K, P    = torch.tensor(K, dtype=torch.float32).to(data_total.device), torch.tensor(P, dtype=torch.float32).to(data_total.device)

        # add terminal cost
        xt_plus = data[:,:2].reshape(data.shape[0],2,1)  # reshape to (bs, 2, 1)
        boj_values_terminal =  (xt_plus-self.x_r).permute(0,2,1)@P@(xt_plus-self.x_r)
        obj_values_total    += boj_values_terminal.squeeze(-1)

        # add terminal constraint
        Add = torch.tile(self.Ad, (data.shape[0], 1, 1))
        Bdd = torch.tile(self.Bd, (data.shape[0], 1, 1))
        x_terminal = xt_plus
        for i in range(20):
            u =  - K@x_terminal
            # add constraint
            constraints_total += self.constrain1(torch.cat((x_terminal.reshape(data.shape[0],2), u.reshape(data.shape[0],1)), dim=1))
            x_terminal = Add@x_terminal + Bdd@u 

        # ==================================================================


        return obj_values_total + constraints_total
    
    
    def obj_fun(self,data):
        # data format: (bs,3): [x1, x2, u]

        Add = torch.tile(self.Ad, (data.shape[0], 1, 1))
        Bdd = torch.tile(self.Bd, (data.shape[0], 1, 1))
        
        x = data[:,:2].reshape(data.shape[0],2,1)
        u = data[:,2].reshape(data.shape[0],1,1)

        QQ = torch.tile(self.Q, (data.shape[0], 1, 1))

        # OBJECTIVE FUNCTION
        y = self.R*u**2 +  (x-self.xr_input).permute(0,2,1)@QQ@(x-self.xr_input)
        y = y.squeeze(-1)

        # next state
        x_dag_prime = torch.permute(Add@x+Bdd@u, (0, 2, 1))
        
        return y ,x_dag_prime       


    def constrain1(self,x):
        # bound the control input U
        
        # input shape: (bs,3): arg order: x1, x2, u
        u = x[:,2].unsqueeze(-1)

        lower_bound = self.acti_fun(self.u_min - u)
        upper_bound = self.acti_fun(u - self.u_max)
        
        torch.cuda.empty_cache()

        return self.c*torch.sum(lower_bound+upper_bound,dim=1).unsqueeze(-1)
        # return self.c*torch.sum(torch.sign(torch.relu(1e-15-eig_values)),dim=1).unsqueeze(-1)




# test code with reliable dataset--generated by multi-agent method.

'''
from dataprocessing import DroneZ_Data_Purify

datapath = 'DroneZ_MPC/dataset/drone_mpc_z.csv'

data_tmp = DroneZ_Data_Purify(datapath,u_max=1e5)

# import dataset 
input_data_valid,label_data_valid = data_tmp.return_data()

input_data_valid = torch.tensor(input_data_valid, dtype=torch.float32)
label_data_valid = torch.tensor(label_data_valid, dtype=torch.float32)

# Initialize the objective formulation
objective_formulation = ObjectiveFormulation()
# Forward pass through the objective formulation
output = objective_formulation.forward(input_data_valid, label_data_valid)
print("Output of the objective formulation:", output.item())

'''


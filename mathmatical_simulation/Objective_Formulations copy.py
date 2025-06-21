import torch 
import numpy as np




#============== DEFINE: OBJECTIVE FUNCTION AND CONSTRAINS:=====================

class ObjectiveFormulation():
    """
    Objective function and constraints for the optimization problem.
    """
    def __init__(self, x_r=np.array([0]),device='cpu'):

        # Ad: system matrix
        # Bd: control matrix
        # dt: time step
        # Q: state cost matrix
        # R: control cost matrix
        # x_r: reference trajectory
        # theta: parameter for constraint


        # % read A,B,C,D matrices:
        self.dt = 0.1
        self.Ad = torch.tensor([[1, self.dt], [0, 1]]).to(device)
        self.Bd=np.array([[0],[0]])
        self.Bd = np.float32(self.Bd)
        self.Bd = torch.tensor(self.Bd).to(device)

        self.Q = torch.tensor([[2.,0],[0.,2.]]).to(device)
        self.R = torch.tensor([0.1]).to(device)


        self.x_r = torch.tensor(x_r).to(device)
        

        self.c = torch.tensor(1e3).to(device)
        self.elu = torch.nn.ReLU()


    def forward(self, x, nn_output):
        """
        Forward pass for the objective function and constraints.
        :param data: input data, shape (batch_size, 6)
        :return: objective values and constraints
        """
        # input x: [x1,x2,xr]
        # nn_output: [p1, p2, p3, u, theta]
        
        # data order: x1, x2, u, p1, p2, p3, theta

        data = torch.cat((x[:,:2], nn_output[:,3].unsqueeze(1),nn_output[:,:3],nn_output[:,-1].unsqueeze(1)), dim=1)
        # Objective values
        obj_values = self.obj_fun(data)


        # Constraints
        constraint1 = self.constrain1(data)
        constraint2 = self.constrain2(data)
        constraint3 = self.constrain3(data)

        # if obj_values.mean()<0 or constraint1.mean()<0 or constraint2.mean()<0:
        #     print(f'Objective values: {obj_values.mean()}, Constraint1: {constraint1.mean()}, Constraint2: {constraint2.mean()}')
        #     obj_values = self.obj_fun(data)

        #     # Constraints
        #     constraint1 = self.constrain1(data)
        #     constraint2 = self.constrain2(data)


        return obj_values.mean() + constraint1.mean() + constraint2.mean() + constraint3.mean()


    def obj_fun(self,data):
        # expend for objective values.
        P = torch.stack([data[:, 3:5], data[:, 4:6]], dim=1)

        theta = torch.stack([data[:, 6]], dim=1).unsqueeze(-1)
        Add = torch.tile(self.Ad, (data.shape[0], 1, 1))
        Add[:,1,0] = 3*self.dt*(data[:,0]**2)

        Bdd = torch.tile(self.Bd, (data.shape[0], 1, 1))
        Bdd[:,1,0] = self.dt*(data[:,1]**2+1)
        x = data[:,:2].reshape(data.shape[0],2,1)
        u = data[:,2].reshape(data.shape[0],1,1)
        # x_rr = data[:,6].reshape(data.shape[0],1,1)
        x_rr = torch.tile(self.x_r,(data.shape[0], 1, 1))
        QQ = torch.tile(self.Q, (data.shape[0], 1, 1))
        x_dag = torch.permute(Add@x+Bdd@u-x_rr, (0, 2, 1))
        # OBJECTIVE FUNCTION
        y = self.R*u**2 + x_dag @ QQ @ (Add@x+Bdd@u-x_rr)+ (x-x_rr).permute(0,2,1)@P@(x-x_rr) + torch.exp(-theta) 
        y = y.squeeze(-1)
        
        return y        


    def constrain1(self,x):
        # input shape: (bs,6): arg order: x1, x2, u, p1, p2, p3, theta
        P = torch.stack([x[:, 3:5], x[:, 4:6]], dim=1)

        eig_values,_ = torch.linalg.eigh(P)
        # two eigen values, sum them up.
        torch.cuda.empty_cache()

        return self.c*torch.sum(self.elu(1e-15-eig_values),dim=1).unsqueeze(-1)
        # return self.c*torch.sum(torch.sign(torch.relu(1e-15-eig_values)),dim=1).unsqueeze(-1)


    # paper Equation for reference, Not applied this moment.
    def constrain2(self,args):

        P = torch.stack([args[:, 3:5], args[:, 4:6]], dim=1)
        x = torch.stack([args[:, 0], args[:, 1]], dim=1).unsqueeze(-1)
        u = torch.stack([args[:, 2]], dim=1).unsqueeze(-1)
        theta = torch.stack([args[:, 6]], dim=1).unsqueeze(-1)

        x_prime = (x-self.x_r).permute(0, 2, 1)

        Add = torch.tile(self.Ad, (args.shape[0], 1, 1))
        Add[:,1,0] = 3*self.dt*(args[:,0]**2)

        Bdd = torch.tile(self.Bd, (args.shape[0], 1, 1))
        Bdd[:,1,0] = self.dt*(args[:,1]**2+1)

        x_dag = torch.permute(Add@x+Bdd@u-self.x_r, (0, 2, 1))

        term_one = torch.sqrt(torch.abs(x_dag@P@ (Add@x+Bdd@u-self.x_r))) - torch.sqrt(torch.abs(x_prime@P@(x-self.x_r)))
        # term_one = torch.sqrt(x_dag@P@ (Add@x+Bdd@u-x_r)) - torch.sqrt(x_prime@P@(x-x_r))
        term_two = theta*torch.sqrt(torch.matmul(x_prime,x-self.x_r))

        return self.c*self.elu((term_one+term_two).squeeze(-1))
        # return self.c*torch.sign(torch.relu((term_one+term_two).squeeze(-1)))

    def constrain3(self,x):
        # input shape: (bs,6): arg order: x1, x2, u, p1, p2, p3, theta

        # force theta to be positive
        theta = torch.stack([x[:, 6]], dim=1).unsqueeze(-1)

        torch.cuda.empty_cache()

        return self.c*torch.sum(self.elu(1e-15-theta),dim=1).unsqueeze(-1)



# test code with reliable dataset--generated by multi-agent method.

'''
from datagenerator import Data_Purify

datapath = 'mathmatical_simulation/dataset/MM_DiffSys_dataset.csv'

data_tmp = Data_Purify(datapath,np.array([0]),p_max=10,u_max=1e5)

# import dataset 
input_data_valid,label_data_valid = data_tmp.purified_data()

input_data_valid = torch.tensor(input_data_valid, dtype=torch.float32)
label_data_valid = torch.tensor(label_data_valid, dtype=torch.float32)

# Initialize the objective formulation
objective_formulation = ObjectiveFormulation()
# Forward pass through the objective formulation
output = objective_formulation.forward(input_data_valid, label_data_valid)
print("Output of the objective formulation:", output.item())

'''


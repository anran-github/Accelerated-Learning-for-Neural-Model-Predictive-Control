import torch
import torch.nn as nn



class P_Net(nn.Module):
    def __init__(self, input_size=3, hidden_size1=128, hidden_size2=256, output_size=4):
    # def __init__(self, input_size=3, hidden_size1=64, hidden_size2=128, output_size=4):
        super(P_Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2*2)
        self.fc2_1 = nn.Linear(hidden_size2*2, hidden_size2*4)
        self.fc2_1_1 = nn.Linear(hidden_size2*4, hidden_size2*4)
        self.fc2_1_2 = nn.Linear(hidden_size2*4, hidden_size2*2)
        self.fc2_2 = nn.Linear(hidden_size2*2, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.softplus = nn.Softplus()  # Smooth positive activation
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2_1(x))
        x = torch.relu(self.fc2_1_1(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc2_1_2(x))
        x = torch.relu(self.fc2_2(x))
        # x = self.dropout(x)

        x = self.fc3(x)

        # Ensure the output is in the desired format
        # # output order: [p1, p2, p3, u, theta]
        # p_part = self.softplus(x[:, :3])     # p1, p2, p3
        # u_part = x[:, 3:4]                   # u (can be negative)
        # theta_part = self.softplus(x[:, 4:5])  # theta (should be positive)

        # x = torch.cat([p_part, u_part, theta_part], dim=1)       


        return x.view(x.size(0), -1)  # Reshape back to (2, n)

class P_Net_ori(nn.Module):
    def __init__(self, input_size=3, hidden_size1=128, hidden_size2=256, output_size=4):
    # def __init__(self, input_size=3, hidden_size1=64, hidden_size2=128, output_size=4):
        super(P_Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2*2)
        self.fc2_1 = nn.Linear(hidden_size2*2, hidden_size2*4)
        self.fc2_1_1 = nn.Linear(hidden_size2*4, hidden_size2*4)
        self.fc2_1_2 = nn.Linear(hidden_size2*4, hidden_size2*2)
        self.fc2_2 = nn.Linear(hidden_size2*2, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2_1(x))
        x = torch.relu(self.fc2_1_1(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc2_1_2(x))
        x = torch.relu(self.fc2_2(x))
        # x = self.dropout(x)

        x = self.fc3(x)
        
        return x.view(x.size(0), -1)  # Reshape back to (2, n)

'''
# Define input and output sizes
input_size = 3
output_size = 4

# Define batch size (n)
batch_size = 5

# Create the fully connected layer model
net = P_Net()
print(net)
# Generate some random input data for demonstration
input_data = torch.randn(batch_size,3)

# Apply the fully connected layer
output = net(input_data)

# Print the output shape
print(output.shape)

'''
import torch
import torch.onnx
import numpy as np

from network import P_Net





model = P_Net(output_size=30)

# test saved model
# Create an instance of the model
# Load the saved model's state dictionary
# model.load_state_dict(torch.load('trained_model_theta0.01_0.21945167709165908.pth'))
model.load_state_dict(torch.load('Heater_Results/uniform_S30_vshape_Iter20_Epoch10.pth'))

# Set the model to evaluation mode (if needed)
model.eval()


# Define example inputs (you may need to adjust the shape and dtype)
# range for x and y direction: x1 [-1,1], x2 [-1,1]
# x1 = 2 * np.random.rand() - 1
# x2 = 2 * np.random.rand() - 1
# r = np.random.rand() - 0.1

# z direction
x1 = np.random.rand() 
x2 = 10 * np.random.rand() 
r = 10*np.random.rand() 

x0 = torch.tensor([[x1],[x2],[r]]).reshape(1,3)
# Define example inputs (you may need to adjust the shape and dtype)
# example_input = torch.randn(1, input_channels, height, width).to(torch.float32)



# Export the model to ONNX
# torch.onnx.export(model, example_input, 'model.onnx', opset_version=12)


# Export the model
torch.onnx.export(model,               # model being run
                  x0,                         # model input (or a tuple for multiple inputs)
                  "Heater_Results/uniform_S10_erf_Iter20_Epoch10.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

print('Done!')
import matplotlib.pyplot as plt
import numpy as np
import os

x = np.arange(1, 201)
train_loss_set = np.random.rand(200) * 0.1 + 0.1
train_loss_set_separate = np.random.rand(200,2) * 0.1 + 0.05
test_loss_set = np.random.rand(200) * 0.1 + 0.15

plt.plot(x, train_loss_set, label='Train Loss', color='red',marker='o',linewidth=2)
plt.plot(x, train_loss_set_separate[:,0], label='Train Loss1',linestyle='dashed', color='green')
plt.plot(x, train_loss_set_separate[:,1], label='Train Loss2',linestyle='dashed', color='orange')
plt.plot(x, test_loss_set, label='Test Loss', color='blue',marker='*',linewidth=2)
plt.grid(linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
# plt.savefig(os.path.join('semi_supervison/weights', 'train_test_loss.png'))
plt.show()
plt.close()
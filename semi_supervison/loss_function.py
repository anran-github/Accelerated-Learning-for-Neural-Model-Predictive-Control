import torch



# Define a running average class to normalize losses
class RunningAverage:
    def __init__(self,device='cpu'):
        
        self.memory_length = 30
        self.avg = torch.tensor([]).to(device)

    def update(self, value):

        if self.avg.shape[0]==0:
            self.avg = value.detach()
            self.avg = self.avg.reshape(1,1)


        # value way less than avg or way larger than avg, ignore it
        elif value/(self.avg + 1e-8) < 1e-3 or value/(self.avg + 1e-8) > 1e3:
            return value/value
        

        # elif len(self.avg) == self.memory_length:
        #     self.avg,_ = torch.sort(self.avg,descending=True)
        #     # self.avg = self.avg[1:,:]

        # mean_val = torch.mean(self.avg)

        return value / (self.avg + 1e-8)



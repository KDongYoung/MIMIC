from sklearn.metrics import mean_absolute_percentage_error
import torch.nn as nn  

class MAPELoss(nn.Module):
    def __init__(self, **args):
        super(MAPELoss, self).__init__()
        self.lossfn = mean_absolute_percentage_error
    
    def forward(self, output, target):
        # epsilon = 1e-8  # 분모가 0이 되는 문제 방지
        # loss = torch.abs((target - output) / (target + epsilon))  # MAPE 계산
        # return loss.mean() * 100 
        #  np.mean((target - output.flatten()) / target) * 100
        loss = self.lossfn(output, target) 
        return loss
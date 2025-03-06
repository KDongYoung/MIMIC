import torch
import torch.nn as nn

class MSELoss(nn.MSELoss):
    def __init__(self, **args):
        super(MSELoss, self).__init__()
        self.lossfn = torch.nn.MSELoss()  
    
    def forward(self, anchor_z, positive_z, negative_z, 
                    x_z, x2_z, contrastive_label,
                    anchor_pred, class_label):
        
        loss = self.lossfn(anchor_pred.flatten(), class_label) 
        return loss
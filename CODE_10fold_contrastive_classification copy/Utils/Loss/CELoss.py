import torch
import torch.nn as nn

class CELoss(nn.CrossEntropyLoss):
    def __init__(self, **args):
        super(CELoss, self).__init__()
        self.lossfn = torch.nn.CrossEntropyLoss()  
    
    def forward(self, anchor_z, positive_z, negative_z, 
                    x_z, x2_z, contrastive_label,
                    anchor_pred, class_label):
        
        loss = self.lossfn(anchor_pred, class_label.long()) 
        return loss
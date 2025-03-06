import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.Loss.loss_utils import *


# 총 손실 = Contrastive Loss + Triplet Loss + Regression Loss
                
class ConsTripLoss(nn.CrossEntropyLoss):
    def __init__(self, **args):
        super(ConsTripLoss, self).__init__()
        self.lossfn = torch.nn.CrossEntropyLoss()  

    def forward(self, anchor_z, positive_z, negative_z, 
                    x_z, x2_z, contrastive_label,
                    anchor_pred, class_label):
        
        loss = self.lossfn(anchor_pred.flatten(), class_label) 
        cons_loss = contrastive_loss(x_z, x2_z, contrastive_label)
        trip_loss = triplet_loss(anchor_z, positive_z, negative_z)
        
        return 0.3*loss + 0.3*cons_loss + 0.3*trip_loss
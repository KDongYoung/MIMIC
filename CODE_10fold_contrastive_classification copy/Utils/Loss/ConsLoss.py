import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.Loss.loss_utils import *


class ConsLoss(nn.CrossEntropyLoss):
    def __init__(self, **args):
        super(ConsLoss, self).__init__()
        self.lossfn = torch.nn.CrossEntropyLoss()  

    def forward(self, anchor_z, positive_z, negative_z, 
                    x_z, x2_z, contrastive_label,
                    anchor_pred, class_label):
        
        loss = self.lossfn(anchor_pred, class_label.long()) 
        cons_loss = contrastive_loss(x_z, x2_z, contrastive_label)
        
        return loss + 0.5*cons_loss
import torch.nn as nn
import torch

class shallowLinearModel_3lowlayer(nn.Module):
    def __init__(self, args):
        super(shallowLinearModel_3lowlayer, self).__init__()
        self.fc1 = nn.Linear(in_features= args["feature_num"], out_features=args["feature_num"]*2)
        self.relu1=nn.ReLU()
        self.norm1=nn.BatchNorm1d(args["feature_num"]*2)
        self.dropout1=nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(in_features= args["feature_num"]*2, out_features=args["feature_num"])
        self.relu2=nn.ReLU()
        self.norm2=nn.BatchNorm1d(args["feature_num"])
        self.dropout2=nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(in_features= args["feature_num"], out_features=args['n_classes'])
        self.sigmoid=nn.Sigmoid()
        
        self._init_weights(args["init_weight"])
    
    def _init_weights(self, name):
        # initialize weights
        if name=="he":
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)
                    m.bias.data.fill_(0)
        
        elif name=="xavier":
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x=self.dropout1(self.relu1(self.fc1(x))) 
        x=self.dropout2(self.relu2(self.fc2(x)))
        x=self.fc4(x)
        return x    
   
    
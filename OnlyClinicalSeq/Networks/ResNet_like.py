import torch
import torch.nn as nn
from Networks.network_utils import make_activation

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, d, d_hidden, norm_layer, dr, activation, track_running=True):
        super(BasicBlock, self).__init__()
        
        self.bn0 = norm_layer(d, track_running_stats=track_running)     
        self.dropdout0 = nn.Dropout(p=dr)
        self.dropdout1 = nn.Dropout(p=dr)
        self.activation = make_activation(activation)
        self.fc1 = nn.Linear(d, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d)
        

    def forward(self, x):
       
        identity = x.clone() # out0
        
        out = self.fc1(self.bn0(x))
        out = self.dropdout1(self.activation(out))
        out = self.dropdout0(self.fc2(out))
        out += identity
        
        return out


class Resnet_like(nn.Module):
    def __init__(self, args, n_layers, d, d_hidden_factor, dr, activation, batch_norm=True):
        super(Resnet_like, self).__init__()
        
        self.track_running= True
        self.n_layers=n_layers
        
        self.do_embedding=args["do_embedding"]
        self.cat_num=args['cat_num']
        # vocab_Size = (항상 input_x보다 커야함 +1), embedding_dim=categorical feature 개수
        self.category_embeddings = nn.ModuleList([nn.Embedding(int(args['vocab_sizes'][i])+1, args['embedding_dim']) for i in range(self.cat_num)])
        
        num_classes = args['n_classes']
        input_ch=(args['feature_num']-self.cat_num) + self.cat_num*args['embedding_dim'] # 총 input 개수
        self.batch_norm = batch_norm
        
        self._norm_layer = nn.BatchNorm1d       
        self.Sequential=nn.Sequential
        self.Conv=nn.Linear
        
        self.fc1 = nn.Linear(input_ch, d)
    
        d_hidden = int(d * d_hidden_factor)
        self.layers = nn.ModuleList([BasicBlock(d, d_hidden, self._norm_layer, dr, activation) for _ in range(self.n_layers)])
        
        self.dropout = nn.Dropout(p=dr)
        self.activation = make_activation(activation)
        self.norm = self._norm_layer(d)
        # self.fc = nn.Linear(d, d)
        self.fc_last = nn.Linear(d, num_classes)


    def forward(self, x):          
        # categorical, numerical 나눠서
        # categorical을 nn.Embedding()
        if self.do_embedding and self.cat_num != 0: # all numerical
            # Embed each categorical feature
            embedded_x_cat = [self.category_embeddings[i](x[:, i].long()) for i in range(self.cat_num)]
            embedded_x_cat = torch.cat(embedded_x_cat, dim=1)
            x_num=x[:, self.cat_num:]
            x=torch.cat((embedded_x_cat, x_num), dim=1)
              
        x = self.fc1(x)
        
        for i in range(self.n_layers):
            x = self.layers[i](x)
                
        x = self.activation(self.norm(x))
        x = self.fc_last(x)
        
        return x
    

def resnet_18(args):
    n_layers = 4
    d = args['d'] 
    d_hidden_factor = 0.8
    dr = args['dropout_rate']
    activation = args["activation"]
    return Resnet_like(args, n_layers, d, d_hidden_factor, dr, activation)

def resnet_8(args):
    n_layers = 3
    d = args['d'] # [256, 128, 32]
    d_hidden_factor = 0.8
    dr = args['dropout_rate']
    activation = args["activation"]
    return Resnet_like(args, n_layers, d, d_hidden_factor, dr, activation)

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mimic4')
    parser.add_argument('--n_classes', type=int, default=2, help='num classes')
    parser.add_argument('--n_channels', type=int, default=1, help='num classes')
    parser.add_argument('--batch_size', type=int, default=16, help='num classes')
    args = parser.parse_args()
    args=vars(args)

    model = resnet_8(args) # CLASS, CHANNEL, TIMEWINDOW
    print(model)
    from pytorch_model_summary import summary

    print(summary(model, torch.zeros(1, 1, 1, 600), show_input=True))
            # (1, 1, channel, timewindow)
from sklearn.metrics import mean_absolute_percentage_error

class MAPELoss():
    def __init__(self, **args):
        super(MAPELoss, self).__init__()
        self.lossfn = mean_absolute_percentage_error
    
    def forward(self, output, target):
        
        loss = self.lossfn(output, target) 
        return loss
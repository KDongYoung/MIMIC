import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import time
import pickle

# EVALUATION
from torch.utils.tensorboard import SummaryWriter
from Utils.Loss.MAPELoss import MAPELoss

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # ✅ UserWarning 숨기기


class Trainer():
    def __init__(self, args, model, fold, model_name="MLP"):
        self.args=args
        # self.y_class=self.args["class"]
        self.model=model  
        self.model_type=args['model_type']  
        self.fold = fold
        
        self.max_epoch=self.args['max_epoch']
        self.metric_dict=self.args['metric_dict']
        
        self.writer = SummaryWriter(f"{self.args['total_path']}/tensorboard") # tensorboard, log directory
        self.args["tensorboard"]=self.writer
    
        self.set_optimizer()
        self.set_learning_rate_Scheduler()
        
    ''' Initialize parameter '''
    def set_optimizer(self):
        if self.args['optimizer']=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])
        elif self.args['optimizer']=='AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        elif self.args['optimizer']=='SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args['lr'], momentum=self.args['momentum'], weight_decay=self.args['weight_decay'], nesterov=self.args['nestrov'])
    
    def set_learning_rate_Scheduler(self):
        if self.args['scheduler']=="CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000//self.args['checkpoint'])
                      
    '''
    ###########################################################################################
    #  Train 
    ###########################################################################################
    '''
    
    def training(self,  train_loaders, valid_loader, test_loader):
        print('Training ...')
        best_score = self.training_dl(train_loaders, valid_loader, test_loader)
        return best_score
    
    def training_dl(self,  train_loaders, valid_loader, test_loader):
        best_score = [1000.0, 100.0, 100.0]
        start_step=0
        
        self.criterion = nn.MSELoss()
        self.model.train()
        
        for step in range(start_step, self.args['steps']):
            
            self.mselosses=[]
            self.maelosses=[]
            batch_num=0
            for x,y, _ in train_loaders:
                minibatches_device = [x.to(self.args['device']), y.to(self.args['device'], dtype=torch.float32)]
                self.train(minibatches_device, step, batch_num) # train  
                batch_num+=1 
            print(f'Mean Train MSE Loss: {sum(self.mselosses)/len(self.mselosses):.4f}, MAE Loss: {sum(self.maelosses)/len(self.maelosses):.4f}')
            self.write_tensorboard(phase="train", step=step, mse=sum(self.mselosses)/len(self.mselosses), mae= sum(self.maelosses)/len(self.maelosses))
            
            valid_score = self.eval("valid", valid_loader, step) # valid 
            self.eval("test", test_loader, step) # test              
            self.scheduler.step()    
            
            for metric in self.args['eval_metric']: 
                # compare the validation score with the best_score (from the previous step)
                best_score[self.metric_dict[metric]] = self.compare_metric_save_model(metric, best_score[self.metric_dict[metric]], valid_score[self.metric_dict[metric]])
                               
        return best_score
    
    def train(self, minibatches, step, batch_num):
        self.model.train()
        
        data, target = minibatches
      
        self.optimizer.zero_grad()
        
        output = self.model(data)
        loss = self.criterion(output, target)  
        
        loss.backward()
        self.optimizer.step()
        
        mae_lossfn=nn.L1Loss()
        mae_loss = mae_lossfn(output, target)
        self.mselosses.append(loss)
        self.maelosses.append(mae_loss)
        
        print(f'Train Epoch: {step+1}\t Batch_Num: {batch_num}\t MSE: {loss:.4f}\t MAE: {mae_loss:.4f}')   
            
    '''
    ###########################################################################################
    #  Evaluation
    ###########################################################################################
    '''    
    ## EVALUATE 
    def eval(self, phase, loader, step):
        
        self.model.eval()
        lossfn = torch.nn.MSELoss() 
        
        time_cost=[]
        outputs=[]
        targets=[]
        with torch.no_grad(): 
            for datas in loader:
                s_time=time.time()
                data, target = datas[0].to(self.args['device']), datas[1].to(torch.float32)
                
                targets.append(target)
                
                output = self.model(data)
                e_time=time.time()
                time_cost.append(e_time-s_time)
                
                outputs.append(output)
            
        outputs=torch.cat(outputs)
        targets=torch.cat(targets)
        
        MSE_LOSS = torch.sqrt(lossfn(outputs.cpu(), targets))
        mape_lossfn = MAPELoss()
        MAPE_LOSS = mape_lossfn(outputs.cpu(), targets)
        mae_lossfn = nn.L1Loss()
        MAE_LOSS = mae_lossfn(outputs.cpu(), targets)
        
        print(phase.capitalize(), f'RMSE Loss: {MSE_LOSS.item():.4f}, MAE Loss: {MAE_LOSS.item():.4f}, MAPE Loss: {MAPE_LOSS.item()*100:.2f}')
         
        # # if self.model_type=="DL":  
        self.write_tensorboard(phase=phase, step=step, mse=MSE_LOSS, mape=MAPE_LOSS, mae=MAE_LOSS)
        
        if phase=="valid":
            return MSE_LOSS.item(), MAPE_LOSS.item(), MAE_LOSS.item()
        elif phase=="test":
            return MSE_LOSS.item(), MAPE_LOSS.item(), MAE_LOSS.item(), time_cost
    
    ''' Prediction '''
    def prediction(self, test_loader, metric="f1"):
        ''' Test the best model '''
        print("== "*10, "Testing", "== "*10)
        
        print(f'{metric}:', end=" ")
      
        self.model.load_state_dict(torch.load(os.path.join(self.args['total_path'], 'models', metric, f"{self.fold}_bestmodel"), map_location=self.args['device']))
        if self.args['cuda']: 
            self.model.cuda(device=self.args['device'])
        
        rmse, mape, mae, cost = self.eval("test", test_loader, self.args['steps']+1)
           
        return [rmse, mape, mae, cost]
  

    
    '''
    ###########################################################################################
    #  Etc.
    ###########################################################################################
    '''
    ############## compare valid_score and evaluation metric and save the best model ##################
    def compare_metric_save_model(self, eval_metric, best_score, valid_score):
        ## compare validation accuracy of this epoch with the best accuracy score
        ## if validation loss <= best loss, then save model(.pt)
        if best_score > valid_score:
            best_score = valid_score
            torch.save(self.model.state_dict(), os.path.join(self.args['total_path'], 'models', eval_metric, f"{self.fold}_bestmodel")) # {self.domainList[self.domain_id]}
    
        return best_score
    
    ########################### Tensorboard ###########################
    def write_tensorboard(self, step, phase, mse=0, mape=0, mae=0):

        if phase=='train':
            self.writer.add_scalar(f'{phase}/mse_loss', mse, step)
            self.writer.add_scalar(f'{phase}/mae_loss', mae, step)
        else:
            self.writer.add_scalar(f'{phase}/mse_loss', mse, step)
            self.writer.add_scalar(f'{phase}/mape_loss', mape, step)
            self.writer.add_scalar(f'{phase}/mae_loss', mae, step) 
     
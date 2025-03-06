import numpy as np
import torch
import os
import time
import pickle
import importlib

# EVALUATION
from torch.utils.tensorboard import SummaryWriter

from Utils.Loss.loss_utils import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Trainer():
    def __init__(self, args, model, fold, model_name="MLP"):
        self.args=args
        self.y_class=self.args["class"]
        self.model=model  
        self.model_type=args['model_type']  
        self.fold = fold
        
        self.model_name=model_name
        self.max_epoch=self.args['max_epoch']
        self.metric_dict={"rmse":0, "mape":1}
        
        self.writer = SummaryWriter(f"{self.args['total_path']}/tensorboard") # tensorboard, log directory
        self.args["tensorboard"]=self.writer
                
        if self.model_type == "DL":
            self.set_optimizer()
            self.set_learning_rate_Scheduler()
        else:
            self.model_pkl_file=''

    ''' Initialize parameter '''
    def set_optimizer(self):
        if self.args['optimizer']=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])
        elif self.args['optimizer']=='AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        elif self.args['optimizer']=='SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args['lr'], momentum=self.args['momentum'], weight_decay=self.args['weight_decay']) #, nesterov=self.args['nestrov'])
    
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
        if self.model_type=="DL":
            best_score = self.training_dl(train_loaders, valid_loader, test_loader)
        elif self.model_type=="ML":
            best_score = self.training_ml(train_loaders, test_loader)
        return best_score
    
    def training_ml(self,  train_loaders, test_loader):
        for x_train, y_train, _, result in train_loaders: #### 여끼뿌터 오류!!!
            mask = result == 1
            x_train = x_train[mask]
            y_train = y_train[mask]
            self.model.fit(x_train, y_train)

        self.model_pkl_file=f"{self.args['total_path']}/models/{self.args['model_name']}_{self.args['normalize']}_{self.args['target']}_{self.fold}fold.pkl"
        with open(self.model_pkl_file, 'wb') as file:  
            pickle.dump(self.model, file)
            
        rmse, mape, cost= self.eval("test", test_loader, step=self.args["steps"]+1)            
        return [rmse, mape, cost]
    
    def training_dl(self,  train_loaders, valid_loader, test_loader):
        best_score = [1000.0, 100.0]
        start_step=0
    
        Loss = importlib.import_module('Utils.Loss.'+self.args['loss'])
        self.criterion = getattr(Loss, self.args['loss'])(**self.args)
        self.model.train()
        
        for step in range(start_step, self.args['steps']):
            
            self.losses=[]
            batch_num=0
            
            for batch in train_loaders:
                anchor, positive, negative, x, x2, contrastive_label, class_label = batch  # 성공 여부 레이블은 필요 없음

                # 모든 데이터를 device로 이동
                anchor, positive, negative, x, x2, contrastive_label, class_label = map(lambda t: t.to(self.args['device']), 
                                                                                        (anchor, positive, negative, x, x2, contrastive_label, class_label))

                # contrastive
                x_z, _ = self.model(x)  
                x2_z, _ = self.model(x2) 
                # Triplet
                anchor_z, anchor_pred = self.model(anchor)
                positive_z, _ = self.model(positive)
                negative_z, _ = self.model(negative)

                # regression만, regression + contrastive, regression + contrastive + triplet
                loss = self.criterion(anchor_z, positive_z, negative_z, 
                                      x_z, x2_z, contrastive_label,
                                      anchor_pred, class_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()  
                batch_num+=1 
                self.losses.append(loss)
                print(f'Train Epoch: {step+1}\t Batch_Num: {batch_num}\t Loss: {loss:.4f}')   
         
            print(f'Mean Train Loss: {sum(self.losses)/len(self.losses)}')
            # self.write_tensorboard(phase="train", step=step, loss=sum(self.losses)/len(self.losses))
            
            valid_score = self.eval("valid", valid_loader, step) # valid 
            self.eval("test", test_loader, step) # test              
            self.scheduler.step()    
            
            for metric in self.args['eval_metric']: 
                # compare the validation score with the best_score (from the previous step)
                best_score[self.metric_dict[metric]] = self.compare_metric_save_model(metric, best_score[self.metric_dict[metric]], valid_score[self.metric_dict[metric]])
             
        self.writer.close()
        
        return best_score
        
    '''
    ###########################################################################################
    #  Evaluation
    ###########################################################################################
    '''    
    ## EVALUATE 
    def eval(self, phase, loader, step, metric=None):
        if self.model_type=="DL":
            self.model.eval()
        lossfn = torch.nn.MSELoss() 
        
        time_cost=[]
        outputs=[]
        targets=[]
        loss=torch.tensor(0)
        results = []
        
        with torch.no_grad(): 
            for datas in loader:
                s_time=time.time()
                data, target, result = datas[0].to(self.args['device']), datas[1], datas[3]
                # , dtype=torch.int64

                targets.append(target)
                results.append(result)
                
                if self.model_type == "DL":
                    _, output_pred = self.model(data)
                    e_time=time.time()
                    time_cost.append(e_time-s_time)
                    
                    outputs.append(output_pred)
                else:
                    output = self.model.predict(data.cpu())
                    e_time=time.time()
                    time_cost.append(e_time-s_time)
                    
                    outputs.append(torch.tensor(output))
                    
        outputs=torch.cat(outputs)
        targets=torch.cat(targets)
        results=torch.cat(results)
        
        mask = results == 1
        outputs_success = outputs[mask]
        targets_success = targets[mask]
        
        loss = torch.sqrt(lossfn(outputs_success.cpu(), targets_success))
        loss2 = mape_loss(outputs_success.cpu(), targets_success)
        
        if self.model_type=="ML":   
            time_cost = time_cost[0]
        
        print(phase.capitalize(), f'RMSE Loss: {loss.item():.4f}, MAPE Loss: {loss2.item():.4f}')
         
        # self.write_tensorboard(phase=phase, step=step)
        
        if phase=="valid":
            return loss.item(), loss2.item()
        elif phase=="test":
            return loss.item(), loss2.item(), time_cost
    
    ''' Prediction '''
    def prediction(self, test_loader, metric="f1"):
        ''' Test the best model '''
        print("== "*10, "Testing", "== "*10)
        
        print(f'{metric}:', end=" ")
        
        if self.model_type=="DL":
            self.model.load_state_dict(torch.load(os.path.join(self.args['total_path'], 'models', metric ,f"{self.fold}_bestmodel"), map_location=self.args['device']))
            if self.args['cuda']: 
                self.model.cuda(device=self.args['device'])
        
        elif self.model_type=="ML":  
            self.model_pkl_file=f"{self.args['total_path']}/models/{self.args['model_name']}_{self.args['normalize']}_{self.args['target']}_{self.fold}fold.pkl"
            
            try:
                with open(self.model_pkl_file, 'rb') as file:
                    self.model = pickle.load(file)
            except EOFError:
                print("Error: The file is either empty or corrupted.")

        rmse, mape, cost = self.eval("test", test_loader, self.args['steps']+1, metric)
           
        return [rmse, mape, cost]
  
    
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
            torch.save(self.model.state_dict(), os.path.join(self.args['total_path'], 'models', eval_metric, f"{self.fold}_bestmodel"))

        return best_score
    
    ########################### Tensorboard ###########################
    def write_tensorboard(self, step, phase, loss=0, acc=0, f1=0, roc_auc=0, sensitivity=0, lr=0, specificity=0):

        if phase=='train':
            self.writer.add_scalar(f'{phase}/lr2', lr, step)
            self.writer.add_scalar(f'{phase}/acc', acc, step)
            self.writer.add_scalar(f'{phase}/loss', loss, step)
        else:
            self.writer.add_scalar(f'{phase}/loss', loss, step)
            self.writer.add_scalar(f'{phase}/acc', acc, step)
            self.writer.add_scalar(f'{phase}/f1score', f1, step)
            self.writer.add_scalar(f'{phase}/roc_auc', roc_auc, step) 
            self.writer.add_scalar(f'{phase}/recall_sensitivity', sensitivity, step) 
            self.writer.add_scalar(f'{phase}/recall_specificity', specificity, step) 

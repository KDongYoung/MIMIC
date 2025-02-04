import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import time
import pickle

# EVALUATION
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, roc_auc_score, recall_score, average_precision_score
from sklearn.preprocessing import LabelBinarizer     
from torch.utils.tensorboard import SummaryWriter
from Utils.Evaluation_utils import specificity_score, save_confusion_matrix

class Trainer():
    def __init__(self, args, domain_id, domainList, model):
        self.args=args
        self.y_class=self.args["n_classes"]
        self.model=model  
        self.model_type=args['model_type'] 
        self.domain_id = domain_id
        self.domainList = domainList 
        self.domains_name = self.domainList[self.domain_id]
        
        if self.args["n_classes"] !=2:
            # Multi-class
            self.lb = LabelBinarizer()
            self.lb.fit(range(self.args['n_classes']))

        self.max_epoch=self.args['max_epoch']
        self.metric_dict={"loss":0, "acc":1, "bacc":2, "f1":3, "recall":4, 'auroc':5, 'auprc':6, 'mean_sensi':7, 'specificity':8}

        self.writer = SummaryWriter(f"{self.args['total_path']}/{self.domains_name}") # tensorboard, log directory
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
        if self.model_type=="DL":
            best_score = self.training_dl(train_loaders, valid_loader, test_loader)
        elif self.model_type=="ML":
            best_score = self.training_ml(train_loaders, test_loader)
        return best_score
    
    def training_ml(self,  train_loaders, test_loader):
        for x_train, y_train, _, _ in train_loaders:
            self.model.fit(x_train, y_train)

        self.model_pkl_file=f"{self.args['total_path']}/models/{self.args['eval_metric'][0]}/{self.domains_name}_bestmodel.pkl"
        with open(self.model_pkl_file, 'wb') as file:  
            pickle.dump(self.model, file)
            
        loss, acc, bacc, f1score, specificity, sensitivity, mean_sensitivity, precision, auroc, auprc, cost = self.eval("test", test_loader, step=self.args["steps"]+1)            
        return [loss, acc, bacc, f1score, specificity, sensitivity, mean_sensitivity, precision, auroc, auprc, cost]
    
    def training_dl(self,  train_loaders, valid_loader, test_loader):
        best_score = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        start_step=0
        
        self.criterion = nn.CrossEntropyLoss()
        
        for step in range(start_step, self.args['steps']):
            
            self.losses=[]
            batch_num=0
            for x,y in train_loaders:
                minibatches_device = [x.to(self.args['device']), y.to(self.args['device'])]
                self.train(minibatches_device, step, batch_num) # train  
                batch_num+=1 
            print(f'Mean Train Loss: {sum(self.losses)/len(self.losses)}')
            self.write_tensorboard(phase="train", step=step, loss=sum(self.losses)/len(self.losses))
            
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
        
        pred = output.argmax(dim=1, keepdim=True)        
        
        loss.backward()
        self.optimizer.step()
        acc=accuracy_score(target.cpu().numpy(), pred.cpu().numpy())
        
        self.losses.append(loss)
        print('Train Epoch: {}\t Batch_Num: {}\t Loss: {:.4f}\t ACC: {:.4f}'.format(step+1, batch_num+1, loss, acc))   
            
    '''
    ###########################################################################################
    #  Evaluation
    ###########################################################################################
    '''    
    ## EVALUATE 
    def eval(self, phase, loader, step, metric=None):
        if self.model_type=="DL":
            self.model.eval()
            lossfn = torch.nn.CrossEntropyLoss() 
        
        time_cost=[]
        outputs=[]
        targets=[]
        preds=[]
        loss=torch.tensor(0)
        
        with torch.no_grad(): 
            for datas in loader:
                s_time=time.time()
                data, target = datas[0].to(self.args['device']), datas[1].to(self.args['device'], dtype=torch.int64)
                
                targets.append(target)
                
                if self.model_type == "DL":
                    output = self.model(data)
                    e_time=time.time()
                    time_cost.append(e_time-s_time)
                    
                    outputs.append(output)
                    preds.append(output.argmax(dim=1, keepdim=False)) 
                    
                else:
                    output = self.model.predict(data.cpu())
                    e_time=time.time()
                    time_cost.append(e_time-s_time)
                    pred=output
                    
                    outputs.append(torch.tensor(output))
                    preds.append(torch.tensor(pred)) 
                    
        outputs=torch.cat(outputs)
        preds=torch.cat(preds)
        targets=torch.cat(targets)
        # domains=torch.cat(domains)
        
        if phase == 'test':
            with open(f"{self.args['total_path']}/test_predicts/{self.domains_name}.txt", 'a') as f:
                f.write(f"Wrong: {' '.join(map(str, (preds != targets).nonzero()[0]))}\n") # 틀리게 맞추는 애들
                f.write(f"Correct: {' '.join(map(str, (preds == targets).nonzero()[0]))}") # 옳게 맞추는 애들
            
        
        if self.model_type=="DL":          
            loss = lossfn(outputs, targets) 
        else:
            time_cost = time_cost[0]
            
        targets=targets.cpu().numpy()
        preds=preds.cpu().numpy()
        
        # draw confusion matrix
        if phase == "test" and step == self.args['steps']+1 and self.args["save_confusion_matrix"]: 
            save_confusion_matrix(targets=targets, preds=preds, save_path=self.args['total_path'], metric=metric)
        
        acc=accuracy_score(targets, preds)
        bacc=balanced_accuracy_score(targets, preds)
        
        if self.args['n_classes'] == 2: 
            f1=f1_score(targets, preds)
            precision=precision_score(targets,preds, zero_division=0) # recall_score(targets, preds, pos_label=0) # 
            auroc=roc_auc_score(targets, preds)
            sensi=recall_score(targets, preds)
            mean_sensi = recall_score(targets, preds) # mean sensitivity (macro)
            auprc=average_precision_score(targets, preds)
            speci = specificity_score(targets, preds)  
            
        elif len(np.unique(targets)) < self.args['n_classes']:
            print(f"{phase.capitalize()} #### Cannot calculate score with less than {self.args['n_classes']} classes.")
            f1, precision, auroc, mean_sensi, sensi, auprc, speci = 0, 0, 0, 0, 0, 0, 0
            
        else:
            f1 = f1_score(targets, preds, average='macro')
            precision = precision_score(targets, preds, average='macro', zero_division=0) # recall_score(targets, preds,  pos_label=0) #
            auroc = roc_auc_score(self.lb.transform(targets), self.lb.transform(preds), average='macro', multi_class='ovr')
            mean_sensi = recall_score(targets, preds, average='macro') # mean sensitivity (macro)
            sensi = recall_score(targets, preds, average='micro') # mean sensitivity (micro)
            auprc = 0 # multi-class no      
            speci = specificity_score(targets, preds)      

        print(phase.capitalize(), f' Loss: {loss.item():.4f}, Acc: {acc:.4f}%, F1: {f1:.4f}, Precision: {precision:.4f}, Specificity: {speci:.4f},',
                f'Sensitiviy(Recall): {sensi:.4f}, Mean Sensitiviy: {mean_sensi:.4f}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}')
        
        # if self.model_type=="DL":  
        self.write_tensorboard(phase=phase, step=step, acc=acc, loss=loss, f1=f1, roc_auc=auroc, sensitivity=mean_sensi, specificity=speci)
        
        if phase=="valid":
            return loss.item(), acc, bacc, f1, speci, sensi, mean_sensi, auroc, auprc
        elif phase=="test":
            return loss.item(), acc, bacc, f1, speci, sensi, mean_sensi, precision, auroc, auprc, time_cost
    
    ''' Prediction '''
    def prediction(self, test_loader, metric="f1"):
        ''' Test the best model '''
        print("== "*10, "Testing", "== "*10)
        
        print(f'{metric}:', end=" ")
        
        if self.model_type=="DL":
            self.model.load_state_dict(torch.load(os.path.join(self.args['total_path'], 'models', metric, f"{self.domainList[self.domain_id]}_bestmodel"), map_location=self.args['device']))
            if self.args['cuda']: 
                self.model.cuda(device=self.args['device'])
        
        elif self.model_type=="ML":  
            self.model_pkl_file=f"{self.args['total_path']}/models/{metric}/{self.domainList[self.domain_id]}_bestmodel.pkl"
            try:
                with open(self.model_pkl_file, 'rb') as file:
                    self.model = pickle.load(file)
            except EOFError:
                print("Error: The file is either empty or corrupted.")

        loss, acc, bacc, f1score, specificity, sensitivity, mean_sensitivity, precision, auroc, auprc, cost = self.eval("test", test_loader, self.args['steps']+1, metric)
           
        return [loss, acc, bacc, f1score, specificity, sensitivity, mean_sensitivity, precision, auroc, auprc, cost]
  

    
    '''
    ###########################################################################################
    #  Etc.
    ###########################################################################################
    '''
    ############## compare valid_score and evaluation metric and save the best model ##################
    def compare_metric_save_model(self, eval_metric, best_score, valid_score):
        ## compare validation accuracy of this epoch with the best accuracy score
        if eval_metric=="loss":
            ## if validation loss <= best loss, then save model(.pt)
            if best_score > valid_score:
                best_score = valid_score
                torch.save(self.model.state_dict(), os.path.join(self.args['total_path'], 'models', eval_metric, f"{self.domainList[self.domain_id]}_bestmodel"))
        else:
            ## if validation accuracy >= best accuracy, then save model(.pt)
            if valid_score >= best_score:
                best_score = valid_score
                torch.save(self.model.state_dict(), os.path.join(self.args['total_path'], 'models', eval_metric, f"{self.domainList[self.domain_id]}_bestmodel"))
        
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
     
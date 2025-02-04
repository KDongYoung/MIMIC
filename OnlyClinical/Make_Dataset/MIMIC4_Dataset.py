import pandas as pd
import os
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

class MIMIC4_Dataset():
    def __init__(self, args) -> None:
        self.args=args
        self.hosp_path = args['data_root']+"hosp/"
        self.icu_path = args['data_root']+"icu/"        
        self.disease_name=args['disease_name']
        self.save_path=args['save_data']

        self.dataset={}
        if not os.path.isdir(f"{self.args['save_data']}"):
            os.makedirs(self.args['save_data'])
            
        save_data_list=os.listdir(self.args['save_data'])
       
        self.ditems()
        self.basic() 
            
        if "icd.csv" not in save_data_list:
            self.diagnoses_icd() 

        if "inputevents.csv" not in save_data_list:
            self.inputevents()  
        if "outputevents.csv" not in save_data_list:
            self.outputevents()  
        if "chartevents.csv" not in save_data_list:
            self.chartevents()
          
        if "labevents.csv" not in save_data_list and self.args['labevents']:
            self.labevents()
        if "microbiologyevents.csv" not in save_data_list and self.args['microbiologyevents']:
            self.microbiologyevents()                           
              
        self.args["name"]=[n.split(".")[0] for n in save_data_list if "_" not in n and "args" not in n and ".csv" in n]
        
        if "column_info.csv" not in save_data_list:
            self.args["column_info"]=self.column_dictionary
            pd.DataFrame(self.column_dictionary, index=[0]).T.reset_index().to_csv(f'{self.save_path}/column_info.csv', index=False)
            
        with open(self.args['save_data'] + '/args.txt', 'w') as f:
            f.write(datetime.now().strftime('%Y%m%d')+"\n")
            f.write("\n".join(self.args["name"]))
    
    
    def ditems(self):
        from Make_Dataset.MIMIC4.mimic_ditems import ditems
        self.d_items, self.column_dictionary, self.contain_label = ditems(self.icu_path)
    
    def basic(self):
        from Make_Dataset.MIMIC4.mimic_basic import basic
        self.icu_stay, self.admission, self.args, self.column_dictionary = basic(self.icu_path, self.hosp_path, self.save_path, self.args, self.column_dictionary)

    def diagnoses_icd(self):
        from Make_Dataset.MIMIC4.mimic_diagnosesicd import diagnoses_icd
        return diagnoses_icd(self.hosp_path, self.save_path)
    
    def inputevents(self):
        from Make_Dataset.MIMIC4.mimic_inputevents import inputevents
        return inputevents(self.icu_path, self.save_path, self.icu_stay, self.d_items, self.column_dictionary)
    
    def outputevents(self):
        from Make_Dataset.MIMIC4.mimic_outputevents import outputevents
        return outputevents(self.icu_path, self.save_path, self.icu_stay, self.d_items)
    
    def chartevents(self):
        from Make_Dataset.MIMIC4.mimic_chartevents import chartevents, chart_height_weight
        height_weight_df, self.column_dictionary = chartevents(self.icu_path, self.save_path, self.icu_stay, self.d_items, self.column_dictionary)
        self.args = chart_height_weight(self.hosp_path, self.save_path, height_weight_df, self.args)  # 키와 몸무게 통일

    def labevents(self):
        from Make_Dataset.MIMIC4.mimic_labevents import labevents
        return labevents(self.hosp_path, self.save_path, self.icu_stay, self.admission, self.column_dictionary)
    
    def microbiologyevents(self):
        from Make_Dataset.MIMIC4.mimic_microbiologyevents import microbiologyevents
        return microbiologyevents(self.hosp_path, self.save_path, self.icu_stay, self.admission)
    
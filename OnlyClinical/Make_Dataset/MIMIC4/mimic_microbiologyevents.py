import pandas as pd
import gzip
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

from Make_Dataset.MIMIC4.mimic_utils import fill_nan_hadm_id

"""
###########################################
###     사용하고자 할때,,, 재수정 필요    ###
###########################################
"""

def microbiologyevents(hosp_path, save_path, icu_stay, admission):
    print(f"Preprocess microbiologyevents.csv", end=" ")
    start=datetime.now()

    microbiologyevents = pd.read_csv(gzip.open(f'{hosp_path}/microbiologyevents.csv.gz'), low_memory=False)
    
    icu_stay['admittime'] = pd.to_datetime(icu_stay['admittime'])
    icu_stay['dischtime'] = pd.to_datetime(icu_stay['dischtime'])
    microbiologyevents['chartdate'] = pd.to_datetime(microbiologyevents['chartdate'])
    microbiologyevents['storedate'] = pd.to_datetime(microbiologyevents['storedate'])

    ## fill nan hadm_id
    microbiologyevents = fill_nan_hadm_id(admission, microbiologyevents)
    
    # merged = pd.merge(self.icu_stay.loc[:, ['subject_id', 'hadm_id', 'admittime', 'dischtime']], microbiologyevents, on='subject_id', suffixes=('_icu', '_micro'))
    # filtered = merged[(merged['admittime'] <= merged['storedate']) & (merged['storedate'] <= merged['dischtime'])] # admittime, dischtime 사이에 처방된 것만 선택

    # update_dict = filtered.set_index(['subject_id', 'storedate'])['hadm_id_icu'].to_dict()
    # # Define a function to update hadm_id using the dictionary
    # def update_hadm_id(row):
    #     return update_dict.get((row['subject_id'], row['storedate']), row['hadm_id'])
    # microbiologyevents['hadm_id'] = microbiologyevents.apply(update_hadm_id, axis=1)

    ## 박테리아 발견
    check_infectious_growth = microbiologyevents.loc[microbiologyevents["ab_itemid"].isnull(), 
                                                ['subject_id', 'hadm_id', 'spec_itemid', 'spec_type_desc', 'test_itemid', 'test_name', 'org_itemid', 'org_name']]
    ## 항생제 투여 
    antibiotic_treatment = microbiologyevents.loc[(microbiologyevents["ab_itemid"].notnull())&((microbiologyevents["interpretation"]=="S")|(microbiologyevents["interpretation"]=="R")), 
                                                ['subject_id', 'hadm_id', 'ab_itemid', 'ab_name', 'dilution_text', 'dilution_comparison', 'dilution_value', 'interpretation']]
    # isolated colony가 뭐지... (이따가 질문하고 우선 빼고 생각하기)
    
    # org_itemid null이면 negative: fill 0 // org_itemid not null이면 positive: fill 1 => EXCEPT 'CANCELED'
    check_infectious_growth.loc[(check_infectious_growth["org_name"].notnull()) & (check_infectious_growth["org_name"]!="CANCELLED"), "reaction"]=1
    check_infectious_growth.loc[check_infectious_growth["org_name"].isnull(), "reaction"]=0
    check_infectious_growth=check_infectious_growth[check_infectious_growth["org_name"]!="CANCELED"] # 결과 나온 SAMPLE들만 선택       
    
    # 시험약, 배지 합치기
    check_infectious_growth["specimen"]=check_infectious_growth["spec_type_desc"]+ " // " +check_infectious_growth["test_name"]
    
    check_infectious_growth = pd.DataFrame(check_infectious_growth.groupby(['subject_id', 'hadm_id', 'specimen'])['reaction'].mean()).reset_index()
    check_infectious_growth_pivot=check_infectious_growth.pivot(index=['subject_id', 'hadm_id'], columns='specimen', values='reaction').reset_index()
    check_infectious_growth_pivot.to_csv(f"{save_path}/microbiologyevents_infect.csv", index=False)
    
    antibiotic_treatment['interpretation'].replace({'R':0, 'S':1}, inplace=True)
    antibiotic_treatment['dilution_comparison'].replace({'<=        ':0, '=         ':1, '=>        ':2}, inplace=True)
    
    antibiotic_treatment = pd.DataFrame(antibiotic_treatment.groupby(['subject_id', 'hadm_id', 'ab_name'])['dilution_comparison', 'dilution_value', 'interpretation'].mean()).reset_index()
    antibiotic_treatment_pivot=antibiotic_treatment.pivot(index=['subject_id','hadm_id'], columns='ab_name', values=['dilution_comparison', 'dilution_value', 'interpretation'])
    antibiotic_treatment_pivot.columns=['_'.join(col) for col in antibiotic_treatment_pivot.columns]
    antibiotic_treatment_pivot=antibiotic_treatment_pivot.reset_index()
    antibiotic_treatment_pivot.to_csv(f"{save_path}/microbiologyevents_anti.csv", index=False)
    
    print(f'{datetime.now()-start} (hh:mm:ss.ms)')
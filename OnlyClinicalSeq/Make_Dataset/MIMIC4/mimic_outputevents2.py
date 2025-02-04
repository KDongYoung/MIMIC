import pandas as pd
import gzip
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거


def outputevents(icu_path, save_path, d_items, column_dictionary):
    print(f"Preprocess outputevents.csv", end=" ")
    start=datetime.now()
    
    subject = pd.read_csv(f'{save_path}/mimic4_subject.csv')
    d_items=d_items.loc[d_items["linksto"]=='outputevents', :]
    outputevents = pd.read_csv(gzip.open(f'{icu_path}/outputevents.csv.gz'), low_memory=False)
    outputevents = outputevents[['subject_id', 'hadm_id', 'charttime', 'itemid', 'value']]
    outputevents = outputevents[(outputevents['subject_id'].isin(subject['subject_id'])) & 
                                (outputevents['hadm_id'].isin(subject['hadm_id']))]
    
    cols = ['charttime']
    outputevents[cols] = outputevents[cols].apply(lambda col: pd.to_datetime(col))
    outputevents = outputevents.sort_values(by=['subject_id', 'hadm_id', 'charttime'])   ################################################################ 나중에 한줄씩 input한다면 time 고려하는 방법 생각해보기
    outputevents['date_hour'] = outputevents['charttime'].dt.floor('H')

    outputevents = pd.merge(outputevents, d_items[['itemid', 'unique_label']], on=['itemid'], how='left')
    outputevents.dropna(subset=['itemid', 'unique_label'], inplace=True)
    outputevents.drop(columns=['itemid', 'charttime'], inplace=True)
    
    outputevents = pd.DataFrame(outputevents.groupby(['subject_id', 'hadm_id', 'date_hour', 'unique_label'])['value'].mean()).reset_index()
    outputevents.columns = ['subject_id', 'hadm_id', 'date_hour', 'unique_label', 'value']
    outputevents.to_csv(f"{save_path}/outputevents.csv", index=False)
    print(f'{datetime.now()-start} (hh:mm:ss.ms)')
    del outputevents
    
    return column_dictionary
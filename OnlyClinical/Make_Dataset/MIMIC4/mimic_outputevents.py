import pandas as pd
import gzip
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거


def outputevents(icu_path, save_path, icu_stay, d_items):
    print(f"Preprocess outputevents.csv", end=" ")
    start=datetime.now()
    # inputevents = pd.read_csv(f'{save_path}/inputevents_pivot.csv')
    
    d_items=d_items.loc[d_items["linksto"]=='outputevents', :]
    outputevents = pd.read_csv(gzip.open(f'{icu_path}/outputevents.csv.gz'), low_memory=False)
    outputevents = pd.merge(outputevents[['subject_id', 'hadm_id', 'charttime', 'itemid', 'value']], 
                            icu_stay[['subject_id', 'hadm_id', 'admittime']], on=['subject_id', 'hadm_id'], how='left')

    ## datetime -> date info only
    cols = ['admittime', 'charttime']
    outputevents[cols] = outputevents[cols].apply(lambda col: pd.to_datetime(col))
    outputevents=outputevents.sort_values(by=['subject_id', 'hadm_id', 'charttime'])   ################################################################ 나중에 한줄씩 input한다면 time 고려하는 방법 생각해보기
     
    # 입원 이후의 데이터
    outputevents = outputevents[(outputevents['charttime'] > outputevents['admittime'])]
    outputevents = pd.merge(outputevents[['subject_id', 'hadm_id', 'itemid', 'value']], d_items[['itemid', 'label', 'unique_label']], on=['itemid'], how='left')
    outputevents.dropna(subset=['itemid', 'label', 'unique_label'], inplace=True)
    output_level=outputevents["label"].unique()

    # 입원 이후의 value의 평균을 사용
    outputevents = pd.DataFrame(outputevents.groupby(['subject_id', 'hadm_id', 'unique_label'])['value'].agg(['median', 'mean'])).reset_index()
    outputevents['result'] = outputevents.apply(lambda row: row['median'] if row['unique_label'] not in output_level else row['mean'], axis=1)
    outputevents=outputevents[outputevents['result'].notnull()]
    
    outputevents_pivot = outputevents.pivot(index=['subject_id','hadm_id'], columns='unique_label', values='result').reset_index()   
    outputevents_pivot.to_csv(f"{save_path}/outputevents.csv", index=False)
    print(f'{datetime.now()-start} (hh:mm:ss.ms)')
    del outputevents, outputevents_pivot
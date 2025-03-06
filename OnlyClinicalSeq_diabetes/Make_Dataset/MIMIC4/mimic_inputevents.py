import numpy as np
import pandas as pd
import gzip
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거


def inputevents(icu_path, save_path, d_items, column_dictionary):
    print(f"Preprocess inputevents.csv", end=" ")
    start=datetime.now()
    
    subject = pd.read_csv(f'{save_path}/mimic4_subject.csv')
    d_items=d_items.loc[d_items["linksto"]=='inputevents', :]
    inputevents = pd.read_csv(gzip.open(f'{icu_path}/inputevents.csv.gz'), low_memory=False)
    inputevents = inputevents[['subject_id', 'hadm_id', 'starttime', 'endtime', 'itemid', 'amount', 'rate', 'rateuom', 'patientweight']]
    inputevents = inputevents[(inputevents['subject_id'].isin(subject['subject_id'])) & 
                              (inputevents['hadm_id'].isin(subject['hadm_id']))]
    cols = ['endtime', 'starttime']
    inputevents[cols] = inputevents[cols].apply(lambda col: pd.to_datetime(col))

    inputevents['dur_input'] = (inputevents['endtime'] - inputevents['starttime']).dt.total_seconds()/3600 # 시간으로 수정
    inputevents['rate'] = np.where('/min' in inputevents['rateuom'], inputevents['rate'] / 60, inputevents['rate']) # min을 hour로 바꾸기
    inputevents.loc[inputevents['rate'].isnull(), 'rate']  = inputevents['amount'] / inputevents['dur_input'] # rate가 nan인 부분 채우기
        
    inputevents = pd.merge(inputevents, d_items[['itemid', 'unique_label']], on=['itemid'], how='left')
    inputevents.dropna(subset=['itemid', 'unique_label'], inplace=True)    

    inputevents['end_date_hour'] = inputevents['endtime'].dt.floor('H')
    inputevents['start_date_hour'] = inputevents['starttime'].dt.floor('H')
    inputevents.drop(columns=['rateuom', 'itemid', 'endtime', 'starttime'], inplace=True)
    
    # inputevents = pd.DataFrame(inputevents.groupby(['subject_id', 'hadm_id', 'storetime', 'unique_label'])['valuenum'].mean()).reset_index()
    # index_cols = ['subject_id', 'hadm_id', 'storetime']
    # inputevents_pivot = inputevents.pivot_table(index=index_cols, columns='unique_label', values='valuenum', aggfunc='mean')
    # column_dictionary.update({label: "numerical" for label in inputevents_pivot.columns if label not in index_cols})
    # column_dictionary.update({row: "time" for row in ['starttime', 'endtime']})

    inputevents.to_csv(f"{save_path}/inputevents.csv", index=False)
    print(f'{datetime.now()-start} (hh:mm:ss.ms)')
    del inputevents
    return column_dictionary
    

def apply_condition_and_pivot(df, index_cols, column, values, suffix, input_level):
    # 조건에 따라 열 선택
    df[values] = df.apply(
        lambda row: row[f"{values}_median"] if row['unique_label'] not in input_level else row[f"{values}_mean"], 
        axis=1
    )
    # 필요한 열만 pivot하고 접미사 추가
    pivot_df = df.pivot(index=index_cols, columns=column, values=values).reset_index()
    pivot_df.columns = [f"{col}_{suffix}" if col not in index_cols else col for col in pivot_df.columns]
    return pivot_df

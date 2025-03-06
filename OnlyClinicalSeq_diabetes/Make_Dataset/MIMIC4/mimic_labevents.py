import pandas as pd
import gzip
import os
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

from Make_Dataset.MIMIC4.mimic_utils import fill_nan_hadm_id #, fill_na_within_group

    
def labevents(hosp_path, save_path, admission, column_dictionary=None):
    print(f"Preprocess labevents.csv")
    start=datetime.now()
    print(start)
    
    subject = pd.read_csv(f'{save_path}/mimic4_subject.csv')
    d_labitems = pd.read_csv(gzip.open(f'{hosp_path}/d_labitems.csv.gz'))
    ## "/".join() column fluid, category, label 
    d_labitems['combined'] = d_labitems[['category', 'fluid', 'label']].apply(lambda x: '//'.join(x.dropna().astype(str)), axis=1)
    # chart_lab=pd.read_csv(f"{save_path}/chartevents_lab.csv")
    hemoglobin = d_labitems[d_labitems['combined'].str.contains("% Hemoglobin")]
    
    labevents_csv = gzip.open(f'{hosp_path}/labevents.csv.gz')
    labevents_result = pd.DataFrame()
    cols = ['subject_id', 'hadm_id', 'itemid', 'storetime', 'valuenum', 'ref_range_lower', 'ref_range_upper', 'flag', 'priority']
    
    # chartevent 용량이 커서 부분부분 불러서 합치기
    for cnt, df in enumerate(pd.read_csv(labevents_csv, chunksize=1e7, usecols=cols, low_memory=False)):
        print(f"{cnt+1} chunk is added in labevent.csv")
        
        if df["hadm_id"].isnull().sum()!=0:
            df = fill_nan_hadm_id(admission, df, "admittime", "dischtime", "storetime")
        df.dropna(subset=['hadm_id', 'valuenum'], inplace=True)

        df = df[(df['subject_id'].isin(subject['subject_id'])) & 
                                (df['hadm_id'].isin(subject['hadm_id']))]
    
        cols = ['storetime']
        df[cols] = df[cols].apply(lambda col: pd.to_datetime(col))
        df['date_hour'] = df['storetime'].dt.floor('H')
    
        df = pd.merge(df, hemoglobin[['itemid', 'label']], on=['itemid'], how='left')
        # df = pd.merge(df, d_labitems[['itemid', 'label']], on=['itemid'], how='left')
        df.dropna(subset=['itemid', 'label'], inplace=True)
        df.drop(columns='itemid', inplace=True)
        labevents_result = pd.concat([labevents_result, df])
    
    labevents=labevents_result.drop_duplicates(subset=['subject_id', 'hadm_id', 'date_hour', 'label', 'valuenum'])
    print(datetime.now())

    # 같은 label은 같은 range를 가지고 있지 않을까
    # bfill = 결측값이 바로 아래값과 동일하게 설정, ffill = 결측값이 바로 위값과 동일하게 설정
    labevents['ref_range_lower'] = (labevents.groupby('label')['ref_range_lower']
                                    .transform(lambda x: x.fillna(method='ffill').fillna(method='bfill')))
    labevents['ref_range_upper'] = (labevents.groupby('label')['ref_range_upper']
                                    .transform(lambda x: x.fillna(method='ffill').fillna(method='bfill')))
    
    ## ref_range_lower, ref_range_upper 값이 없는 row flag를 unknown으로 추가하기- 한쪽만 null인 곳은 없음
    # range는 없는데, flag는 abnormal로 적혀있는 row가 있음 (ex, Microcytes, Macrocytes)
    unknown_mask = (labevents['ref_range_lower'].isnull() & labevents['ref_range_upper'].isnull() & labevents['flag'].isnull())
    normal_mask = (labevents["flag"].isnull() & (labevents["ref_range_lower"] <= labevents["valuenum"]) & (labevents["valuenum"] <= labevents["ref_range_upper"]))
    abnormal_mask = (labevents["flag"].isnull() & ((labevents["valuenum"] < labevents["ref_range_lower"]) | (labevents["ref_range_upper"] < labevents["valuenum"])))
    labevents.loc[unknown_mask, 'flag'] = "unknown"
    labevents.loc[normal_mask, 'flag'] = "normal"
    labevents.loc[abnormal_mask, 'flag'] = "abnormal"
    labevents['flag'] = labevents['flag'].map({'normal': 0, 'abnormal': 1, 'unknown': 2})
    ## priority
    labevents['priority'] = labevents['priority'].fillna("UNK").map({'ROUTINE': 0, 'STAT': 1, 'UNK': 2}) 
    print(datetime.now())
    
    labevents = (labevents.groupby(['subject_id', 'hadm_id', 'date_hour', 'label'])
                .agg({'valuenum': 'mean'}).reset_index()) #, 'flag': 'median' , 'priority': 'median'
    labevents = labevents.drop_duplicates(subset=['subject_id', 'hadm_id', 'date_hour', 'label']) # 중복 제거
    
    labevents.columns = ['subject_id', 'hadm_id', 'date_hour', 'unique_label', 'value']
    labevents.to_csv(f"{save_path}/labevents_value.csv", index=False)
    
    print(f'{datetime.now()-start} (hh:mm:ss.ms)')
    # del labevents, labevents_pivot
    
    return column_dictionary



def pivot_and_rename(labevents, index_cols, value_col, suffix):
    pivot_df = labevents.pivot(index=index_cols, columns='label', values=value_col).reset_index()
    pivot_df.columns = [col if col in index_cols else f"{col}_{suffix}" for col in pivot_df.columns]
    return pivot_df

def process_batch(args):
        indices, pivot_df_value, pivot_df_flag, pivot_df_priority = args
        i, j = indices
        batch_value = pivot_df_value.iloc[i:j]
        batch_flag = pivot_df_flag.iloc[i:j]
        batch_priority = pivot_df_priority.iloc[i:j]
        return pd.concat([batch_value, batch_flag, batch_priority], axis=1)
import pandas as pd
import gzip
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

from Make_Dataset.MIMIC4.mimic_utils import fill_nan_hadm_id, fill_na_within_group


def labevents(hosp_path, save_path, icu_stay, admission, column_dictionary):
    print(f"Preprocess labevents.csv", end=" ")
    start=datetime.now()
    print(start)
    
    d_labitems = pd.read_csv(gzip.open(f'{hosp_path}/d_labitems.csv.gz'))
    ## "/".join() column fluid, category, label 
    d_labitems['combined'] = d_labitems[['category', 'fluid', 'label']].apply(lambda x: '//'.join(x.dropna().astype(str)), axis=1)
    chart_lab=pd.read_csv(f"{save_path}/chartevents_lab.csv")
    
    labevents_csv = gzip.open(f'{hosp_path}/labevents.csv.gz')
    labevents_result = pd.DataFrame()
    cols = ['subject_id', 'hadm_id', 'itemid', 'storetime', 'valuenum', 'ref_range_lower', 'ref_range_upper', 'flag']
    
    # chartevent 용량이 커서 부분부분 불러서 합치기
    for cnt, df in enumerate(pd.read_csv(labevents_csv, chunksize=1e7, usecols=cols, low_memory=False)):
        print(f"{cnt+1} chunk is added in labevent.csv")
        
        if df["hadm_id"].isnull().sum()!=0:
            df = fill_nan_hadm_id(admission, df, "admittime", "dischtime", "storetime")
        df.dropna(subset=['hadm_id', 'valuenum'], inplace=True)
            
        df = pd.merge(df, icu_stay[['subject_id', 'hadm_id', 'admittime']], on=['subject_id', 'hadm_id'], how='left')
        
        cols = ['admittime', 'storetime']
        df[cols] = df[cols].apply(lambda col: pd.to_datetime(col))
        df = df[df['storetime'] > df['admittime']]
        
        df = pd.merge(df, d_labitems[['itemid', 'label']], on=['itemid'], how='left')
        df.dropna(subset=['itemid', 'label'], inplace=True)
    
        labevents_result = pd.concat([labevents_result, df[['subject_id', 'hadm_id', 'label', 'valuenum', 'ref_range_lower', 'ref_range_upper', 'flag']]])

    # chartevents lab과 labevent 값들과 결합 
    chart_lab = chart_lab[['subject_id', 'hadm_id', 'unique_label', 'valuenum']]
    chart_lab.columns=['subject_id', 'hadm_id', 'label', 'valuenum']
    labevents=pd.concat([chart_lab, labevents_result]) 
    labevents=labevents.drop_duplicates(subset=['subject_id', 'hadm_id', 'label', 'valuenum'])
    print(datetime.now())

    # # 공통되는 label 찾기
    # # list(set(chart_lab['label']).intersection(set(labevents_result['label'])))
    # chart_labels = set(chart_lab['label'])
    # lab_labels = set(labevents_result['label'])

    # # 공통되지 않는 label 값 찾기
    # list(chart_labels - lab_labels)
    
    # 같은 'label' -> range
    labevents = labevents.groupby(['label']).apply(fill_na_within_group) # labevents.groupby(['subject_id', 'hadm_id', 'label']).apply(fill_na_within_group)
    print(datetime.now())
    
    ## ref_range_lower, ref_range_upper 값이 없는 row flag를 unknown으로 추가하기- 한쪽만 null인 곳은 없음
    labevents.loc[labevents['ref_range_lower'].isnull() & labevents['ref_range_upper'].isnull() & labevents['flag'].isnull(), "flag"] ="unknown"
    # range는 없는데, flag는 abnormal로 적혀있는 row가 있음 (ex, Microcytes, Macrocytes)
    ## ref_range_lower, ref_range_upper 값이 있는 row
    labevents.loc[(labevents["flag"].isnull()) & (labevents["ref_range_lower"]<=labevents["valuenum"]) & (labevents["valuenum"]<=labevents["ref_range_upper"]), "flag"]="normal"
    labevents.loc[(labevents["flag"].isnull()) & ((labevents["valuenum"]<labevents["ref_range_lower"]) | (labevents["ref_range_upper"]<labevents["valuenum"])), "flag"]="abnormal"
    labevents['flag'].replace({'normal':0, 'abnormal':1, 'unknown':2}, inplace=True)
    labevents=labevents[labevents['flag'].notnull()] 
    
    labevents=pd.DataFrame(labevents.groupby(['subject_id', 'hadm_id', 'label'])[['valuenum', 'flag']].median()).reset_index()
    index_cols = ['subject_id', 'hadm_id']
    pivot_df_value=labevents.pivot(index=index_cols, columns='label', values='valuenum')
    pivot_df_value.columns=[f'{col}_value' for col in pivot_df_value.columns]
    pivot_df_flag=labevents.pivot(index=index_cols, columns='label', values='flag')
    pivot_df_flag.columns=[f'{col}_flag' for col in pivot_df_flag.columns]
    
    labevents_pivot = pivot_df_value.merge(pivot_df_flag, on=index_cols)
    labevents_pivot=labevents_pivot.reset_index()
    
    column_dictionary.update({row: "numerical" for row in pivot_df_value.columns})
    column_dictionary.update({row: "categorical" for row in pivot_df_flag.columns})
    
    labevents_pivot.to_csv(f"{save_path}/labevents.csv", index=False)
    print(f'{datetime.now()-start} (hh:mm:ss.ms)')
    del labevents, labevents_pivot
    
    return column_dictionary


    # idx=(labevents['flag']==1) & (labevents["valuenum"]>labevents["ref_range_upper"])
    # labevents.loc[idx, "importance"] = (labevents.loc[idx, "valuenum"]-labevents.loc[idx, "ref_range_upper"])
    # idx=(labevents['flag']==1) & (labevents["valuenum"]<labevents["ref_range_lower"])
    # labevents.loc[idx, "importance"] = (labevents.loc[idx, "valuenum"]-labevents.loc[idx,"ref_range_lower"])
    # labevents.loc[labevents['flag']==0, "importance"] = 0
    
import pandas as pd
import gzip
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거


def inputevents(icu_path, save_path, icu_stay, d_items, column_dictionary):
    print(f"Preprocess inputevents.csv", end=" ")
    start=datetime.now()
    
    d_items=d_items.loc[d_items["linksto"]=='inputevents', :]
    inputevents = pd.read_csv(gzip.open(f'{icu_path}/inputevents.csv.gz'), low_memory=False)
    inputevents = pd.merge(inputevents[['subject_id', 'hadm_id', 'starttime', 'endtime', 'itemid', 'amount']], 
                            icu_stay[['subject_id', 'hadm_id', 'admittime']], on=['subject_id', 'hadm_id'], how='left') # icu에 입장하고 난 뒤가 아니라 병원에 내원하고 난 뒤부터
    
    ## datetime -> date info only
    cols = ['admittime', 'endtime', 'starttime']
    inputevents[cols] = inputevents[cols].apply(lambda col: pd.to_datetime(col))

    inputevents = inputevents[(inputevents['endtime'] > inputevents['starttime']) & (inputevents['starttime'] >= inputevents['admittime'])]
    inputevents['dur_before_input'] = (inputevents['starttime'] - inputevents['admittime']).dt.total_seconds()/3600 ##### 시로 맞추자
    inputevents['dur_input'] = (inputevents['endtime'] - inputevents['starttime']).dt.total_seconds()/3600
    inputevents['amountMinRate']=inputevents['amount']/(inputevents['dur_input']/3600) # 1시간당
    inputevents=inputevents.sort_values(by=['subject_id', 'hadm_id', 'starttime'])   ################################################################ 나중에 한줄씩 input한다면 time 고려하는 방법 생각해보기
        
    inputevents = pd.merge(inputevents[['subject_id', 'hadm_id', 'itemid', 'amount', 'dur_input', 'amountMinRate', 'dur_before_input']], d_items[['itemid', 'label', 'unique_label']], on=['itemid'], how='left')
    inputevents.dropna(subset=['itemid', 'label', 'unique_label'], inplace=True) 
    input_level=inputevents["label"].unique()

    # value의 중앙값을 사용
    inputevents = inputevents.groupby(['subject_id', 'hadm_id', 'unique_label']).agg(amount_median=('amount', 'median'),
                                                                                    amount_mean=('amount', 'mean'),
                                                                                    dur_before_input_mean=('dur_before_input', 'mean'),
                                                                                    dur_before_input_median=('dur_before_input', 'median'),
                                                                                    dur_input_median=('dur_input', 'median'),
                                                                                    dur_input_mean=('dur_input', 'mean'),
                                                                                    amountMinRate_median=('amountMinRate', 'median'),
                                                                                    amountMinRate_mean=('amountMinRate', 'mean')
                                                                                ).reset_index()
    inputevents = inputevents[inputevents['amount_median'].notnull() & inputevents['amount_mean'].notnull() &
                              inputevents['dur_before_input_mean'].notnull() & inputevents['dur_before_input_median'].notnull() & 
                              inputevents['dur_input_median'].notnull() & inputevents['dur_input_mean'].notnull() & 
                              inputevents['amountMinRate_median'].notnull() & inputevents['amountMinRate_mean'].notnull()
                            ]
    
    index_cols = ['subject_id', 'hadm_id']
    inputevents_pivot_amount = apply_condition_and_pivot(inputevents, index_cols, 'unique_label', 'amount', 'amount', input_level)
    inputevents_pivot_inputdur = apply_condition_and_pivot(inputevents, index_cols, 'unique_label', 'dur_input', 'durinput', input_level)
    inputevents_pivot_rate = apply_condition_and_pivot(inputevents, index_cols, 'unique_label', 'amountMinRate', 'amountrate', input_level)
    inputevents_pivot_beforedur = apply_condition_and_pivot(inputevents, index_cols, 'unique_label', 'dur_before_input', 'durbeforeinput', input_level)
    
    column_dictionary.update({label: "numerical" for label in inputevents_pivot_amount.columns if label not in index_cols})
    column_dictionary.update({label: "numerical" for label in inputevents_pivot_inputdur.columns if label not in index_cols})
    column_dictionary.update({label: "numerical" for label in inputevents_pivot_rate.columns if label not in index_cols})
    column_dictionary.update({label: "numerical" for label in inputevents_pivot_beforedur.columns if label not in index_cols})
    
    # 각 pivot 테이블 병합
    inputevents_pivot = (
        inputevents_pivot_amount
        .merge(inputevents_pivot_inputdur, on=index_cols)
        .merge(inputevents_pivot_rate, on=index_cols)
        .merge(inputevents_pivot_beforedur, on=index_cols)
    )
    inputevents_pivot.to_csv(f"{save_path}/inputevents.csv", index=False)
    print(f'{datetime.now()-start} (hh:mm:ss.ms)')
    del inputevents, inputevents_pivot
    

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
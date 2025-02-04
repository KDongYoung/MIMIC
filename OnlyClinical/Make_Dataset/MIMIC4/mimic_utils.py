import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

def fill_nan_hadm_id(df, events, intime, outtime, storedate):
    
    df[intime] = pd.to_datetime(df[intime])
    df[outtime] = pd.to_datetime(df[outtime])
    events[storedate] = pd.to_datetime(events[storedate])
    merged = pd.merge(df[['subject_id', 'hadm_id', intime, outtime]], 
                      events[['subject_id', 'hadm_id', storedate]], 
                      on='subject_id', suffixes=('', '_event'))
    filtered = merged[(merged[intime] <= merged[storedate]) & (merged[storedate] <= merged[outtime])] # admittime, dischtime 사이에 처방된 것만 선택
    del merged
    
    filtered = filtered.drop_duplicates(subset=['subject_id', storedate, 'hadm_id'])
    filtered[['subject_id', storedate, 'hadm_id']].drop_duplicates(subset=['subject_id', storedate])

    filtered = filtered[['subject_id', storedate, 'hadm_id']]

    events = pd.merge(events, filtered, on=['subject_id', storedate], how='left', suffixes=('', '_x'))
    events['hadm_id'].fillna(events['hadm_id_x'], inplace=True)
    events.drop(columns=['hadm_id_x'], inplace=True)
    
    return events

def fill_na_within_group(group):
    # 같은 label은 같은 range를 가지고 있지 않을까
    ## bfill = 결측값이 바로 아래값과 동일하게 설정, ffill = 결측값이 바로 위값과 동일하게 설정
    if group['ref_range_lower'].isnull().sum():
        group['ref_range_lower'] = group['ref_range_lower'].fillna(method='ffill').fillna(method='bfill')
    if group['ref_range_upper'].isnull().sum():
        group['ref_range_upper'] = group['ref_range_upper'].fillna(method='ffill').fillna(method='bfill')
    
    return group
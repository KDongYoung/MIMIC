import pandas as pd
import gzip
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def chartevents(icu_path, save_path, d_items, column_dictionary):
    print(f"Preprocess chartevents.csv")
    start=datetime.now()

    subject = pd.read_csv(f'{save_path}/mimic4_subject.csv')
    d_items=d_items.loc[(d_items["linksto"]=='chartevents'), :] # apache 값들 제거
    
    chartevent_csv = gzip.open(f'{icu_path}/chartevents.csv.gz')
    chartevent_result = pd.DataFrame()
    cols = ['subject_id', 'hadm_id', 'storetime', 'itemid', 'valuenum']
    
    # chartevent 용량이 커서 부분부분 불러서 합치기
    for cnt, df in enumerate(pd.read_csv(chartevent_csv, chunksize=1e7, usecols=cols, low_memory=False)):
        print(f"{cnt+1} chunk is added in chartevent.csv")
        df.dropna(subset=['valuenum'], inplace=True)
        df = df[(df['subject_id'].isin(subject['subject_id'])) & 
                (df['hadm_id'].isin(subject['hadm_id']))]
        
        ## datetime -> date info only
        cols = ['storetime']
        df[cols] = df[cols].apply(lambda col: pd.to_datetime(col))

        df = pd.merge(df, d_items[['itemid', 'unique_label', 'category']], on=['itemid'], how='left')            
        df.dropna(subset=['itemid', "unique_label"], inplace=True)
        chartevent_result = pd.concat([chartevent_result, df])
    
    chartevent_result['date_hour'] = chartevent_result['storetime'].dt.floor('H')
    chartevent_result.drop(columns=['storetime', 'itemid'], inplace=True)
    
    # chartevent_result_labs = chartevent_result[chartevent_result["category"]=="Labs"] # labevents   
    # chartevent_result_labs.to_csv(f"{save_path}/chartevents_lab.csv", index=False)    
    # chartevent_result = chartevent_result[chartevent_result["category"]!="Labs"]
    
    # chartevent_result.drop(columns=['category'], inplace=True)
    chartevent_result = pd.DataFrame(chartevent_result.groupby(['subject_id', 'hadm_id', 'date_hour', 'unique_label'])['valuenum'].mean()).reset_index()
    index_cols = ['subject_id', 'hadm_id', 'date_hour']
    chartevent_pivot=chartevent_result.pivot(index=index_cols, columns='unique_label', values='valuenum').reset_index()
    
    # 온도 정보 수정
    import time
    a=time.time()
    chartevent_pivot.loc[chartevent_pivot['Temperature Celsius'].isnull(), 'Temperature Celsius']=(chartevent_pivot['Temperature Fahrenheit']-32)*5/9
    # chartevent_pivot.loc[chartevent_pivot['Temperature Fahrenheit'].isnull(), 'Temperature Fahrenheit']=(chartevent_pivot.loc[chartevent_pivot['Temperature Fahrenheit'].isnull(), 'Temperature Celsius']*9/5)-32
    chartevent_pivot.drop(columns='Temperature Fahrenheit', inplace=True)
    print(f"{time.time()-a:.8f}s")    

    column_types = {
        **{c: "numerical" for c in chartevent_pivot.columns if c not in ['subject_id', 'hadm_id', 'date_hour']},
        **{c: "id" for c in ['subject_id', 'hadm_id']},
        'date_hour': "time"
    }
    
    column_dictionary.update(column_types) 
    
    height_weight_df=chartevent_pivot[['subject_id', 'hadm_id', 'date_hour', 'Height', 'Height (cm)', 'Admission Weight (lbs.)', 'Admission Weight (Kg)']]    
    height_weight_df.to_csv(f"{save_path}/chart_hw.csv", index=False)
    chartevent_pivot.drop(columns=['Height', 'Height (cm)', 'Admission Weight (lbs.)', 'Admission Weight (Kg)'], inplace=True)
    
    # chartevent_pivot = chartevent_pivot.loc[:, chartevent_pivot.isnull().mean()<0.9] 
    # chartevent_pivot = chartevent_pivot.fillna(0)
    chartevent_pivot.to_csv(f"{save_path}/chartevents.csv", index=False)
        
    print(f'Duration: {datetime.now()-start} (hh:mm:ss.ms)')
    del chartevent_result, chartevent_pivot
    
    return column_dictionary


# 'height' 열의 데이터를 숫자로 변환하는 함수 정의
def convert_to_numeric(val):
    try:
        return float(val)
    except ValueError:
        return float('nan')

def chart_height_weight_bp(hosp_path, save_path, args):
    print("Make height_weight dataset...")
    omr=pd.read_csv(gzip.open(f"{hosp_path}/omr.csv.gz"), low_memory = False)
    omr.rename(columns={'chartdate': 'date_hour'}, inplace=True)
    
    # Height = Height (Inches), Weight = Weight (Lbs)
    height_df = omr.loc[(omr['result_name'] == 'Height') | (omr['result_name'] == 'Height (Inches)'), ['subject_id', 'date_hour', 'result_name', 'result_value']]
    height_df['result_value'] = height_df['result_value'].apply(convert_to_numeric)
    height_df = height_df.groupby(['subject_id', 'date_hour', 'result_name'])['result_value'].mean().reset_index()
    height_df.drop_duplicates(inplace=True)
    
    weight_df = omr.loc[(omr['result_name'] == 'Weight') | (omr['result_name'] == 'Weight (Lbs)'), ['subject_id', 'date_hour', 'result_name', 'result_value']]
    weight_df['result_value'] = weight_df['result_value'].apply(convert_to_numeric)
    weight_df = weight_df.groupby(['subject_id', 'date_hour', 'result_name'])['result_value'].mean().reset_index()
    weight_df.drop_duplicates(inplace=True)

    df=pd.concat([height_df, weight_df])
    df['date_hour'] = pd.to_datetime(df['date_hour'])
    df = df.pivot(index=['subject_id', 'date_hour'], columns='result_name', values='result_value').reset_index()
    
    df['Weight'].fillna(df['Weight (Lbs)']/2.2, inplace=True)
    df['Weight (Lbs)'].fillna(df['Weight']*2.2, inplace=True)
    df['Height (Inches)'].fillna(df['Height']/2.53, inplace=True)
    df['Height'].fillna(df['Height (Inches)']*2.53, inplace=True)
    df[['Height', 'Height (Inches)']] = df[['Height', 'Height (Inches)']].round(1)
    df[['Weight', 'Weight (Lbs)']] = df[['Weight', 'Weight (Lbs)']].round(1) 
    df.drop(columns = ['Height (Inches)', 'Weight (Lbs)'], inplace=True)
    
    ##############################################################################
    hw_df = pd.read_csv(f"{save_path}/chart_hw.csv")
    hw_df['date_hour'] = pd.to_datetime(hw_df['date_hour'])
    hw_df['year_date'] = hw_df['date_hour'].dt.to_period('D')
    
    ## Height
    hw_df['Height'].fillna(hw_df['Height (cm)']/2.53, inplace=True)
    hw_df['Height (cm)'].fillna(hw_df['Height']*2.53, inplace=True)
    hw_df[['Height', 'Height (cm)']] = hw_df[['Height', 'Height (cm)']].round(1)
    ## Weight
    hw_df['Admission Weight (Kg)'].fillna(hw_df['Admission Weight (lbs.)']/2.2, inplace=True)
    hw_df['Admission Weight (lbs.)'].fillna(hw_df['Admission Weight (Kg)']*2.2, inplace=True)
    hw_df[['Admission Weight (Kg)', 'Admission Weight (lbs.)']] = hw_df[['Admission Weight (Kg)', 'Admission Weight (lbs.)']].round(1) 
    hw_df.drop(columns = ['date_hour', 'Height', 'Admission Weight (lbs.)'], inplace=True)
    hw_df.rename(columns={'year_date': 'date_hour', 'Height (cm)': 'Height', 'Admission Weight (Kg)': 'Weight'}, inplace=True)
    hw_df.drop_duplicates(inplace=True)
    
    hw_df = pd.merge(hw_df, df, on=['subject_id', 'date_hour', 'Height', 'Weight'], how='outer')
    hw_df = hw_df[~(hw_df['Height'].isna() & hw_df['Weight'].isna()) & ((hw_df['Height'] >= 50) | (hw_df['Weight'] >= 40))]
    hw_df.drop_duplicates(inplace=True)
    
    grouped = hw_df.groupby('subject_id')
    for sbj, group in grouped:
        if group[['Height', 'Weight']].notnull().all(axis=1).any() and group[['Height', 'Weight']].isnull().any(axis=1).any():
            col_means = group[['Height', 'Weight']].dropna().mean()
            hw_df.loc[hw_df['subject_id'] == sbj, ['Height', 'Weight']] = group[['Height', 'Weight']].fillna(col_means)
    hw_df = hw_df.dropna(subset=hw_df.columns)
    hw_df.to_csv(f"{save_path}/hws.csv", index=False)
    print("Finished height_weight dataset")
    
    print("Make height_weight dataset...")
    bp = pd.DataFrame(omr[omr['result_name'] == 'Blood Pressure'], columns=['subject_id','date_hour','result_value'])
    bp[['BPs', 'BPd']] = bp['result_value'].str.split('/', expand=True)
    
    bp['BPs'] = pd.to_numeric(bp['BPs'], errors='coerce')
    bp['BPd'] = pd.to_numeric(bp['BPd'], errors='coerce')
    bp = bp[['subject_id', 'date_hour', 'BPs', 'BPd']]
    bp.to_csv(f"{save_path}/bps.csv", index=False)
    print("Finished bp dataset")
    
    return args
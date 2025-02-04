import pandas as pd
import gzip
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

def chartevents(icu_path, save_path, icu_stay, d_items, column_dictionary):
    print(f"Preprocess chartevents.csv")
    start=datetime.now()

    d_items=d_items.loc[(d_items["linksto"]=='chartevents'), :] # apache 값들 제거
    
    chartevent_csv = gzip.open(f'{icu_path}/chartevents.csv.gz')
    chartevent_result = pd.DataFrame()
    cols = ['subject_id', 'hadm_id', 'storetime', 'itemid', 'valuenum']
    
    # chartevent 용량이 커서 부분부분 불러서 합치기
    for cnt, df in enumerate(pd.read_csv(chartevent_csv, chunksize=1e7, usecols=cols, low_memory=False)):
        print(f"{cnt+1} chunk is added in chartevent.csv")
        df.dropna(subset=['valuenum'], inplace=True)
        df = pd.merge(df, icu_stay[['subject_id', 'hadm_id', 'admittime']], on=['subject_id','hadm_id'], how='left')
        
        ## datetime -> date info only
        cols = ['admittime', 'storetime']
        df[cols] = df[cols].apply(lambda col: pd.to_datetime(col))

        df = df[df['storetime'] > df['admittime']]
        df = pd.merge(df, d_items[['itemid', 'label', 'unique_label', 'category']], on=['itemid'], how='left')            
        df.dropna(subset=['itemid', "label", "unique_label"], inplace=True)
        chartevent_result = pd.concat([chartevent_result, df])
        
    # chartevent_result = chartevent_result.dropna(subset=["label", "unique_label"]) 
    chart_level= chartevent_result["label"].unique()
    
    chartevent_result_labs = chartevent_result[chartevent_result["category"]=="Labs"] # labevents   
    chartevent_result_labs.to_csv(f"{save_path}/chartevents_lab.csv", index=False)    
    chartevent_result = chartevent_result[chartevent_result["category"]!="Labs"]
    
    chartevent_result = pd.DataFrame(chartevent_result.groupby(['subject_id', 'hadm_id', 'unique_label'])['valuenum'].agg(['median', 'mean'])).reset_index()
    chartevent_result['result'] = chartevent_result.apply(lambda row: row['median'] if row['unique_label'] not in chart_level else row['mean'], axis=1)
    chartevent_result=chartevent_result[chartevent_result['result'].notnull()]
    
    chartevent_pivot = chartevent_result.pivot(index=['subject_id', 'hadm_id'], columns='unique_label', values='result').reset_index()
    chartevent_pivot.loc[chartevent_pivot['Temperature Celsius'].isnull(), 'Temperature Celsius']=(chartevent_pivot.loc[chartevent_pivot['Temperature Celsius'].isnull(), 'Temperature Fahrenheit']-32)*5/9
    chartevent_pivot.loc[chartevent_pivot['Temperature Fahrenheit'].isnull(), 'Temperature Fahrenheit']=(chartevent_pivot.loc[chartevent_pivot['Temperature Fahrenheit'].isnull(), 'Temperature Celsius']*9/5)-32
    chartevent_pivot.drop(columns='Temperature Fahrenheit', inplace=True)
    
    ## 값이 다 nan인 것 제외
    nan_col=[]
    for col in chartevent_pivot.columns:
        if (chartevent_pivot[col].isnull().sum())==chartevent_pivot.shape[0]:
            nan_col.append(col)
    chartevent_pivot.drop(columns=nan_col, inplace=True)
    column_dictionary = {key: value for key, value in column_dictionary.items() if key not in nan_col} # remove from column dictionary
    
    height_weight_df=chartevent_pivot[['subject_id', 'hadm_id', 'Height', 'Height (cm)', 'Admission Weight (lbs.)', 'Admission Weight (Kg)']]
    chartevent_pivot.drop(columns=['Height', 'Height (cm)', 'Admission Weight (lbs.)', 'Admission Weight (Kg)'], inplace=True)
    
    pd.DataFrame(chartevent_pivot.columns).to_csv(f"{save_path}/chartevents_label.csv", index=False)
    chartevent_pivot.to_csv(f"{save_path}/chartevents.csv", index=False)
        
    print(f'{datetime.now()-start} (hh:mm:ss.ms)')
    del chartevent_result, chartevent_pivot
    
    return height_weight_df, column_dictionary


# 'height' 열의 데이터를 숫자로 변환하는 함수 정의
def convert_to_numeric(val):
    try:
        return float(val)
    except ValueError:
        return float('nan')

def chart_height_weight(hosp_path, save_path, hw_df, args):
    omr=pd.read_csv(gzip.open(f"{hosp_path}/omr.csv.gz"), low_memory = False)

    # Height = Height (Inches), Weight = Weight (Lbs)
    height_df = pd.DataFrame(omr[(omr['result_name'] == 'Height') | (omr['result_name'] == 'Height (Inches)')], columns=['subject_id','chartdate','result_value'])
    height_df.columns=['subject_id', 'chartdate', 'height']
    height_df['height'] = height_df['height'].apply(convert_to_numeric)
    height_df = height_df.groupby(['subject_id', 'chartdate'])['height'].mean().reset_index()
    height_df.drop_duplicates(inplace=True)
    weight_df = pd.DataFrame(omr[(omr['result_name'] == 'Weight') | (omr['result_name'] == 'Weight (Lbs)')], columns=['subject_id','chartdate','result_value'])
    weight_df.columns=['subject_id', 'chartdate', 'weight']
    weight_df['weight'] = weight_df['weight'].apply(convert_to_numeric)
    weight_df = weight_df.groupby(['subject_id', 'chartdate'])['weight'].mean().reset_index()
    weight_df.drop_duplicates(inplace=True)

    df=pd.merge(height_df, weight_df, on=['subject_id', 'chartdate'])
    
    df=pd.DataFrame(df.groupby('subject_id')[['height', 'weight']].mean()).reset_index()
    hw_df=pd.merge(hw_df, df, on='subject_id', how='left')
    
    ## Height
    mask=hw_df['Height'].isnull() & hw_df['Height (cm)'].isnull() & hw_df['height'].notnull()
    hw_df.loc[mask, 'Height']=hw_df.loc[mask, 'height']
    hw_df['Height'].fillna(hw_df['Height (cm)']/2.53, inplace=True)
    hw_df['Height (cm)'].fillna(hw_df['Height']*2.53, inplace=True)
    hw_df[['Height', 'Height (cm)']] = hw_df[['Height', 'Height (cm)']].round(1)

    ## Weight
    mask=hw_df['Admission Weight (lbs.)'].isnull() & hw_df['Admission Weight (Kg)'].isnull() & hw_df['weight'].notnull()
    hw_df.loc[mask, 'Admission Weight (lbs.)']=hw_df.loc[mask, 'weight']
    hw_df['Admission Weight (Kg)'].fillna(hw_df['Admission Weight (lbs.)']/2.2, inplace=True)
    hw_df['Admission Weight (lbs.)'].fillna(hw_df['Admission Weight (Kg)']*2.2, inplace=True)
    hw_df[['Admission Weight (Kg)', 'Admission Weight (lbs.)']] = hw_df[['Admission Weight (Kg)', 'Admission Weight (lbs.)']].round(1) 
    
    hw_df[['subject_id', 'hadm_id', 'Height (cm)', 'Admission Weight (Kg)']].to_csv(f"{save_path}/hw.csv", index=False)
    if len(args["missing_value"])<=2: args["missing_value"].extend(['Height (cm)', 'Admission Weight (Kg)'])
    
    return args
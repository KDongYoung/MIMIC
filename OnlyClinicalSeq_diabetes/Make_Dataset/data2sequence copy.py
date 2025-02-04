import numpy as np 
import pandas as pd 
import ast
from datetime import datetime
import os


def data2seq(args, dataset):
    
    # os.makedirs(f"{args['save_data']}/sequence/", exist_ok=True)
    """        
    ##########################
    ###### Load Dataset ######        
    ##########################
    """
    print(f"Start making sequence dataset, {datetime.now()}")
    transfer = pd.read_csv(f"{args['save_data']}/transfer.csv", low_memory = False)
    inputevents = pd.read_csv(f"{args['save_data']}/inputevents.csv", low_memory = False)
    
    print(f"Loaded all dataset, {datetime.now()}")
    
    transfer['intime'] = pd.to_datetime(transfer['intime'])
    transfer['outtime'] = pd.to_datetime(transfer['outtime'])
    
    with open(args['save_data'] + '/category_encoding.txt', 'r') as f:
        line=f.read().split("\n")
    category_dictionary = ast.literal_eval(line[0])
    category_dictionary["ED"] = max(category_dictionary.values())+1
    
    
    
    """   
    ###################################     
    ###### 시간 단위로 데이터 분할 ######
    ###################################
    """
    index_cols = ["subject_id", 'hadm_id', 'time']
    
    total = []
    print(f"Start making, {datetime.now()}")
    for i, row in dataset.iterrows():
        sbj_df = split_to_hourly_intervals(row)
        
        """        ###### Basic ######        """
        sbj_df[['location', 'calculated_age', 'gender', 'anchor_year_group', 'mortality']] = 0
        sbj_df.loc[sbj_df['time'] == row['admit_date_hour'], 'location'] = row['admission_location']
        sbj_df.loc[sbj_df['time'] == row['disch_date_hour'], 'location'] = row['discharge_location']
        sbj_df.loc[sbj_df['time'] == row['in_date_hour'], 'location'] = row['first_careunit']
        sbj_df.loc[sbj_df['time'] == row['out_date_hour'], 'location'] = row['last_careunit']
        
        sbj_df.loc[:, 'calculated_age'] = row['calculated_age']
        sbj_df.loc[:, 'gender'] = row['gender']
        sbj_df.loc[:, 'anchor_year_group'] = row['anchor_year_group']
        sbj_df.loc[:, 'mortality'] = row['mortality']
        
        if not pd.isna(row['dur_ed']):
            sbj_df.loc[(row['edreg_date_hour'] <= sbj_df['time']) & (sbj_df['time'] <= row['edout_date_hour']), 'location'] = category_dictionary["ED"]
            
        if not pd.isna(row['dur_icu']):
            if row['first_careunit'] != row['last_careunit']:              
                trans = transfer[(transfer['subject_id'] == row['subject_id']) & (transfer['hadm_id'] == row['hadm_id'])]
                
                for _, t_row in trans.iterrows():
                    if t_row['careunit'] not in category_dictionary.keys(): 
                        sbj_df.loc[(t_row['intime'] <= sbj_df['time']) & (sbj_df['time'] <= t_row['outtime']), 'location'] = category_dictionary["Unknown"]
                    else:
                        sbj_df.loc[(t_row['intime'] <= sbj_df['time']) & (sbj_df['time'] <= t_row['outtime']), 'location'] = category_dictionary[t_row['careunit']]
            else: 
                # if row['first_careunit'] == next((k for k, v in category_dictionary['first_careunit'].items() if v == "None"), None):
                sbj_df.loc[(row['in_date_hour'] <= sbj_df['time']) & (sbj_df['time'] <= row['out_date_hour']), 'location'] = row['first_careunit']
        
        """        ###### Inputevents ######        """
        unique_labels = inputevents['unique_label'].unique()
        new_columns = pd.DataFrame(0, index=sbj_df.index, columns=unique_labels)
        new_columns = pd.concat([sbj_df[['subject_id', 'hadm_id', 'time']], new_columns], axis=1)
        in_event = inputevents[(inputevents['subject_id'] == row['subject_id']) & (inputevents['hadm_id'] == row['hadm_id'])]

        for _, in_row in in_event.iterrows():
            if new_columns.loc[new_columns['time'] == in_row['start_date_hour'], in_row['unique_label']].any():
                condition = (new_columns['time'] > in_row['start_date_hour']) & (new_columns['time'] <= in_row['end_date_hour'])
            else: # == 0
                condition = (new_columns['time'] >= in_row['start_date_hour']) & (new_columns['time'] <= in_row['end_date_hour'])    
            
            new_columns.loc[condition, in_row['unique_label']] = np.where(in_row['amount'] < in_row['rate'], 
                                                                     in_row['amount'], in_row['rate'])

        sbj_df = pd.concat([sbj_df, new_columns.drop(columns=['subject_id', 'hadm_id', 'time'])], axis=1)
        
        if i%2000 == 0:
            print(f"{i} row making... {datetime.now()}")               
        total.append(sbj_df)
        # sbj_df.to_csv(f"{args['save_data']}/sequence/{row['subject_id']}_{row['hadm_id']}.csv", index=False)
   
    del sbj_df, new_columns, transfer, inputevents

    print(f"Finished, {datetime.now()}") # 12시간 걸림
    total = pd.concat(total)
    print(f"Finished making basic, inutevents dataset, {datetime.now()}") # 12시간 걸림
    total.to_csv(f"{args['save_data']}/sequence_total.csv", index=False)  
    del total
    
    """        ###### hw, chartevents, outputevents ######        """
    hws=pd.read_csv(f"{args['save_data']}/hw.csv", low_memory = False)
    chartevents=pd.read_csv(f"{args['save_data']}/chartevents.csv", low_memory = False)
    outputevents=pd.read_csv(f"{args['save_data']}/outputevents.csv", low_memory = False)
    
    if args['labevents']:
        ## labevents 결합
        labevents_value=pd.read_csv(f"{args['save_data']}/labevents_value.csv", low_memory = False)
        # labevents_flag=pd.read_csv(f"{args['save_data']}/labevents_flag.csv", low_memory = False)
        # labevents_priority=pd.read_csv(f"{args['save_data']}/labevents_priority.csv", low_memory = False)
    
    hws.rename(columns={'year_month': 'time'}, inplace=True)
    hws['time'] = pd.to_datetime(hws['time'])
    dataframes = [chartevents, outputevents, labevents_value] # , labevents_flag
    for df in dataframes:
        df.rename(columns={'date_hour': 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'])         
    del dataframes    
    
    
    
    # hw_df['date_hour'] = pd.to_datetime(hw_df['date_hour'])
    # hw_df['year_month'] = hw_df['date_hour'].dt.floor('H')
    
    # ## 같은 날/달에 잰 키와 몸무게는 같겠지
    # hw_df.update(hw_df.groupby(['subject_id', 'year_month'])[
    #                                 ['Height', 'Height (cm)', 'Admission Weight (lbs.)', 'Admission Weight (Kg)']
    #                             ].transform(lambda x: x.ffill().bfill())
    #                         )
    # hw_df.drop(columns=['date_hour'], inplace=True)

    
    # 온도 정보 수정
    # import time
    # a=time.time()
    # chartevent_pivot.loc[chartevent_pivot['Temperature Celsius'].isnull(), 'Temperature Celsius']=(chartevent_pivot['Temperature Fahrenheit']-32)*5/9
    # chartevent_pivot.drop(columns='Temperature Fahrenheit', inplace=True)
    # print(f"{time.time()-a:.8f}s")    

    # column_dictionary.update({'date_hour': "time"}) 
    # height_weight_df=chartevent_pivot[['subject_id', 'hadm_id', 'date_hour', 'Height', 'Height (cm)', 'Admission Weight (lbs.)', 'Admission Weight (Kg)']]    
    # chartevent_pivot.drop(columns=['Height', 'Height (cm)', 'Admission Weight (lbs.)', 'Admission Weight (Kg)'], inplace=True)
    
    
    df_merged = pd.merge(hws, outputevents, on=index_cols, how='outer')
    df_merged = pd.merge(df_merged, chartevents, on=index_cols, how='outer')
    print(f"Merged hw, output, chartevents, {datetime.now()}") 
    del hws, outputevents, chartevents 
    
    total = pd.read_csv(f"{args['save_data']}/sequence_total.csv")    
    total = pd.merge(total, df_merged, on=index_cols, how='left')
    total = total.fillna(0)   
    
    del sbj_df, in_event, transfer, trans, dataset 
    total.to_csv(f"{args['save_data']}/sequence_dataset.csv", index=False)
    print(f"Finished making sequence dataset, {datetime.now()}") # 12시간 걸림
    

def input_event_value(df, row, identify_info, index_cols):
    
    partial = df[(df['subject_id'] == row['subject_id']) & (df['hadm_id'] == row['hadm_id'])]
    unique_labels = [col for col in df.columns if col not in index_cols]
    new_columns = pd.DataFrame(0, index=identify_info.index, columns=unique_labels)
    new_columns = pd.concat([identify_info, new_columns], axis=1)
    
    for _, h_row in partial.iterrows():
        new_columns.loc[new_columns['time'] == h_row['time'], unique_labels] = h_row[unique_labels]
    return new_columns[unique_labels]
    
# 1시간 단위로 쪼개는 함수    
def split_to_hourly_intervals(row):
    # 시간 간격 생성
    time_range = pd.date_range(start=row["admit_date_hour"], end=row["disch_date_hour"], freq="1H", closed="left")
    return pd.DataFrame({
        "subject_id": row["subject_id"],
        "hadm_id": row["hadm_id"],
        "time": time_range
    })
import numpy as np 
import pandas as pd 
import ast
import os
from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


def data2seq(args, dataset):
    
    # os.makedirs(f"{args['save_data']}/sequence/", exist_ok=True)
    """        
    ##########################
    ###### Load Dataset ######        
    ##########################
    """
    print(f"Start making sequence dataset, {datetime.now()}")
    
    for col in ['admit_date_hour', 'disch_date_hour', 'in_date_hour', 'out_date_hour', 'edreg_date_hour', 'edout_date_hour']:
        dataset[col] = pd.to_datetime(dataset[col]) 
    
    """        ###### hw, chartevents, outputevents ######        """
    hws=pd.read_csv(f"{args['save_data']}/hws.csv", low_memory = False)
    bps=pd.read_csv(f"{args['save_data']}/bps.csv", low_memory = False)
    chartevents=pd.read_csv(f"{args['save_data']}/chartevents.csv", low_memory = False)
    outputevents=pd.read_csv(f"{args['save_data']}/outputevents.csv", low_memory = False)
    
    dataframes = [hws, bps, chartevents, outputevents] # , labevents_flag
    for df in dataframes:
        df.rename(columns={'date_hour': 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'])         
    del dataframes    
    hws = hws.dropna(subset=hws.columns)
    
    index_cols = ["subject_id", 'hadm_id', 'time']
    each_csv_columns = {"outputevents":[col for col in outputevents.columns if col not in index_cols], 
                        "chartevents":[col for col in chartevents.columns if col not in index_cols], 
                        "basic":[], "inputevents":[]}    
    output_chart = pd.merge(outputevents, chartevents, on=index_cols, how='outer', suffixes=('_output', '_chart'))
    print(f"Merged hw, output, chartevents, {datetime.now()}") 
    del outputevents, chartevents 
    
    transfer = pd.read_csv(f"{args['save_data']}/transfer.csv", low_memory = False)
    inputevents = pd.read_csv(f"{args['save_data']}/inputevents.csv", low_memory = False)
    
    print(f"Load dataset, {datetime.now()}")
    
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
    os.makedirs(f"{args['save_data']}/sequence", exist_ok=True)
    if "sequence_0.csv" not in os.listdir(f"{args['save_data']}/sequence"):
        make_basic_sequence(dataset, transfer, inputevents, hws, bps, output_chart, index_cols, args['save_data'], category_dictionary, each_csv_columns)   
    print(f"Finished making sequence dataset, {datetime.now()}") # 12시간 걸림
    
def make_basic_sequence(dataset, transfer, inputevents, hws, bps, output_chart, index_cols, save_path, category_dictionary, each_csv_columns):
    
    total = []
    k=0
    print(f"Start making, {datetime.now()}")
    for i, row in dataset.iterrows():
        if row['admit_date_hour'] > row['disch_date_hour']: continue # 사망하고 도착하는 경우?
        
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
        
        
        new_columns = sbj_df[['subject_id', 'hadm_id', 'time']]
        new_columns['time_hour'] = new_columns['time']
        new_columns['time'] = pd.to_datetime(new_columns['time'].dt.date)
        """        ###### hws ######        """
        hw = hws[(hws['subject_id'] == row['subject_id']) & (hws['hadm_id'] == row['hadm_id'])]
        hw = hw.groupby('time', as_index=False).mean()
        if not hw.empty:
            new_columns = new_columns.merge(hw[['time', 'Height', 'Weight']], on='time', how='left')
            new_columns[['Height', 'Weight']] = new_columns[['Height', 'Weight']].interpolate()
        # else:
        #     new_columns.loc[:, ['Height', 'Weight']] = 0
            
        """        ###### bps ######        """
        bp = bps[bps['subject_id'] == row['subject_id']]
        bp = bp.groupby('time', as_index=False).mean()
        if not bp.empty and ((row['admit_date_hour'] <= bp['time']) & (bp['time'] <= row['disch_date_hour'])).any():
            new_columns = new_columns.merge(bp[['time', 'BPs', 'BPd']], on='time', how='left')
            new_columns[['BPs', 'BPd']] = new_columns[['BPs', 'BPd']].interpolate()
        # else: 
        #     new_columns.loc[:, ['BPs', 'BPd']] = 0
        
        if sbj_df.shape[0] != new_columns.shape[0]:
           print(row['subject_id']) 
        sbj_df = pd.concat([sbj_df, new_columns.drop(columns=['subject_id', 'hadm_id', 'time', 'time_hour'])], axis=1)
        each_csv_columns['basic'] = [col for col in sbj_df.columns if col not in index_cols]
        
        """        ###### Inputevents ######        """
        unique_labels = inputevents['unique_label'].unique()
        each_csv_columns['inputevents'] = unique_labels
        new_columns = pd.DataFrame(np.nan, index=sbj_df.index, columns=unique_labels) # [f"input_{label}" for label in unique_labels]
        new_columns = pd.concat([sbj_df[['subject_id', 'hadm_id', 'time']], new_columns], axis=1)
        in_event = inputevents[(inputevents['subject_id'] == row['subject_id']) & (inputevents['hadm_id'] == row['hadm_id'])]

        
        for _, in_row in in_event.iterrows():
            if new_columns.loc[new_columns['time'] == in_row['start_date_hour'], in_row['unique_label']].any():
                    condition = (new_columns['time'] > in_row['start_date_hour']) & (new_columns['time'] <= in_row['end_date_hour'])
            else: # == 0
                condition = (new_columns['time'] >= in_row['start_date_hour']) & (new_columns['time'] <= in_row['end_date_hour'])    
            
            new_columns.loc[condition, in_row['unique_label']] = np.where(in_row['amount'] < in_row['rate'], 
                                                                     in_row['amount'], in_row['rate'])

        if sbj_df.shape[0] != new_columns.shape[0]:
           print(row['subject_id']) 
        sbj_df = pd.concat([sbj_df, new_columns.drop(columns=['subject_id', 'hadm_id', 'time'])], axis=1)
        
        before = sbj_df.shape[0]
        sbj_df = sbj_df.dropna(subset='subject_id')
        
        if sbj_df.shape[0] != before:
            print(f"{before} -> {sbj_df.shape[0]}")
                       
        if i%1000 == 0:
            print(f"{i} row making... {datetime.now()}")               
        total.append(sbj_df)
        # sbj_df.to_csv(f"{args['save_data']}/sequence/{row['subject_id']}_{row['hadm_id']}.csv", index=False)
        
        if ((i%5000 == 0) & (i != 0)) | (i==(dataset.shape[0]-1)):            
            concat_df = pd.concat(total, ignore_index=True)
            
            df = pd.merge(concat_df, output_chart, on=index_cols, how='left', suffixes=('_input', '_output_chart'))
            df.to_csv(f"{save_path}/sequence/sequence_{k}.csv", index=False)
            k+=1
            total = []
            del df
        
        if i==(dataset.shape[0]-1) and total != []:            
            concat_df = pd.concat(total, ignore_index=True)
            
            df = pd.merge(concat_df, output_chart, on=index_cols, how='left', suffixes=('_input', '_output_chart'))
            df.to_csv(f"{save_path}/sequence/sequence_{k}.csv", index=False)
            k+=1
            total = []
            del df
    
    import pickle
    with open(save_path + '/columns_each_csv.pkl', 'wb') as f:
        pickle.dump(each_csv_columns, f)
    del sbj_df, new_columns, transfer, inputevents, in_event, trans, dataset, total
    print(f"Finished basic sequence, {datetime.now()}") # 12시간 걸림
    


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
    
    max_end_time = row["admit_date_hour"] + pd.Timedelta(hours=168) # 168시간 7일   
    limited_end_time = min(row["disch_date_hour"], max_end_time)
    
    # 시간 간격 생성
    time_range = pd.date_range(start=row["admit_date_hour"], end=limited_end_time, freq="1H", closed="left")

    return pd.DataFrame({
        "subject_id": row["subject_id"],
        "hadm_id": row["hadm_id"],
        "time": time_range
    })
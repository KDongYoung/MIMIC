import pandas as pd
import gzip
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거


def basic(icu_path, hosp_path, save_path, args, column_dictionary):
    print(f"Preprocess icustays.csv, admissions.csv, patients.csv", end=" ")
    start=datetime.now()
    
    ## read 'icustays', 'admissions', 'patients' files
    icu_stays = pd.read_csv(gzip.open(f'{icu_path}/icustays.csv.gz'))
    admissions = pd.read_csv(gzip.open(f'{hosp_path}/admissions.csv.gz'))
    patients = pd.read_csv(gzip.open(f'{hosp_path}/patients.csv.gz'))
    transfers = pd.read_csv(gzip.open(f'{hosp_path}/transfers.csv.gz'))
    
    # remove unneeded columns
    # icu_stay = icu_stay[['subject_id', 'hadm_id', 'intime', 'outtime', 'los', 'first_careunit']]
    icu_stays['first_careunit'] = icu_stays['first_careunit'].str.extract(r'\((.*?)\)', expand=False).fillna(icu_stays['first_careunit'])
    icu_stays['last_careunit'] = icu_stays['last_careunit'].str.extract(r'\(([^)]+)\)', expand=False).fillna(icu_stays['last_careunit'])
    icu_stays = icu_stays.drop(columns="stay_id")
    admissions = admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime',
                             'admission_location', 'discharge_location',  #  'admission_type',
                             'edregtime', 'edouttime', 'hospital_expire_flag']]

    ## anchor year 연도 맞춰서 나이 wo계산
    admissions['admittime_year'] = pd.to_datetime(admissions['admittime']).dt.year    
    merged = pd.merge(admissions, patients[['subject_id', 'anchor_year', 'anchor_age']], on='subject_id', how='left')
    merged['calculated_age'] = merged['admittime_year'] - merged['anchor_year'] + merged['anchor_age']
    admissions['calculated_age'] = merged['calculated_age']
    
    basic = pd.merge(admissions, patients[['subject_id', 'gender', 'anchor_year_group', 'dod']], on='subject_id')
    basic = pd.merge(basic, icu_stays, on=['subject_id','hadm_id'], how='left')

    # datetime으로 변형
    cols = ["edregtime", "edouttime", "dischtime", "admittime", "intime", "outtime", 'deathtime', 'dod']
    basic[cols] = basic[cols].apply(lambda col: pd.to_datetime(col))
    
    # 1) admittime 보다 edregtime 먼저
    idx= basic['edregtime']<basic['admittime']
    basic.loc[idx, 'admittime'] = basic.loc[idx, 'edregtime']
    # 2) admittime 보다 intime 먼저
    idx = (basic['edregtime'].isnull()) & (basic['intime']<basic['admittime'])
    basic.loc[idx, 'admittime'] = basic.loc[idx, 'intime']
    # icu 갔다가 ed 가는 경우는 없음
    # 3) icu out 보다 dischtime 나중
    idx = (basic['dischtime']<basic['outtime'])
    basic.loc[idx, 'dischtime'] = basic.loc[idx, 'outtime']
    
    cols = ['intime', 'outtime']
    transfers[cols] = transfers[cols].apply(lambda col: pd.to_datetime(col))
    filtered_transfers = transfers.loc[transfers['eventtype'] == "discharge", ['subject_id', 'hadm_id', 'intime']].rename(columns={'intime': 'dischtime'})
    basic = basic.merge(filtered_transfers, on=['subject_id', 'hadm_id'], how='left')
    basic['dischtime'] = basic['dischtime_y'].combine_first(basic['dischtime_x'])
    basic.drop(columns=['dischtime_y', 'dischtime_x'], inplace=True)

    transfers = transfers[transfers['eventtype']!="discharge"]
    transfers.loc[transfers['eventtype']=="ED", 'careunit'] = 'ED'
    transfers = transfers[['subject_id', 'hadm_id', 'careunit', 'intime', 'outtime']]
    transfers['careunit'] = transfers['careunit'].str.extract(r'\(([^)]+)\)', expand=False).fillna(transfers['careunit'])  
    transfers['intime'] = transfers['intime'].dt.floor('H')
    transfers['outtime'] = transfers['outtime'].dt.floor('H')
    transfers.to_csv(f"{save_path}/transfer.csv", index=False)
    
    # basic['dur_before_ed'] = (basic['edregtime'] - basic['admittime']).dt.total_seconds()/3600 ## ED 가기 전 걸린 시간 (h)
    # basic['dur_before_icu'] = (basic['intime'] - basic['admittime']).dt.total_seconds()/3600 ## ICU 가기 전 걸린 시간 (h)
    basic['dur_icu'] = (basic['outtime'] - basic['intime']).dt.total_seconds()/3600 ## icu 내에 있었던 시간 = los => outtime-intime
    basic['dur_ed'] = (basic['edouttime'] - basic['edregtime']).dt.total_seconds()/3600 ## ed 내에 있었던 시간 => edouttime-edregtime
    basic['dur_inhospital'] = (basic['dischtime'] - basic['admittime']).dt.total_seconds()/3600 ## 병원에 내원한 시간 (h) => dischtime - admittime
    
    basic['time_diff'] = basic.apply(lambda row: (row['dischtime'] - row['outtime']).days
                                     if pd.notnull(row['outtime']) else (row['dischtime'] - row['edregtime']).days,
                                     axis=1)
    basic = basic[(0<basic['dur_inhospital'])] # & (basic['dur_inhospital']<=168)]
    
    basic['dod'].fillna(pd.to_datetime("2222-12-31"), inplace=True) ## NaT는 데이터를 작성할 때까지 생존했음을 의미하는 것으로 고려 (basic['dod'].dt.year.max()가 2211)
    basic.loc[basic['deathtime'].notnull(), 'dischtime'] = basic.loc[basic['deathtime'].notnull(), 'deathtime'] # deathtime이 있는 경우
    basic.loc[basic['deathtime'].notnull(), 'dod'] = basic.loc[basic['deathtime'].notnull(), 'deathtime'] # deathtime이 있는 경우
    basic.loc[(basic['deathtime'].isnull()) & (basic['hospital_expire_flag']==1), 'dod'] = basic.loc[(basic['deathtime'].isnull()) & (basic['hospital_expire_flag']==1), 'dischtime'] # deathtime은 없는데 hospital in-mortality
    
    ## ed, icu, hospital in, out 나눠서 저장해보기 (아니면 column 수정)
    basic['admit_date_hour'] = basic['admittime'].dt.floor('H')
    basic['disch_date_hour'] = basic['dischtime'].dt.floor('H')
    basic['in_date_hour'] = basic['intime'].dt.floor('H')
    basic['out_date_hour'] = basic['outtime'].dt.floor('H')
    basic['edreg_date_hour'] = basic['edregtime'].dt.floor('H')
    basic['edout_date_hour'] = basic['edouttime'].dt.floor('H')
    
    if args['target'] == 'mortality':
        basic['mortality'] = (basic['dod'] - basic['dischtime']).dt.days
        basic = basic[basic['mortality'] >= 0] # dod < dischtime 제거
    # elif args['target'] == 'readmission':
    #     basic = basic.sort_values(['subject_id', 'edregtime'])
    #     # 여러개의 icu를 방문한 사람도 같은 readmission변수를 부여하기 위해
    #     a=basic[['subject_id', "hadm_id", 'edregtime', 'dischtime']].drop_duplicates()
    #     a['readmission'] = (a.groupby('subject_id')['edregtime'].shift(-1) - a['dischtime']).dt.days
    #     basic = pd.merge(basic, a, on = ["subject_id", "hadm_id", "admittime", "dischtime"])
    #     # icu_stay['readmission'] = (icu_stay.groupby('subject_id')['admittime'].shift(-1) - icu_stay['dischtime']).dt.days
    
    ## categorical feature 수치형으로 바꾸기
    basic.loc[basic[['intime', 'outtime']].isnull().all(axis=1), ['first_careunit', 'last_careunit']] = 'None'
    basic['discharge_location'].fillna('Unknown', inplace=True)
    
    all_unique_values = pd.concat([basic[col] for col in ['discharge_location', 'admission_location', 'first_careunit', 'last_careunit']]).unique() # 'admission_type', 
    value_mapping = {value: idx for idx, value in enumerate(all_unique_values)}
    value_mapping["ED"] = max(value_mapping.values())+1
    
    with open(f'{save_path}/category_encoding.txt', 'w') as f:
        f.write(str(value_mapping))
    
    for column in ['discharge_location', 'admission_location', 'first_careunit', 'last_careunit']: # 'admission_type', 
        basic[column] = basic[column].map(value_mapping).astype(int)
    
    # basic['admission_type'] = basic['admission_type'].astype('category')    
    # for c in ['admission_type', 'discharge_location', 'admission_location', 'first_careunit', 'last_careunit']:
    #     mapping = dict(enumerate(basic[c].cat.categories))
    #     basic[c] = basic[c].cat.codes
    #     with open(f'{save_path}/category_encoding.txt', 'a') as f:
    #         f.write(f'{c} // {mapping}\n')
        
    basic = basic.drop(columns=['admittime', 'dischtime', 'deathtime', 'dod', 'edregtime', 'edouttime', 'admittime_year', 'intime', 'outtime', 'time_diff', 'los']) # los는 dur_icu로 
    basic.to_csv(f"{save_path}/mimic4_{args['target']}.csv", index=False)
    basic[['subject_id','hadm_id']].drop_duplicates().to_csv(f"{save_path}/mimic4_subject.csv", index=False)
    print(f'{datetime.now()-start} (hh:mm:ss.ms)')
    
    column_types = {
        "numerical": ['calculated_age', 'dur_icu', 'dur_inhospital', 'dur_ed', 'mortality'],
        "categorical": ['gender', 'mortality', 'first_careunit', 'last_careunit', # 'admission_type', 
                        'anchor_year_group', 'discharge_location', 'admission_location', 'hospital_expire_flag'],
        "id": ['subject_id', 'hadm_id'],
        "time": ['admit_date_hour', 'disch_date_hour', 'in_date_hour', 
                 'out_date_hour', 'edreg_date_hour', 'edout_date_hour']
        }

    column_dictionary = {row: col_type for col_type, rows in column_types.items() for row in rows}
    
    return admissions, column_dictionary
    
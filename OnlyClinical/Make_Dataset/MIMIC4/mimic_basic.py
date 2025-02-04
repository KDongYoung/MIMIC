import pandas as pd
import gzip
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거


def basic(icu_path, hosp_path, save_path, args, column_dictionary):
    print(f"Preprocess icustays.csv, admissions.csv, patients.csv", end=" ")
    start=datetime.now()
    
    ## read 'icustays', 'admissions', 'patients' files
    icu_stay = pd.read_csv(gzip.open(f'{icu_path}/icustays.csv.gz'))
    admission = pd.read_csv(gzip.open(f'{hosp_path}/admissions.csv.gz'))
    patients = pd.read_csv(gzip.open(f'{hosp_path}/patients.csv.gz'))

    # remove unneeded columns
    icu_stay = icu_stay[['subject_id', 'hadm_id', 'intime', 'outtime', 'los', 'first_careunit']]
    icu_stay['first_careunit'] = icu_stay['first_careunit'].str.extract(r'\(([^)]+)\)')
    admission = admission[['subject_id', 'hadm_id', 'hospital_expire_flag', 'admittime', 'dischtime', 'edregtime', 'edouttime', 'deathtime']]
    patients = patients[['subject_id', 'gender', 'anchor_age', 'dod', 'anchor_year_group']]

    admission = pd.merge(admission, patients, on='subject_id')
    icu_stay = pd.merge(icu_stay, admission, on=['subject_id','hadm_id']) #, how='right')
    
    # datetime으로 변형
    cols = ["edregtime", "edouttime", "dischtime", "admittime", "intime", "outtime", 'deathtime', 'dod']
    icu_stay[cols] = icu_stay[cols].apply(lambda col: pd.to_datetime(col))
    
    # 1) admittime 보다 edregtime 먼저
    idx= icu_stay['edregtime']<icu_stay['admittime']
    icu_stay.loc[idx, 'admittime'] = icu_stay.loc[idx, 'edregtime']
    # 2) admittime 보다 intime 먼저
    idx = (icu_stay['edregtime'].isnull()) & (icu_stay['intime']<icu_stay['admittime'])
    icu_stay.loc[idx, 'admittime'] = icu_stay.loc[idx, 'intime']
    # icu 갔다가 ed 가는 경우는 없음
    # 3) icu out 보다 dischtime 나중
    idx = (icu_stay['dischtime']<icu_stay['outtime'])
    icu_stay.loc[idx, 'dischtime'] = icu_stay.loc[idx, 'outtime']
    
    # icu_stay['dur_before_ed'] = (icu_stay['edregtime'] - icu_stay['admittime']).dt.total_seconds()/60 ## ED 가기 전 걸린 시간 (분)
    icu_stay['dur_before_icu'] = (icu_stay['intime'] - icu_stay['admittime']).dt.total_seconds()/60 ## ICU 가기 전 걸린 시간 (분)
    icu_stay['dur_icu'] = (icu_stay['outtime'] - icu_stay['intime']).dt.total_seconds()/60 ## icu 내에 있었던 시간 = los => outtime-intime
    # icu_stay['dur_ed'] = (icu_stay['edouttime'] - icu_stay['edregtime']).dt.total_seconds()/60 ## icu 내에 있었던 시간 => edouttime-edregtime
    icu_stay['dur_inhospital'] = (icu_stay['dischtime'] - icu_stay['admittime']).dt.total_seconds()/60 ## 병원에 내원한 시간 (분) => dischtime - admittime
    
    
    # dischtime가 outtime 보다 하루 이상 이전인 경우 제외 / dur 계열들이 nan인 곳 0으로 채우기
    icu_stay = icu_stay[(icu_stay['outtime']-icu_stay['dischtime']).dt.days<1]
    # icu_stay[['dur_before_ed', 'dur_before_icu', 'dur_ed', 'dur_icu']].fillna(0) ### ed를 방문에 관한 정보에 null이 많음, 이는 나중에 안 지워질까?
    
    ## datetime -> date info only
    icu_stay['dod'].fillna(pd.to_datetime("2210-12-31"), inplace=True) ## NaT는 데이터를 작성할 때까지 생존했음을 의미하는 것으로 고려
    
    # dod only has date (no time)
    icu_stay.loc[icu_stay['deathtime'].notnull(), 'dischtime'] = icu_stay.loc[icu_stay['deathtime'].notnull(), 'deathtime'] # deathtime이 있는 경우
    icu_stay.loc[icu_stay['deathtime'].notnull(), 'dod'] = icu_stay.loc[icu_stay['deathtime'].notnull(), 'deathtime'] # deathtime이 있는 경우
    icu_stay.loc[(icu_stay['deathtime'].isnull()) & (icu_stay['hospital_expire_flag']==1), 'dod'] = icu_stay.loc[(icu_stay['deathtime'].isnull()) & (icu_stay['hospital_expire_flag']==1), 'dischtime'] # deathtime은 없는데 hospital in-mortality
    # icu_stay.loc[(icu_stay['deathtime'].isnull()) & (icu_stay['hospital_expire_flag']==0), 'dod'] = icu_stay.loc[(icu_stay['deathtime'].isnull()) & (icu_stay['hospital_expire_flag']==0), 'dod']
    
    ## target으로 사용될 column
    icu_stay = icu_stay.sort_values(['subject_id', 'edregtime'])
    
    if args['target'] == 'mortality':
        icu_stay['mortality'] = (icu_stay['dod'] - icu_stay['dischtime']).dt.days
        icu_stay = icu_stay[icu_stay['mortality'] >= 0] # dod < dischtime 제거
    elif args['target'] == 'readmission':
        icu_stay = icu_stay.sort_values(['subject_id', 'edregtime'])
        # 여러개의 icu를 방문한 사람도 같은 readmission변수를 부여하기 위해
        a=icu_stay[['subject_id', "hadm_id", 'edregtime', 'dischtime']].drop_duplicates()
        a['readmission'] = (a.groupby('subject_id')['edregtime'].shift(-1) - a['dischtime']).dt.days
        icu_stay = pd.merge(icu_stay, a, on = ["subject_id", "hadm_id", "admittime", "dischtime"])
        # icu_stay['readmission'] = (icu_stay.groupby('subject_id')['admittime'].shift(-1) - icu_stay['dischtime']).dt.days
    
    icu_stay.drop(columns=['dod', 'hospital_expire_flag', 'intime', 'outtime'])     
    if len(args["missing_value"])==0: args["missing_value"].extend(['gender', 'anchor_age', 'mortality', 'los'])
    icu_stay.to_csv(f"{save_path}/mimic4_{args['target']}.csv", index=False)
    print(f'{datetime.now()-start} (hh:mm:ss.ms)')
    
    column_dictionary.update({row: "numerical" for row in ['anchor_age', 'los', 'dur_before_icu', 'dur_icu', 'dur_inhospital', 'mortality']})
    column_dictionary.update({row: "categorical" for row in ['gender', 'mortality', 'subject_id', 'hadm_id', 'first_careunit', 'anchor_year_group']})

    return icu_stay, admission, args, column_dictionary
    

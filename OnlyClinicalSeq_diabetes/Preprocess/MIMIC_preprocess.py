import pandas as pd

from Utils.Preprocess.Normalization import Normalize
from Utils.Preprocess.Imputation import Imputation
import os
import ast
from Data_Load.Datasets import *

def main(args):
    args["column_info"] = pd.read_csv(f"{args['save_data']}/column_info.csv", index_col=[0]).to_dict()['0'] # {key: value['0'] for key, value in d.items()}    
    column_types = {
        **{c: "numerical" for c in ['calculated_age', 'dur_icu', 'dur_inhospital', 'dur_ed', 'mortality', 'Height', 'Weight', 'BPs', 'BPd']},
        **{c: "categorical" for c in ['gender', 'mortality', 'location', 'anchor_year_group', args['target']]},
        **{c: "id" for c in ['subject_id', 'hadm_id']},
        'time': "time"
    }
    args['column_info'].update(column_types) 
    
    with open(args['save_data'] + '/category_encoding.txt', 'r') as f:
        line=f.read().split("\n")
    category_dictionary = ast.literal_eval(line[0])
    
    if not os.path.exists(f"{args['save_data']}/domain/{args['domain']}.csv"):
        icd=pd.read_csv(f"{args['save_data']}/icd.csv", low_memory = False)
        icd=find_icd(args['data_root'], args['save_data'], args['total_path'], icd, args['domain'])  
        
        DOMAIN = []
        sbj_cnt = 0
        for d_name in sorted(os.listdir(f"{args['save_data']}/sequence")):
            print(f"{'-'*5} {d_name} {'-'*5}")
            dataset = pd.read_csv(f"{args['save_data']}/sequence/{d_name}")
            
            dataset['gender'].replace({'M':0, 'F':1}, inplace=True)
            dataset['anchor_year_group'].replace({'2008 - 2010':0 , '2011 - 2013':1 ,'2014 - 2016':2, '2017 - 2019':3}, inplace=True)

            dataset['unique_id']=dataset['subject_id'].astype(str) + "_" + dataset['hadm_id'].astype(str)
            
            domain = pd.merge(dataset, icd, on=['subject_id', 'hadm_id'])

            domain.drop(columns=['icd_code', 'long_title', 'Feeding Weight', 'Daily Weight'], inplace=True)
            
            if args['over_age']: domain=over_age(domain) # admit 나이 기준
            if args['over_height_over_weight']: domain=over_height_over_weight(domain)
            if args['over_24_icustay']: domain=over_24_icustay(domain, category_dictionary ,args["icustay_day"]) 
            
            domain = domain.drop(columns=['subject_id', 'hadm_id'])
            null_unique_ids = domain.groupby('unique_id')['HbA1c'].apply(lambda x: x.isna().all())
            null_unique_ids = null_unique_ids[null_unique_ids].index.tolist()
            domain = domain[~domain['unique_id'].isin(null_unique_ids)]
            DOMAIN.append(domain) # subject_id, hadm_id, icd_code 제외
            
            a = domain.drop_duplicates(subset=['unique_id'])
            sbj_cnt += a.shape[0]
                
        print(sbj_cnt)    

        df=pd.concat(DOMAIN)
        
        os.makedirs(f"{args['save_data']}/domain", exist_ok=True)
        df.to_csv(f"{args['save_data']}/domain/{args['domain']}.csv", index=False)
    
    else:
        df = pd.read_csv(f"{args['save_data']}/domain/{args['domain']}.csv")
        
    time=168
    if 'mortality' in df.columns:
        df.drop(columns="mortality", inplace=True) ##dataset 만들때 생겨버림
    
    group_counts = df.groupby('unique_id').size()
    valid_ids = group_counts[group_counts == time].index # 개수가 168인 unique_id 필터링
    df = df[df['unique_id'].isin(valid_ids)]
    
    subject_mean = df.groupby('unique_id')['HbA1c'].mean() #.transform(lambda x: x.mean() if x.notna().any() else x)
    domain_targets = list(subject_mean)
    # df['HbA1c'] = df['HbA1c'].fillna(subject_mean)
    domain_datasets = df    
        
    rate = 0.1
    selected_feature_name = exist_feature_in_domains(domain_datasets, args['domain'], rate) # rate% 미만으로 비어 있는 feature 찾기
    print(f"{len(selected_feature_name)} feature below {rate*100}% null")

    index_cols = ['unique_id', 'icd_code', 'anchor_year_group', args['target']]
    args["selected_feature_name"]=sorted([col for col in selected_feature_name if col not in index_cols])
    args['category'] = [key for key, value in args["column_info"].items() if value == 'categorical' and key in args['selected_feature_name']]
    args['number'] = [key for key, value in args["column_info"].items() if value == 'numerical' and key in args['selected_feature_name']]
      
    domain = domain_datasets[args["selected_feature_name"] + ["unique_id"]]
    domain_target = domain_targets
    domian_id = domain_datasets["unique_id"]
    
    # imputation
    domain = Imputation(domain, args['category'], args['number'], args['target'])

    ### sliding window로 dataset 만들기
    seq_length = args['seq_length']  # 원하는 seq_length (3, 6, 12 등으로 설정 가능)
    total_df= TabSeqDataset(domain.groupby('unique_id').apply(lambda x: x[[col for col in x.columns if col !="unique_id" and args['column_info'][col]=="categorical"] + \
                                                                          [col for col in x.columns if col !="unique_id" and args['column_info'][col]=="numerical"]].values), 
                                domain_target, 
                                seq_length,
                                args['stride']) 

    return total_df


### 모든 domain 별로 값이 있는 feature 선택
def exist_feature_in_domains(data, domain, rate):
    common_feature = None
    
    domain = data.groupby('unique_id').mean()
    below_missing_features = domain.loc[:, domain.isnull().sum() < int(rate*domain.shape[0])].columns
    
    if common_feature is None:
        common_feature = set(below_missing_features)
    else:
        common_feature.intersection_update(below_missing_features)
    return common_feature

def find_icd(data_root, save_root, total_path, data, domain_group):
    print(f"Select related ICD-10 ")
    
    if not os.path.isfile(f"{save_root}/unique_ICD({domain_group}).csv"): 
        sheet_name=domain_group  
        
        icd_prefix = [pd.read_excel(f"{data_root}/icd_diagnoses.xlsx", sheet_name=s_name)["icd_code"].str[:3].unique() for s_name in sheet_name]
        data=data[data['seq_num']<=5] # 5
        
        total_df=pd.DataFrame([])
        for i, prefixes in enumerate(icd_prefix):       
            df=pd.concat([data[data["icd_code"].str.contains(prefix)] for prefix in prefixes])
            df['icd_code'] = i # +1 # domain 별로 icd relabeling
            df.reset_index(drop=True, inplace=True)
            df=df.drop_duplicates(subset=['subject_id', 'hadm_id'])
            print(f"{sheet_name[i]} disease subject: {df.shape[0]}")
            
            df["unique_id"] = df[["subject_id", "hadm_id"]].apply(lambda row: '_'.join(row.astype(str)), axis=1)
            total_df=pd.concat([total_df, df], axis=0)
        
        # global filtering based on minimum 'seq_num' for each unique_id
        total_df.reset_index(inplace=True, drop=True) # reset index안하면 중복 index 존재
        total_df = total_df.loc[total_df.groupby("unique_id")["seq_num"].idxmin()]
        
        # SUBJECT는 확인용
        SUBJECT={}
        for i in range(len(domain_group)):
            SUBJECT[sheet_name[i]] = list(set(total_df.loc[total_df["icd_code"]==i, 'unique_id']))
            
        total_df = total_df.drop(columns=['seq_num', 'unique_id'])
        
        with open(total_path + '/args.txt', 'a') as f:
            f.write('# subjects in each domain ' + str({key: len(values) for key, values in SUBJECT.items()}) + '\n')
            
        print(f"Total subject: {total_df.shape[0]}")
        total_df.to_csv(f"{save_root}/unique_ICD({','.join(domain_group)}).csv", index=False) # ({','.join(domain_group)})
    else:
        total_df=pd.read_csv(f"{save_root}/unique_ICD({domain_group}).csv") #.drop(columns='index')   
            
    return total_df
    
def over_age(df):
    age=[18, 89]
    df=df.astype({'calculated_age':int})
    df=df[(age[0]<=df["calculated_age"]) & (df["calculated_age"]<=age[1])]
    return df

def over_height_over_weight(df):
    df = df.groupby(['subject_id', 'hadm_id']).filter(
                                                        lambda x: (50 <= x.loc[x['Height'] != 0, 'Height'].mean() <= 250) and (40 <= x.loc[x['Weight'] != 0, 'Weight'].mean() <= 300)
                                                    )
    return df

def over_24_icustay(df, category_dictionary, icustay_day):
    icu_location = {'CCU', 'MICU', 'Neuro Intermediate', 'SICU', 'CVICU', 'Neuro SICU', 'Neuro Stepdown', 'MICU/SICU', 'TSICU'}
    
    df_filtered = df[df['location'].isin([category_dictionary[k] for k in icu_location])]
    group_counts = df_filtered.groupby(['subject_id', 'hadm_id'])['location'].count()
    valid_groups = group_counts[group_counts >= icustay_day * 24].index
    df = df[df.set_index(['subject_id', 'hadm_id']).index.isin(valid_groups)]
    return df

def class_division(df, target):     ## MAPE를 계산하는 것으로 해당 함수 사용 x  
    
    """
    2 class
    HbA1c < 6.5: normal 0
    HbA1c >= 6.5: diabetes 1
    """
    df[df[target]<6.5]=0
    df[df[target]>=6.5]=1
    n_class=2   

    return df, n_class

def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i+seq_length])
    return np.array(X)






    # # 값이 있는 feature 찾기
    # domain2=df.groupby('unique_id').mean()
    # print(f"{len(domain2.loc[:, domain2.isnull().sum() < int(0.1*domain2.shape[0])].columns)} feature: {domain2.loc[:, domain2.isnull().sum() < int(0.1*domain2.shape[0])].columns}")
    # print(f"{len(domain2.loc[:, domain2.isnull().sum() < int(0.2*domain2.shape[0])].columns)} feature: {domain2.loc[:, domain2.isnull().sum() < int(0.2*domain2.shape[0])].columns}")
    # print(f"{len(domain2.loc[:, domain2.isnull().sum() < int(0.3*domain2.shape[0])].columns)} feature: {domain2.loc[:, domain2.isnull().sum() < int(0.3*domain2.shape[0])].columns}")
    # print(f"{len(domain2.loc[:, domain2.isnull().sum() < int(0.4*domain2.shape[0])].columns)} feature: {domain2.loc[:, domain2.isnull().sum() < int(0.4*domain2.shape[0])].columns}")
    # print(f"{len(domain2.loc[:, domain2.isnull().sum() < int(0.5*domain2.shape[0])].columns)} feature: {domain2.loc[:, domain2.isnull().sum() < int(0.5*domain2.shape[0])].columns}")
    # print(f"{len(domain2.loc[:, domain2.isnull().sum() < int(0.6*domain2.shape[0])].columns)} feature: {domain2.loc[:, domain2.isnull().sum() < int(0.6*domain2.shape[0])].columns}")
    # print(f"{len(domain2.loc[:, domain2.isnull().sum() < int(0.7*domain2.shape[0])].columns)} feature: {domain2.loc[:, domain2.isnull().sum() < int(0.7*domain2.shape[0])].columns}")
    # print(f"{len(domain2.loc[:, domain2.isnull().sum() < int(0.8*domain2.shape[0])].columns)} feature: {domain2.loc[:, domain2.isnull().sum() < int(0.8*domain2.shape[0])].columns}")
    # print(f"{len(domain2.loc[:, domain2.isnull().sum() < int(0.9*domain2.shape[0])].columns)} feature: {domain2.loc[:, domain2.isnull().sum() < int(0.9*domain2.shape[0])].columns}")
    # print(f"{len(domain2.loc[:, domain2.isnull().sum() < int(1.0*domain2.shape[0])].columns)} feature: {domain2.loc[:, domain2.isnull().sum() < int(1.0*domain2.shape[0])].columns}")
    
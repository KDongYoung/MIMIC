import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
# from Utils.Preprocess.Feature_selection import Feature_Selection
from Utils.Preprocess.Normalization import Normalize
from Utils.Preprocess.Imputation import Imputation
import os
import ast

"""
Make a tabular dataset
X: row data
Y: class score
"""
class CustomDataset(Dataset):
    def __init__(self, X, y, dis_id, group):
        self.X = X
        self.y = y
        self.len = len(self.y)
        self.dis_id = dis_id
        self.group = group
        
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        X = self.X.iloc[idx].to_numpy().astype('float32')  
        y = int(self.y.iloc[idx])
    
        return X, y, self.dis_id.iloc[idx], self.group.iloc[idx]


def main(args):
    args["column_info"] = pd.read_csv(f"{args['save_data']}/column_info.csv", index_col=[0]).to_dict()['0'] # {key: value['0'] for key, value in d.items()}    
    column_types = {
        "numerical": ['calculated_age', 'dur_icu', 'dur_inhospital', 'dur_ed', 'mortality'],
        "categorical": ['gender', 'mortality', 'loaction', 'anchor_year_group'],
        "id": ['subject_id', 'hadm_id'],
        "time": ['admit_date_hour', 'disch_date_hour', 'in_date_hour', 
                 'out_date_hour', 'edreg_date_hour', 'edout_date_hour']
        }
    args['column_info'].update(column_types) 
    
    icd=pd.read_csv(f"{args['save_data']}/icd.csv", low_memory = False)
    icd=find_icd(args['data_root'], args['save_data'], args['total_path'], icd, args['domain'])  
    
    with open(args['save_data'] + '/category_encoding.txt', 'r') as f:
        line=f.read().split("\n")
    category_dictionary = ast.literal_eval(line[0])
    
    time=24
    sbj_info=pd.DataFrame([])
    DOMAIN=[[] for _ in range(len(args['domain']))]
    sbj_cnt = [[0, [0,0]] for _ in range(len(args['domain']))]
    for d_name in sorted(os.listdir(f"{args['save_data']}/sequence")):
        print(f"{'-'*5} {d_name} {'-'*5}")
        dataset = pd.read_csv(f"{args['save_data']}/sequence/{d_name}")
        
        dataset['gender'].replace({'M':0, 'F':1}, inplace=True)
        dataset['anchor_year_group'].replace({'2008 - 2010':0 , '2011 - 2013':1 ,'2014 - 2016':2, '2017 - 2019':3}, inplace=True)

        dataset['unique_id']=(dataset['subject_id'].astype(str) + "_" + dataset['hadm_id'].astype(str)).astype(int)
        
        for idx, domain_name in enumerate(args['domain']):
            domain = pd.merge(dataset, icd, on=['subject_id', 'hadm_id'])
            domain = domain.loc[domain['icd_code']==idx, :]
            domain.drop(columns=['icd_code', 'long_title', 'Feeding Weight', 'Daily Weight'], inplace=True)
            
            if args['over_age']: domain=over_age(domain) # admit 나이 기준
            if args['over_height_over_weight']: domain=over_height_over_weight(domain)
            if args['over_24_icustay']: domain=over_24_icustay(domain, category_dictionary ,args["icustay_day"]) 
            
            
            domain['time'] = pd.to_datetime(domain['time'])
            domain['adjust_time'] = domain.groupby('unique_id')['time'].transform(lambda x: (x - x.min()).dt.total_seconds())
            
            
            domain = domain.drop(columns=['subject_id', 'hadm_id'])
            
            
            
            
            DOMAIN[idx].append(domain) # subject_id, hadm_id, icd_code 제외
            
            a = domain.drop_duplicates(subset=['unique_id'])
            sbj_cnt[idx][0] +=a.shape[0]
            b=Counter(a[args['target']])
            sbj_cnt[idx][1] = [sbj_cnt[idx][1][i] + b.get(i, 0) for i in range(2)]
        
        # dataset = dataset.groupby('unique_id').filter(lambda x: x.shape[0] >= time)
        # # missing_info = valid_groups.groupby('unique_id').apply(lambda x: x.head(time).notnull().sum())
        # # print(Counter(valid_groups.groupby('unique_id')['mortality'].mean()))
        # # sbj_info = pd.concat([sbj_info, missing_info]) 
        # missing_info = dataset.groupby('unique_id').apply(lambda x: x.notnull().sum())
        # missing_info['inhospital_time'] = dataset.groupby('unique_id').size()
        # sbj_info = pd.concat([sbj_info, missing_info])
        
        
    print(sbj_cnt)    
    print(sbj_info)    
        
    for idx, domain_name in enumerate(args['domain']):
        print(f"{domain_name}: {sbj_cnt[idx][0]} SAMPLE, {sbj_cnt[idx][1]}")
        df=pd.concat(DOMAIN[idx])
        pd.DataFrame(df.loc[:, ~((df == 0).all() | df.isnull().all())].columns).to_csv(f"{args['save_data']}/domain/{domain_name}.csv",)

      
    
        
    
    #### 1,3,5,10,24시간 선택
    time = 24
    dataset = []
    index_cols = ["subject_id", 'hadm_id']
    for group in DOMAIN.groupby(index_cols):
        if group.shape[0]>24:
            # dataset[i][:time].mean()
            group.drop(columns= "time", inplace=True)
            dataset.append(group[:time].mean())
    dataset = pd.concat(dataset)
    
       
    print(f'Dataset shape: ', dataset.shape, end=' -> ')
    dataset = dataset.loc[:, dataset.isnull().sum()<int(0.9*dataset.shape[0])] # nan이 90%이상인 feature 제외
    selected_feature_name = exist_feature_in_domains(dataset, args['domain'])    
    dataset = dataset[selected_feature_name]
    print(dataset.shape, end=' -> ')
    
    # 같은 subject_id이면 mortality가 같을 것이다
    dataset = dataset.groupby('subject_id').apply(lambda group: fill_mortality(args['target'], group)).reset_index(drop=True)
    dataset=dataset.drop_duplicates(ignore_index=True) # row의 모든 column값이 같은 것 제거
    print(dataset.shape)  
    
    
    dataset["unique_id"] = dataset[index_cols].apply(lambda row: '_'.join(row.astype(str)), axis=1) ## subject_id, hadm_id 합치기
    dataset.drop(columns=index_cols, inplace=True)
    
    args['category'] = [key for key, value in args["column_info"].items() if value == 'categorical']
    args['number'] = [key for key, value in args["column_info"].items() if value == 'numerical']
    
    index_cols = ['unique_id', 'icd_code', 'anchor_year_group', args['target']]
    args["selected_feature_name"]=sorted([col for col in dataset.columns if col not in index_cols])
    args['column_info']={col:args['column_info'][col] for col in dataset.columns if col not in index_cols}
    args['cat_num']=sum(value == 'categorical' for value in args['column_info'].values()) # categorical 변수 개수
    args['vocab_sizes'] = dataset[[col for col in dataset.columns if col not in index_cols and args['column_info'][col]=="categorical"]].max().tolist()
    
    total_df=[]
    c=[0]*args["n_classes"]
    
    for idx, domain_name in enumerate(args['domain']):
        
        domain = dataset.loc[dataset['icd_code']==idx, :]
        domain = domain[[col for col in domain.columns if col not in ['icd_code']]] # subject_id, hadm_id, icd_code 제외
        print(f"{args['domain'][idx]}: {domain[args['target']].shape[0]} SAMPLE, {dict(sorted(Counter(domain[args['target']]).items()))}")
                
        ## imputation
        domain, args["column_info"] = Imputation(args['dataset_name'], domain, args['imputation'], args['category'], args['number'], args['target'], args['seed'], args["column_info"])
        domain[args["selected_feature_name"]] = Normalize(args['normalize'], domain[args["selected_feature_name"]])
        
        
        ## CATEGORICAL, NUMERICAL 순서로 재배열 (나중 embedding을 위해)
        total_df.append(CustomDataset(domain[[col for col in domain.columns if col not in ["unique_id", args['target'], 'anchor_year_group'] and args['column_info'][col]=="categorical"] + \
                                          [col for col in domain.columns if col not in ["unique_id", args['target'], 'anchor_year_group'] and args['column_info'][col]=="numerical"]], 
                                   domain[args['target']], 
                                   domain["unique_id"], 
                                   domain['anchor_year_group']))
        
        print(f"{domain_name}: {Counter(domain[args['target']])}")
        for class_idx, count in Counter(domain[args['target']]).items():
            c[int(class_idx)] += count
            
    print(f"class별: {c}")

    return total_df


### 모든 domain 별로 값이 있는 feature 선택
def exist_feature_in_domains(data, domain):
    common_feature = None
    for idx, _ in enumerate(domain):
        domain = data.loc[data['icd_code']==idx, :]
        not_missing_features = domain.columns[domain.isnull().sum() < domain.shape[0]]
        
        if common_feature is None:
            common_feature = set(not_missing_features)
        else:
            common_feature.intersection_update(not_missing_features)
    return common_feature

def find_icd(data_root, save_root, total_path, data, domain_group):
    print(f"Select related ICD-10 ")
    
    if not os.path.isfile(f"{save_root}/unique_ICD({','.join(domain_group)}).csv"): 
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
        total_df.reset_index(inplace=True) # reset index안하면 중복 index 존재
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
        total_df=pd.read_csv(f"{save_root}/unique_ICD({','.join(domain_group)}).csv")    
            
    return total_df
    
def over_age(df):
    age=[18, 89]
    # print("Include subject over 18 and below 89: ", end="")
    df=df.astype({'calculated_age':int})
    df=df[(age[0]<=df["calculated_age"]) & (df["calculated_age"]<=age[1])]
    # print(df[['subject_id', 'hadm_id']].drop_duplicates().shape[0])
    return df

def over_height_over_weight(df):
    # print(f"Include subject height (50~250cm) and weight (40~300kg): ", end=" ")
    # ## 단위를 고려하면 743 row로 줄어듦 -> 90 subject   HEIGHT 2.54, WEIGHT 2.2
    df = df.groupby(['subject_id', 'hadm_id']).filter(
                                                        lambda x: (50 <= x.loc[x['Height'] != 0, 'Height'].mean() <= 250) and (40 <= x.loc[x['Weight'] != 0, 'Weight'].mean() <= 300)
                                                    )
    # df=df[(50.0 <= df["Height"]) & (df["Height"]<= 250.0) & 
    #       (40.0 <= df["Weight"]) & (df["Weight"]<= 300.0)] 
    # df=df.groupby(['subject_id', 'hadm_id' , 'anchor_year_group']).mean().reset_index() 
    # print(df[['subject_id', 'hadm_id']].drop_duplicates().shape[0])
    return df

def over_24_icustay(df, category_dictionary, icustay_day):
    # print("Include subject icu 24hr stay: ", end="")
    icu_location = {'CCU', 'MICU', 'Neuro Intermediate', 'SICU', 'CVICU', 'Neuro SICU', 'Neuro Stepdown', 'MICU/SICU', 'TSICU'}
    
    df_filtered = df[df['location'].isin([category_dictionary[k] for k in icu_location])]
    group_counts = df_filtered.groupby(['subject_id', 'hadm_id'])['location'].count()
    valid_groups = group_counts[group_counts >= icustay_day * 24].index
    df = df[df.set_index(['subject_id', 'hadm_id']).index.isin(valid_groups)]
    
    # df=df[df_filtered.groupby(['subject_id', 'hadm_id'])['location'].count()>=icustay_day*24] # los=1은 1day
    # print(df[['subject_id', 'hadm_id']].drop_duplicates().shape[0])
    return df

def fill_mortality(target, group):
    if group[target].isnull().all():
        return group
    group[target] = group[target].fillna(group[target].mode().sort_values().iloc[0])
    return group
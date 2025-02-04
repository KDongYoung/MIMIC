import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
from Utils.Preprocess.Feature_selection import Feature_Selection
from Utils.Preprocess.Normalization import Normalize
from Utils.Preprocess.Imputation import Imputation
import os

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


def main(args, dataset):
    args["column_info"] = pd.read_csv(f"{args['save_data']}/column_info.csv", index_col=[0]).to_dict()['0'] # {key: value['0'] for key, value in d.items()}

    ## icd, mimic4(icustays), hw, chartevents, inputevents, outputevents
    # dataset=mimic4
    icd=pd.read_csv(f"{args['save_data']}/icd.csv", low_memory = False)
    hw=pd.read_csv(f"{args['save_data']}/hw.csv", low_memory = False)
    chartevents=pd.read_csv(f"{args['save_data']}/chartevents.csv", low_memory = False)
    inputevents=pd.read_csv(f"{args['save_data']}/inputevents.csv", low_memory = False)
    outputevents=pd.read_csv(f"{args['save_data']}/outputevents.csv", low_memory = False)
    
    if args['labevents']:
        labevents=pd.read_csv(f"{args['save_data']}/labevents.csv", low_memory = False)
    
    icd=find_icd(args['data_root'], args['save_data'], args['total_path'], icd, args['domain'])  
    
    target_columns = ['subject_id', 'hadm_id', 'los', 'first_careunit', 'gender', 'anchor_age', 'anchor_year_group', 'dur_before_icu', 'dur_icu', 'dur_inhospital', args['target']]
    dataset = dataset.loc[:, ~dataset.columns.str.contains('time')][target_columns]
    dataset['first_careunit'].replace({key:i for i, key in enumerate(set(dataset['first_careunit']))}, inplace=True) ## first care unit의 text를 category로 만들기 ('nan'이 0)
    index_cols=["subject_id", "hadm_id"]
    basic=pd.merge(dataset, hw, on=index_cols) # [['subject_id', 'hadm_id', 'gender', 'anchor_age', 'anchor_year_group', args['target'], 'los']]
    basic['gender'].replace({'M':0, 'F':1}, inplace=True)
    basic['anchor_year_group'].replace({'2008 - 2010':0 , '2011 - 2013':1 ,'2014 - 2016':2, '2017 - 2019':3}, inplace=True)
    
    if args['over_age']: basic=over_age(basic)
    if args['over_height_over_weight']: basic=over_height_over_weight(basic)
    if args['over_24_icustay']: basic=over_24_icustay(basic, args["icustay_day"]) 
    
    df=(
        basic.merge(icd[['subject_id', 'hadm_id', 'icd_code']], on = index_cols)
             .merge(inputevents, on = index_cols)
             .merge(outputevents, on = index_cols)
             .merge(chartevents, on = index_cols)
       )
    
    if args['labevents']:
        df=df.merge(labevents, on = index_cols, suffixes=('[chart]', '[lab]'))
        addition_col = list(set(col.split("[")[0] for col in df.columns if col not in ['subject_id', "hadm_id", "anchor_age", "icd_code", "anchor_year_group"]))
        addition_col_type=[args["column_info"][key] for key in addition_col]
        args["column_info"].update({f"{addition_col[i]}[lab]":addition_col_type[i] for i in range(len(addition_col_type))})
        args["column_info"].update({f"{addition_col[i]}[chart]":addition_col_type[i] for i in range(len(addition_col_type))})

    print(f'Dataset shape: ', df.shape, end=' -> ')
    merge_data = df.loc[:, df.isnull().sum()<int(0.9*df.shape[0])] # nan이 90%이상인 feature 제외
    selected_feature_name = exist_feature_in_domains(merge_data, args['domain'])    
    merge_data = merge_data[selected_feature_name]
    print(merge_data.shape, end=' -> ')
    
    # 같은 subject_id이면 mortality가 같을 것이다
    merge_data = merge_data.groupby('subject_id').apply(lambda group: fill_mortality(args['target'], group)).reset_index(drop=True)
    merge_data=merge_data.drop_duplicates(ignore_index=True) # row의 모든 column값이 같은 것 제거
    print(merge_data.shape)  
    
    merge_data["unique_id"] = merge_data[index_cols].apply(lambda row: '_'.join(row.astype(str)), axis=1) ## subject_id, hadm_id 합치기
    merge_data.drop(columns=index_cols, inplace=True)
    args['category'] = [key for key, value in args["column_info"].items() if value == 'categorical']
    args['number'] = [key for key, value in args["column_info"].items() if value == 'numerical']
    
    index_cols = ['unique_id', 'icd_code', 'anchor_year_group', args['target']]
    args["selected_feature_name"]=sorted([col for col in merge_data.columns if col not in index_cols])
    args['column_info']={col:args['column_info'][col] for col in merge_data.columns if col not in index_cols}
    args['cat_num']=sum(value == 'categorical' for value in args['column_info'].values()) # categorical 변수 개수
    args['vocab_sizes'] = merge_data[[col for col in merge_data.columns if col not in index_cols and args['column_info'][col]=="categorical"]].max().tolist()
    
    total_df=[]
    c=[0]*args["n_classes"]
    
    for idx, _ in enumerate(args['domain']):
        
        domain = merge_data.loc[merge_data['icd_code']==idx, :]
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
    
    if not os.path.isfile(f"{save_root}/unique_ICD({','.join(domain_group)}).csv"): # ({','.join(domain_group)})
        sheet_name=domain_group # ['dementia', 'myocardial_infarction', 'COPD', 'asthma', 'diabetes', 'hypertension', 'stroke', 'obesity', 'dyslipidemia', 'cancer'] 
                    # 치매, 심근경색, 만성폐쇄성 폐질환, 천식, 당뇨병, 고혈압, 뇌졸중, 비만, 이상지질혈증, 암 
        
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
    print("Include subject over 18 and below 89 ", end="")
    df=df.astype({'anchor_age':int})
    df=df[(age[0]<=df["anchor_age"])] # & (df["anchor_age"]<=age[1])]
    print(df[['subject_id', 'hadm_id']].drop_duplicates().shape[0])
    return df

def over_height_over_weight(df):
    print(f"Include subject height (50~250cm) and weight (40~300kg)", end=" ")
    # ## 단위를 고려하면 743 row로 줄어듦 -> 90 subject   HEIGHT 2.54, WEIGHT 2.2
    df=df[(50.0 <= df["Height (cm)"]) & (df["Height (cm)"]<= 250.0) & 
          (40.0 <= df["Admission Weight (Kg)"]) & (df["Admission Weight (Kg)"]<= 300.0)] 
    df=df.groupby(['subject_id', 'hadm_id' , 'anchor_year_group']).mean().reset_index() 
    print(df.shape[0])
    return df

def over_24_icustay(df, icustay_day):
    print("Include subject icu 24hr stay ", end="")
    df=df[df["los"]>=icustay_day] # los=1은 1day
    print(df[['subject_id', 'hadm_id']].drop_duplicates().shape[0])
    return df

def fill_mortality(target, group):
    if group[target].isnull().all():
        return group
    group[target] = group[target].fillna(group[target].mode().sort_values().iloc[0])
    return group
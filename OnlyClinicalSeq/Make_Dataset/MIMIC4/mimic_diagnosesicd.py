import pandas as pd
import gzip
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거


def diagnoses_icd(hosp_path, save_path):
    print(f"Merge dataset for diagnoses icd", end=" ")
    start=datetime.now()
    
    icd=pd.read_csv(gzip.open(f"{hosp_path}/diagnoses_icd.csv.gz"), low_memory = False)
    d_icd=pd.read_csv(gzip.open(f"{hosp_path}/d_icd_diagnoses.csv.gz"), low_memory = False)

    # icd=icd[icd['seq_num']<=5] # 8, 10 -> # preprocess 파일에서 진행
    df=pd.merge(icd[['subject_id','hadm_id', 'seq_num', 'icd_code']], d_icd[['icd_code', "long_title"]], on='icd_code', how='left')
    df.to_csv(f"{save_path}/icd.csv", index=False) 
    print(f'{datetime.now()-start} (hh:mm:ss.ms)')   
    del icd, d_icd
    
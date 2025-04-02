import pandas as pd
import os

path="/opt/workspace/data/MIMIC/250318"
seed=2024
imputatation = 'simpleimputer_zero'

final_csv = f'{seed}_{imputatation}'
if not os.path.isdir(f"{path}/Total_Performance"):
    os.makedirs(f"{path}/Total_Performance")
    # os.makedirs(f"{path}/Total_Performance/figure")

metrics = ["rmse", "mape", "mae"]
    
# 함수 정의: 데이터 로드 및 병합
def load_results(file_path, params):
    try:
        df = pd.read_csv(file_path, index_col=[0]).reset_index()
        for key, value in params.items():
            df[key] = value
        return df
    except FileNotFoundError:
        print(f"파일 없음: {file_path}")
        return pd.DataFrame([])


target = "HbA1c"
print(f"{'*'*20} {target} {'*'*20}")
root_folder = f"{path}/{target}/"
subfolders =  sorted([f for f in os.listdir(root_folder)
                if os.path.isdir(os.path.join(root_folder, f))])

total_df = pd.DataFrame([])

for subfolder in subfolders:
    print(subfolder)
    root_folder2 = f"{root_folder}/{subfolder}/Results/"
    
    
    results = [f for f in os.listdir(root_folder2) if ".csv" in f]
    m = subfolder.split("_")
    
    for result in results:
        df = pd.read_csv(f"{root_folder2}/{result}")
        
        df[["step", "batch_size", "lr", "lstm_hidden_unit_factor"]]=m  
        df['metric'] = result.split("_")[0] 
        total_df = pd.concat([total_df, df])
                 
    
output_path = f"{path}/Total_Performance/{final_csv}.xlsx"
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    df_sorted = total_df.sort_values(by='Unnamed: 0')
    df_sorted.to_excel(writer, sheet_name=f"{target}", index=False)
    
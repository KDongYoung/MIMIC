import numpy as np 
import pandas as pd
import os
import warnings
import datetime
import re
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거


class CHA_INFERTILITY_Dataset():
    def __init__(self, args) -> None:
        self.args=args
        
        if not os.path.isdir(f"{args['save_data']}"):
            os.makedirs(args['save_data'])
        
        print("Load Start!")
        with open(self.args['total_path'] + '/args.txt', 'a') as f:
            f.write('Make Dataset Start: '+ datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
        self.dataset=self.load()
        self.dataset.to_csv(f"{args['save_data']}/cha_infertility.csv", index=False, encoding='utf-8-sig')
        self.args["name"] = [n.split(".")[0] for n in os.listdir(self.args['save_data']) if "args" not in n and ".csv" in n]
        
        with open(self.args['save_data'] + '/args.txt', 'w') as f:
            f.write(datetime.datetime.now().strftime('%Y%m%d'))
            if self.args['name'] != []:
                f.write("\n")
                f.write("\n".join(self.args["name"]))
            
        with open(self.args['total_path'] + '/made_date.txt', 'a') as f:
            f.write('Load End: '+ datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("Make Dataset End!")  
        
    def load(self):
        ## Load dataset
        data_after = self.change_column_name_value("infertility_250108")   
        data_before = self.change_column_name_value("infertility_241217")                                                                                     
    
        data = pd.concat([data_before, data_after], axis=0, ignore_index=True)      

        data = self.change_info(data) 
        data.drop(columns=['부인환자번호', '부인명', '남편환자번호', '남편명'], inplace=True)
        return data.drop_duplicates()
    
    def change_column_name_value(self, name):
        data = pd.read_excel(f"{self.args['data_root']}/{name}.xlsx", skiprows=[0], dtype=str)
        cleaned_columns = [
                            "" if "Unnamed" in str(col) else re.sub(r'\.\d+', '', str(col))
                            for col in data.columns
                        ]
        data.columns = [
                        f"{data.iloc[0, idx].rstrip()}//{cleaned_columns[idx]}" if cleaned_columns[idx] else data.iloc[0, idx]
                        for idx in range(len(data.columns))
                    ]                                
        # data.columns = data.apply(lambda col: f"{col.loc[1].rstrip()}" if pd.isna(col.loc[0]) else f"{col.loc[1].rstrip()}//{col.loc[0]}", axis=0).tolist()
        data = data.iloc[1:, :].reset_index(drop=True)
        data['SUBJNO'] = data[['부인환자번호', '남편환자번호']].astype(str).apply(lambda x: '_'.join(x), axis=1)
        data = data.apply(lambda col: col.astype(str).str.replace(r'\.\.', '.', regex=True))                                                                                            
        data[['Height//차병원 검사', 'BMI//차병원 검사', 'AMH//cycle직전 가장최근 수치', 'FSH//cycle직전 가장최근 수치']] = (
                                                                                          data[['Height//차병원 검사', 'BMI//차병원 검사', 'AMH//cycle직전 가장최근 수치', 'FSH//cycle직전 가장최근 수치']]
                                                                                          .apply(lambda col: col.astype(str).replace(r'^<([\d.]+)$',  r'\1', regex=True)
                                                                                                                            .replace(r'^([\d.]+) \([\w\s\d\-.]+\)$', r'\1', regex=True))
                                                                                        .astype(float)
                                                                                        )    
        # data.drop(columns=['부인환자번호', '부인명', '남편환자번호', '남편명'], inplace=True)
        return data
    
    def change_info(self, data):
        """ BMI 이상 """
        bmi=pd.read_excel(self.args["data_root"]+"data_error.xlsx", sheet_name='bmi')
        bmi.columns = ['부인환자번호', '부인명', '남편환자번호', '남편명', 'new_Height', 'new_Weight', 'Height//차병원 검사', 'Weight//차병원 검사', 'BMI//차병원 검사']   
        for _, row in bmi.iterrows():
            sbjno = f"{str(row['부인환자번호']).rstrip('.0')}_{str(row['남편환자번호']).rstrip('.0') if pd.notna(row['남편환자번호']) else str(row['남편환자번호'])}"
            # f"{str(int(row['부인환자번호']))}_{str(int(row['남편환자번호']))}"      
            data.loc[data['SUBJNO']==sbjno, 'Height//차병원 검사'] = row['new_Height']
            data.loc[data['SUBJNO']==sbjno, 'Weight//차병원 검사'] = row['new_Weight']
            data.loc[data['SUBJNO']==sbjno, 'BMI//차병원 검사'] = round(row['new_Weight'] / ((row['new_Height']/100) ** 2), 2)

                
        """ FSH 이상 """
        fsh=pd.read_excel(self.args["data_root"]+"data_error.xlsx", sheet_name='fsh')
        fsh.columns = ['부인환자번호', '부인명', '남편환자번호', '남편명', 'new_FSH', 'FSH//cycle직전 가장최근 수치']      
        
        for _, row in fsh.iterrows():
            sbjno = f"{str(row['부인환자번호']).rstrip('.0')}_{str(row['남편환자번호']).rstrip('.0') if pd.notna(row['남편환자번호']) else str(row['남편환자번호'])}"
            # f"{str(int(row['부인환자번호']))}_{str(int(row['남편환자번호']))}"
            data.loc[data['SUBJNO']==sbjno, 'FSH//cycle직전 가장최근 수치'] = row['new_FSH']
         
         
        """ 남편 환자번호 이상 """
        w_idxs=['11967394', '12061897', '12097711']
        # 특정 부인환자번호에 대해 남편환자번호의 최빈값 찾기
        for w_idx in w_idxs:
            if w_idx not in data["부인환자번호"]:
                continue
            
            most_frequent_husband = data.loc[data["부인환자번호"] == w_idx, "남편환자번호"].mode()[0]  # 최빈값 찾기

            # 해당 부인환자번호의 남편환자번호를 최빈값으로 변경
            data.loc[data["부인환자번호"] == w_idx, "남편환자번호"] = most_frequent_husband
            data.loc[data["부인환자번호"] == w_idx, "SUBJNO"] = (
                                                                    data.loc[data["부인환자번호"] == w_idx, "부인환자번호"].astype(str) +
                                                                    "_" +
                                                                    data.loc[data["부인환자번호"] == w_idx, "남편환자번호"].astype(str)
                                                                )

            
        return data
    

        # data_after = self.change_info(data_after)
        # data_before = self.change_info(data_before)
        
        # ### 분포 확인 용
        # data_after = data_after[['SUBJNO', '시술일자', 'FSH제재'] + self.args["dr_sel"]]
        # data_before = data_before[['SUBJNO', '시술일자', 'FSH제재'] + self.args["dr_sel"]]
        
        # data_after.columns = ['SUBJNO', '시술일자', 'FSH_Type', 'AGE', 'BMI', 'AMH', 'FSH', '용량', '시술결과', '시술유형']
        # data_before.columns = ['SUBJNO', '시술일자', 'FSH_Type', 'AGE', 'BMI', 'AMH', 'FSH', '용량', '시술결과', '시술유형']
        
        # data_after["시술일자"] = pd.to_datetime(data_after["시술일자"])
        # data_after = data_after.sort_values(by=["SUBJNO",'시술일자'])
        # data_before["시술일자"] = pd.to_datetime(data_before["시술일자"])
        # data_before = data_before.sort_values(by=["SUBJNO",'시술일자'])
    
        # ## 각자 저장
        # data_after.to_csv(f"data_after_selected.csv", index=False, encoding='utf-8-sig')
        # data_before.to_csv(f"data_before_selected.csv", index=False, encoding='utf-8-sig')
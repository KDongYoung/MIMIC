import numpy as np
import pandas as pd
import re
# from Utils.Preprocess.Feature_selection import Feature_Selection
from Utils.Preprocess.Normalization import Normalize
from Utils.Preprocess.Imputation import Imputation
from collections import Counter

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

   
def main(args, dataset):

    # 5번의 rekovelle => 용량x15로 환산
    
    data = dataset[['SUBJNO', '시술일자', 'FSH제재'] + args["dr_sel"]] # 'BMI_CATEGORY', 'AGE_CATEGORY',
    data.columns = ['SUBJNO', '시술일자', 'FSH_Type', 'AGE', 'BMI', 'AMH', 'FSH', '용량', '시술결과', '시술유형']

    data["시술일자"] = pd.to_datetime(data["시술일자"])
    data = data.sort_values(by='시술일자')
    
    ## 시술 IVF인데 용량 값이 없는 경우 제거
    data = data[((data['시술유형'] == 'IVF') &  (data['용량'].notnull())) | (data['시술유형'] == 'thET')]
    
    # 1. IVF와 thET가 모두 존재하는 SUBJNO 선별
    ivf_subjno = set(data[data['시술유형'] == 'IVF']['SUBJNO'])
    thet_subjno = set(data[data['시술유형'] == 'thET']['SUBJNO'])
    common_subjno = ivf_subjno.intersection(thet_subjno)
    data = data[(data['SUBJNO'].isin(common_subjno))].sort_values(["SUBJNO", "시술일자"])

    # 패턴을 저장할 리스트
    matching_patients = []
    
    # 각 환자별로 그룹핑하여 패턴 확인
    for patient, group in data.groupby("SUBJNO"):
        procedures = group["시술유형"].tolist()  # 시술 목록 리스트

        # IVF → thET 패턴 찾기
        pattern = ["IVF", "thET"]
        count = 0  # 패턴 반복 횟수

        # 리스트에서 특정 패턴이 몇 번 등장하는지 확인
        i = 0
        while i < len(procedures) - 1:
            if procedures[i:i+2] == pattern:
                count += 1
                i += 2  # 패턴이 발견되면 2칸 점프 (연속된 IVF-thET 방지)
            else:
                i += 1  # 패턴이 없으면 1칸 이동

        # 패턴이 2번 이상 등장하면 저장
        if count >= 1:
            matching_patients.append(patient)
            
    result = data[data["SUBJNO"].isin(matching_patients)]
    
    # 각 SUBJNO별 thET 이후 데이터 제거
    result = result.groupby('SUBJNO').apply(lambda group: group.loc[:group[group['시술유형'] == 'thET'].index.max()]).reset_index(drop=True)
    
    
    # SUBJNO 중 시술결과가 임신(1)인 SUBJNO 선별
    thet_succes_sbj = result.loc[(result['시술유형'] == 'thET') & (result['시술결과'] == '1'), 'SUBJNO'].unique()
    
    # 임신 성공한 경우 포기 약물 용량
    success = ( 
               result[result['SUBJNO'].isin(thet_succes_sbj)]
                   .groupby('SUBJNO')  # SUBJNO 기준 그룹화
                   .apply(lambda group: group[
                    (group['시술유형'] == 'IVF') &  # IVF 시술만 선택
                    (group['시술일자'] < group.loc[(result['시술유형'] == 'thET') & (result['시술결과'] == '1'), '시술일자'].min())  # thET 이전 데이터만
                ].nlargest(1, '시술일자')  # 가장 최신 시술일자 선택
                )).reset_index(drop=True)
    
    
    fail = (
            result[result['SUBJNO'].isin(thet_succes_sbj)]
                .groupby('SUBJNO')  # SUBJNO 기준 그룹화
                .apply(lambda group: 
                    group[
                        (group['시술유형'] == 'IVF') &  # IVF 시술만 선택
                        (group['시술일자'] < group.loc[(group['시술유형'] == 'thET') & (group['시술결과'] != '1'), '시술일자'].min())  # thET 이전 데이터만
                    ].nlargest(1, '시술일자')  # 가장 최신 시술일자 선택
                    if (group['시술유형'] == 'thET').any() and (group['시술결과'] != 1).all()  # 모든 시술 결과가 1이 아닌 경우만 선택
                    else pd.DataFrame()  # 조건이 맞지 않으면 빈 DataFrame 반환
                )
        ).reset_index(drop=True)
    

    """
    ###################### 
    여기 이상함!!!!!
    ###################### 
    """
    
    selected_ivf_rows = []
    for subjno, group in result.groupby('SUBJNO'):
        previous_ivf_for_success = None
        group_has_success = False

        for i in range(len(group)):
            if group.iloc[i]["시술유형"] == "IVF":
                previous_ivf_for_success = group.iloc[i]  # IVF가 나오면 저장

            elif group.iloc[i]["시술유형"] == "thET":
                if not group_has_success and (group.iloc[i]["시술결과"] == '1'):  # 시술 결과가 1이면, 해당 IVF 선택
                    previous_ivf_for_success["시술결과"]=1
                    selected_ivf_rows.append(previous_ivf_for_success)
                    group_has_success = True  # 현재 IVF-THET 그룹에서 성공 기록

                elif group_has_success and group.iloc[i]["시술결과"] != '1':
                    # 이전 그룹에서 성공이 있었고, 현재 그룹에서 성공이 없는 경우 다음 IVF 선택
                    if i + 1 < len(group) and group.loc[i + 1, "시술유형"] == "IVF":
                        previous_ivf = i + 1  # 새로운 IVF 선택
                        group_has_success = False  # 새로운 그룹 시작
                    previous_ivf_for_success["시술결과"]=0
                    selected_ivf_rows.append(previous_ivf_for_success)
                        
                # elif group_has_success and i+1 <len(group) and group.iloc[i+1]["시술유형"] == "IVF":
                    # group_has_success = False  # 새로운 IVF-THET 그룹 시작
    
    result = result.loc[selected_ivf_rows]
    
    result['용량'] = result['용량'].astype(str).str.replace('//', '/')
    result['용량'] = result['용량'].apply(process_value) ## 용량//프로토콜 변수에서 초기 용량 뽑아내기 
    result.loc[result['FSH_Type']=='5', '용량'] *= 15     # FSH제재 5 수정
    result.rename(columns={'용량': args['target'], "시술결과": "result"}, inplace=True)
    
    result.drop(columns=['FSH_Type', "시술일자", "시술유형"], inplace=True)
    result = result.reset_index(drop=True)
    print(f"data shape before: {result.shape}", end=' // ') 

    ## classification용
    result, args['n_classes'] = class_division(result, args['target'])
    
    # ## target은 다른데 값은 다른 column들은 같은 data 찾기
    # feature_columns = [col for col in result.columns if col not in ["volume", 'result']]
    # conflicting_data = result.duplicated(subset=feature_columns, keep=False)
    # result[conflicting_data].sort_values(["SUBJNO"]).to_csv("conflict_data2.csv", index=False)
    
    """  현재 no categorical feature  """
    number = [col for col in result.columns if col not in ['SUBJNO', args['target'], 'result']]
    ## imputation 
    result = Imputation(result, args['imputation'], [], number, args['target'], args['seed'])
        
    ## 수치형 데이터 NORMALIZATION
    result[number] = Normalize(args['normalize'], result[number])
    result[number] = result[number].astype(float) 
    
    args["selected_feature_name"] = number

    
    return result





# 사용자 정의 함수: "/" 또는 "*" 포함 시 분리하고, 그렇지 않으면 그대로 반환
def process_value(value):
    if "/" in value:
        value = value.split("/")[0]
    elif "*" in value:
        value = value.split("*")[0]
    
    extracted_value = re.search(r'(\d+\.\d+|\d+)', str(value))
    
    return float(extracted_value.group()) if extracted_value else None  # NaN 처리
    # else:
    #     return value

# 정규 표현식을 사용하여 '*숫자' 또는 '*숫자일'에서 숫자를 추출
def extract_periods_replace(row):
    periods = re.findall(r'\*(\d)', row['용량//프로토콜'])
    if len(periods) == 3:
        row['기간(d)//프로토콜'] = f"{row['기간(d)//프로토콜']}({periods[0]}/{periods[1]}/{periods[2]})"
    elif len(periods) == 2:
        row['기간(d)//프로토콜'] = f"{row['기간(d)//프로토콜']}({periods[0]}/{periods[1]})"
    
    return row

def class_division(df, target):
    # print(df)

    n_class=5
    
    """
    해당 값에 해당하는 데이터만 선별
    150, 200, 225, 250, 300
    """
    df = df[df[target].isin([150, 200, 225, 250, 300])]
    
    """
    가까이 있는 것으로 치환
    150, 200, 225, 250, 300
    """
    # df[target] = df[target].apply(closest_target)
    
    value_mapping = {150: 0, 200: 1, 225: 2, 250: 3, 300: 4}
    df[target] = df[target].replace(value_mapping) # `replace()`를 사용하여 한 번에 변환

    return df, n_class

def closest_target(value):
    target_values = np.array([150, 200, 225, 250, 300])
    return target_values[np.argmin(np.abs(target_values - value))]
   
   
   
    ## 규칙
    ## 정상 bmi  35~40 -> 225
    ##             ~35 -> 150
    ##           40~   -> 300 
    
    # ## 1. BMI 그룹화
    # # 저체중: BMI < 18.5
    # # 정상체중: 18.5 ≤ BMI < 24.9
    # # 과체중: 25.0 ≤ BMI < 29.9
    # # 비만: 30.0 ≤ BMI < 34.9
    # # 고도비만: BMI ≥ 35.0
    # dataset['BMI_CATEGORY'] = dataset['BMI//차병원 검사'].apply(bmi_category)
    ## AGE FLAG
    # dataset['AGE_CATEGORY'] = dataset['부인연령'].apply(age_category)
    # ## BMI & AGE FLAG
    # dataset.loc[:, 'BMI_AGE_CATEGORY'] = dataset['BMI_CATEGORY'] * dataset['AGE_CATEGORY']
    
    
    
    # # 용량에 있는 용량*일 -> 용량*일 형태의 데이터에서 일들을 뽑아서 기간(d) 변수에 작성하기
    # dataset = dataset.apply(lambda row: extract_periods_replace(row), axis=1)
    
    
    # # 원본 데이터에서 dataset의 SUBJNO와 시술일자가 같은 경우 시술결과를 1로 변경 / 나머지는 0으로 변경
    # result.loc[result['시술유형'] == 'IVF', '시술결과'] = 0
    # result.loc[result.set_index(['SUBJNO', '시술일자']).index.isin(dataset.set_index(['SUBJNO', '시술일자']).index), '시술결과'] = 1
    # result = result[result['시술유형'] == 'IVF']
    
    
    # # SUBJNO 중 시술결과가 임신(1)인 SUBJNO 선별
    # thet_succes_sbj = result.loc[(result['시술유형'] == 'thET') & (result['시술결과'] == '1'), 'SUBJNO'].unique()
    
    # # 임신 성공한 경우 포기 약물 용량
    # success = ( 
    #            result[result['SUBJNO'].isin(thet_succes_sbj)]
    #                .groupby('SUBJNO')  # SUBJNO 기준 그룹화
    #                .apply(lambda group: group[
    #                 (group['시술유형'] == 'IVF') &  # IVF 시술만 선택
    #                 (group['시술일자'] < group.loc[(result['시술유형'] == 'thET') & (result['시술결과'] == '1'), '시술일자'].min())  # thET 이전 데이터만
    #             ].nlargest(1, '시술일자')  # 가장 최신 시술일자 선택
    #             )).reset_index(drop=True)
        
    # fail = ( 
    #         result[result['SUBJNO'].isin(thet_succes_sbj)]
    #             .groupby('SUBJNO')  # SUBJNO 기준 그룹화
    #             .apply(lambda group: group[
    #             (group['시술유형'] == 'IVF') &  # IVF 시술만 선택
    #             (group['시술일자'] < group.loc[(result['시술유형'] == 'thET') & (result['시술결과'] != '1'), '시술일자'].min())  # thET 이전 데이터만
    #         ].nlargest(1, '시술일자')  # 가장 최신 시술일자 선택
    #         )).reset_index(drop=True)
    # fail = (
    #         result[result['SUBJNO'].isin(thet_succes_sbj)]
    #             .groupby('SUBJNO')  # SUBJNO 기준 그룹화
    #             .apply(lambda group: 
    #                 group[
    #                     (group['시술유형'] == 'IVF') &  # IVF 시술만 선택
    #                     (group['시술일자'] < group.loc[(group['시술유형'] == 'thET') & (group['시술결과'] != '1'), '시술일자'].min())  # thET 이전 데이터만
    #                 ].nlargest(1, '시술일자')  # 가장 최신 시술일자 선택
    #                 if (group['시술유형'] == 'thET').any() and (group['시술결과'] != 1).all()  # 모든 시술 결과가 1이 아닌 경우만 선택
    #                 else pd.DataFrame()  # 조건이 맞지 않으면 빈 DataFrame 반환
    #             )
    #     ).reset_index(drop=True)

    
    # # fail 데이터에서 success 데이터에 없는 행 찾기 (success에 없는 fail 데이터)
    # fail_not_in_success = fail.merge(success, on=list(fail.columns), how="left", indicator=True)
    # fail_not_in_success = fail_not_in_success[fail_not_in_success["_merge"] == "left_only"].drop(columns=["_merge"])

    # success['시술결과'] = 1
    # fail_not_in_success['시술결과'] = 0
    # result = pd.concat([success, fail_not_in_success])
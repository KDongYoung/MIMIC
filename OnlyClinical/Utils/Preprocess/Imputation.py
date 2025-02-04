import numpy as np
import pandas as pd

def Imputation(name, dataset, imputation, category, numeric, target, seed, column_info):
    # numeric - category 순
    # mode는 범주형 only
    # simple imputer + mice: mice는 수치형에 자주 사용
    imput_columns = [col for col in dataset.columns if col not in ['SUBJNO', 'MONTH_DIFF', target, 'subject_id', 'hadm_id', 'icd_code', 'admittime', 'dischtime', 'anchor_year_group']]
    
    cat, num = imputation.split("_")
    
    """
    #######################
    # CATEGORICAL FEATURE #
    #######################
    """
    
    if cat == "median":
        from sklearn.impute import SimpleImputer
        imputer_median = SimpleImputer(strategy='median')
        dataset[[col for col in category if col in imput_columns]] = imputer_median.fit_transform(dataset[[col for col in category if col in imput_columns]])
    elif cat == "simpleimputer":
        from sklearn.impute import SimpleImputer
        for column in category:
            if column in imput_columns:
                imputer = SimpleImputer(strategy='constant', fill_value=dataset[column].dropna().max()+1)
                dataset[column] = imputer.fit_transform(dataset[[column]])
    else: # fill 0
        dataset[[col for col in category if col in imput_columns]] = dataset[[col for col in category if col in imput_columns]].fillna(0)
    
    """
    #####################
    # NUMERICAL FEATURE #
    #####################
    """
    if name == "cha":
        float_columns = ['GSUA', 'LPUA', '3PUA', 'GSA', 'LPA', '3PA']    
        # LQSA, LQLA, NEC, HON, SES, SOL, MIX, CUP => 소수점 // 그 외는 정수
        integer_columns = [col for col in numeric if col in imput_columns and col not in float_columns]
    else:
        integer_columns = ['gender', 'anchor_age', 'Orientation']
        float_columns = [col for col in dataset.columns if col in imput_columns and col not in integer_columns]
        
    if num == "median":
        from sklearn.impute import SimpleImputer
        imputer_median = SimpleImputer(strategy='median')
        dataset[[col for col in numeric if col in imput_columns]] = imputer_median.fit_transform(dataset[[col for col in numeric if col in imput_columns]])
    elif num == "mode":
        from sklearn.impute import SimpleImputer
        imputer_mode = SimpleImputer(strategy='most_frequent')
        dataset[[col for col in numeric if col in imput_columns]] = imputer_mode.fit_transform(dataset[[col for col in numeric if col in imput_columns]])
    elif num == 'mice':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        imputer_mice = IterativeImputer(random_state=seed)    
        dataset.loc[:, float_columns] = imputer_mice.fit_transform(dataset[float_columns])                 
        dataset.loc[:, integer_columns] = np.round(imputer_mice.fit_transform(dataset[integer_columns])).astype(int)

    elif num == 'flagmedian':
        missing_flags = {
            f'{col}_missing': dataset[col].isna().astype(int)
            for col in dataset.columns
            if col in imput_columns and col not in category
        }
        dataset = pd.concat([dataset, pd.DataFrame(missing_flags)], axis=1)
        column_info.update({f'{col}_missing': "categorical" for col in dataset.columns if col in imput_columns and col not in category})
        
        imputer_median = SimpleImputer(strategy='median')
        dataset.loc[:, [col for col in dataset.columns if col in imput_columns and col not in category]] = imputer_median.fit_transform(dataset[[col for col in dataset.columns if col in imput_columns and col not in category]])                 

    elif num == 'flagmice':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
            
        missing_flags = {
            f'{col}_missing': dataset[col].isna().astype(int)
            for col in dataset.columns
            if col in imput_columns and col not in category
        }
        dataset = pd.concat([dataset, pd.DataFrame(missing_flags)], axis=1)
        column_info.update({f'{col}_missing': "categorical" for col in dataset.columns if col in imput_columns and col not in category})

        imputer_mice = IterativeImputer(random_state=seed)    
        dataset.loc[:, float_columns] = imputer_mice.fit_transform(dataset[float_columns])                 
        dataset.loc[:, integer_columns] = np.round(imputer_mice.fit_transform(dataset[integer_columns])).astype(int)
    
    elif num == 'flagzero':
        missing_flags = {
            f'{col}_missing': dataset[col].isna().astype(int)
            for col in dataset.columns
            if col in imput_columns and col not in category
        }
        dataset = pd.concat([dataset, pd.DataFrame(missing_flags)], axis=1)
        column_info.update({f'{col}_missing': "categorical" for col in dataset.columns if col in imput_columns and col not in category})

        dataset[[col for col in numeric if col in imput_columns]] = dataset[[[col for col in numeric if col in imput_columns]]].fillna(0)
    
    else: # fill 0
        dataset[[col for col in numeric if col in imput_columns]] = dataset[[col for col in numeric if col in imput_columns]].fillna(0)
             
    return dataset, column_info
import numpy as np
import pandas as pd

def Imputation(dataset, category, numeric, target):
    # numeric - category 순
    # mode는 범주형 only
    # simple imputer + mice: mice는 수치형에 자주 사용
    imput_columns = [col for col in dataset.columns if col not in ['unique_id', target]]
    
    """
    #######################
    # CATEGORICAL FEATURE #
    #######################
    """
    from sklearn.impute import SimpleImputer
    for column in category:
        if column in imput_columns:
            imputer = SimpleImputer(strategy='constant', fill_value=dataset[column].dropna().max()+1)
            dataset[column] = imputer.fit_transform(dataset[[column]])
    
    """
    #####################
    # NUMERICAL FEATURE #
    #####################
    """
    dataset[[col for col in numeric if col in imput_columns]] = dataset[[col for col in numeric if col in imput_columns]].fillna(0)
             
    return dataset
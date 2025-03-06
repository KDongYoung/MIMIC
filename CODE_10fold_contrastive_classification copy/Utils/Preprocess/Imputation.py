import numpy as np
import pandas as pd

def Imputation(dataset, imputation, category, numeric, target, seed):
    # numeric - category 순
    # mode는 범주형 only
    # simple imputer + mice: mice는 수치형에 자주 사용
    imput_columns = [col for col in dataset.columns if col not in ['SUBJNO', target]]
    
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
    if num == "median":
        from sklearn.impute import SimpleImputer
        imputer_median = SimpleImputer(strategy='median')
        dataset[[col for col in numeric if col in imput_columns]] = imputer_median.fit_transform(dataset[[col for col in numeric if col in imput_columns]])
    elif num == "mode":
        from sklearn.impute import SimpleImputer
        imputer_mode = SimpleImputer(strategy='most_frequent')
        dataset[[col for col in numeric if col in imput_columns]] = imputer_mode.fit_transform(dataset[[col for col in numeric if col in imput_columns]])
    elif num == "mean":
        from sklearn.impute import SimpleImputer
        imputer_mean = SimpleImputer(strategy='mean')
        dataset[[col for col in numeric if col in imput_columns]] = imputer_mean.fit_transform(dataset[[col for col in numeric if col in imput_columns]])
    elif num == 'mice':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        imputer_mice = IterativeImputer(random_state=seed)    
        dataset.loc[:, float_columns] = imputer_mice.fit_transform(dataset[float_columns])                 
        dataset.loc[:, integer_columns] = np.round(imputer_mice.fit_transform(dataset[integer_columns])).astype(int)

    else: # fill 0
        dataset[[col for col in numeric if col in imput_columns]] = dataset[[col for col in numeric if col in imput_columns]].fillna(0)
             
    return dataset
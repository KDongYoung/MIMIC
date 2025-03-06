import pandas as pd

def Feature_Selection(name, dataset, target, seed):
    
    X,Y = dataset.loc[:, [col for col in sorted(dataset.columns) if col not in ['SUBJNO', target, "icd_code"]]], dataset.loc[:, target]
    
    if name=="lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis(n_components=X.shape[1]//2)
        lda_fitted = lda.fit(X, Y)

        print(f'{lda_fitted.coef_=}') # LDA의 계수
        print(f'{lda_fitted.explained_variance_ratio_=}') # LDA의 분산에 대한 설명력

        X = lda_fitted.transform(X)
        
    elif name=="corrcoef":  # 필터 기법
        pass
    elif name=="rfecv": 
        from sklearn.feature_selection import RFECV
        from xgboost import XGBClassifier
        xgb=XGBClassifier(n_estimators=20,random_state=seed)
        col=X.columns
        # RFE+CV(Cross Validation), 10개의 폴드, 5개씩 제거
        rfe_cv = RFECV(estimator=xgb, step=2, cv=8, scoring="accuracy") 
        rfe_cv.fit(X, Y)
        
        X = rfe_cv.transform(X) # rank가 1인 피쳐들만 선택, rfe_cv.ranking_=1
        feature=[col[i] for i in rfe_cv.get_support(indices=True)]
        print(F"{len(feature)} Selected feature by RFECV: {feature}") # 선택된 피쳐들의 이름
        
    elif name=="rfe":
        from sklearn.feature_selection import RFE
        from xgboost import XGBClassifier

        xgb = XGBClassifier(n_estimators=50, random_state=seed)
        col = X.columns

        # RFE, 5개씩 제거
        rfe = RFE(estimator=xgb, step=5, n_features_to_select=X.shape[1]//2)  # 원하는 피처의 개수를 설정합니다 (예: 5개)
        rfe.fit(X, Y)

        # 선택된 피쳐로 X를 변환
        X = rfe.transform(X)  # 선택된 피처만 선택

        # 선택된 피처 이름 출력
        feature = [col[i] for i in rfe.get_support(indices=True)]
        print(F"{len(feature)} Selected feature by RFE: {feature}")

        # 모델에서 변수 중요도 추출 (전체 피처에 대한 중요도)
        xgb.fit(X, Y)  # 전체 데이터로 다시 학습
        importances = xgb.feature_importances_

        # 중요도 시각화
        importance_df = pd.DataFrame({
            'Feature': feature,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        print(importance_df)
        
        
    if name=='':
        return pd.concat([pd.concat([dataset['SUBJNO'], X], axis=1), Y], axis=1)
    else:
        return pd.concat([pd.concat([dataset['SUBJNO'], pd.DataFrame(X, columns=feature)], axis=1), Y.reset_index(drop=True)], axis=1)
    

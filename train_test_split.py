"""
Splitting data into training and testing
takes inputs 
"""
import feature_building as fb
import pandas as pd
import numpy as np

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error #for evaluating model performance



def train_test_split(input_df, test_years, doy_cut, test_col_1, test_col_2, test_col_3, test_col_4, feature_importance):

    importance= []
    df = input_df.copy() # sets the same year CDL dataframe


    print(f"\n===== DOY {doy_cut} =====")

    r2_pred = []


    for yr in test_years:

        # ------------------------------------------------
        # SAME-YEAR CDL MODEL
        # ------------------------------------------------
        train = df[df[test_col_1] < yr].copy() # train on all previous years
        test = df[df[df[test_col_2] == yr]].copy() # test on current year only (strictly OOS)

        X_train = fb.build_features(train, doy_cut) # builds features for same-year CDL model using the build_features function from feature_building.py
        X_test = fb.build_features(test, doy_cut) # builds features for same-year CDL model using the build_features function from feature_building.py

        if not X_train.empty and not X_test.empty: # checks if the training and testing feature sets are not empty before proceeding with model training and evaluation

            X_train[test_col_2] = train.loc[X_train.index, test_col_2] # adds the historical 5-year average yield feature to the training set for the same-year CDL model
            X_test[test_col_2] = test.loc[X_test.index, test_col_2] # adds the historical 5-year average yield feature to the testing set for the same-year CDL model

            X_train[test_col_1] = train.loc[X_train.index, test_col_1] # adds the year feature to the training set for the same-year CDL model3
            X_test[test_col_1] = test.loc[X_test.index, test_col_1]  # adds the year feature to the testing set for the same-year CDL model

            X_train = X_train.fillna(X_train.median()) #
            X_test = X_test.fillna(X_train.median())

            y_train = train.loc[X_train.index, test_col_3]
            y_test = test.loc[X_test.index, test_col_4]

            model = XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

            model.fit(X_train, y_train)
            pred = model.predict(X_test.fillna(X_train.median()))
            pred_filled = np.where(np.isnan(pred), np.median(pred), pred)
            r2_pred.append((r2_score(y_test.fillna(np.median(y_test))), pred))

            if feature_importance == True   :
                feature_importance(model, X_train, doy_cut, yr, importance)
            
    return model, pred, r2_pred, importance

def feature_importance(model, X_train, doy_cut, yr, importance):
            importance.append({
                "feature": X_train.columns,
                "importance": model.feature_importances_,
                "DOY": doy_cut,
                "year": yr,
                "dataset": "Same-Year CDL"
            })


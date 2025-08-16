#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 07:34:34 2025

@author: shubhrajit
"""

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

df=pd.read_csv("house_price_trim2.csv")
#print(df.info())

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index, test_index in split.split(df, df["Title(BHK)"]):
    strat_train_set=df.loc[train_index]
    strat_test_set=df.loc[test_index]
    
#we will work on the copy of training data
df=strat_train_set.copy()

#seperate features and labels
housing_labels=df["Price"].copy()
housing_features=df.drop("Price",axis=1)

#print(housing_labels,housing_features)

num_attribs=housing_features.drop(["location","Transaction","Furnishing","facing","overlooking","Parking Type"],axis=1,errors="Ignore").columns.tolist()
cat_attribs=["location","Transaction","Furnishing","facing","overlooking","Parking Type"]
#print(num_attribs,cat_attribs)

#lets make pipeline
#for numerical cols
num_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
#for catagorical cols
cat_pipeline=Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])
#for full-pipeline
full_pipeline=ColumnTransformer([
        ("num",num_pipeline,num_attribs),
        ("cat",cat_pipeline,cat_attribs)
    ])

#transform data
housing_prepared=full_pipeline.fit_transform(housing_features)
#print(housing_prepared.shape)

#Model Training
#Linear-Regression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_pred=lin_reg.predict(housing_prepared)
lin_rmses=-cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10) #cross-validation
print("Linear-Regression RMSE thru cross-validation:")
print(pd.Series(lin_rmses).describe())

#DecisionTree-Regression
tree_reg=DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)
tree_pred=tree_reg.predict(housing_prepared)
tree_rmses=-cross_val_score(tree_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10) #cross-validation
print("DecisionTree-Regression RMSE thru cross-validation:")
print(pd.Series(tree_rmses).describe())

#SupportVector Regression
sv_reg=SVR()
sv_reg.fit(housing_prepared,housing_labels)
sv_rmses=-cross_val_score(sv_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print("SupportVector-Regression RMSE thru cross-validation:")
print(pd.Series(sv_rmses).describe())

#RandomForest-Regression
rndm_frst_reg=RandomForestRegressor()
rndm_frst_reg.fit(housing_prepared,housing_labels)
rndm_frst_pred=rndm_frst_reg.predict(housing_prepared)
rndm_frst_rmses=-cross_val_score(rndm_frst_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10) #cross-validation
print("RandomForest-Regression RMSE thru cross-validation:")
print(pd.Series(rndm_frst_rmses).describe())
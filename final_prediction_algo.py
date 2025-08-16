#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 08:50:07 2025

@author: shubhrajit
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"

def build_pipeline(num_attribs,cat_attribs):
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
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df=pd.read_csv("house_price_trim2.csv")
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    for train_index, test_index in split.split(df, df["Title(BHK)"]):
        df.loc[test_index].to_csv("input.csv",index=False) #creating a test-set by taking random data from main file
        df=df.loc[train_index]
        
    #seperate features and labels
    housing_labels=df["Price"].copy()
    housing_features=df.drop("Price",axis=1)
    
    num_attribs=housing_features.drop(["location","Transaction","Furnishing","facing","overlooking","Parking Type"],axis=1,errors="Ignore").columns.tolist()
    cat_attribs=["location","Transaction","Furnishing","facing","overlooking","Parking Type"]
    
    pipeline=build_pipeline(num_attribs, cat_attribs)
    housing_prepared=pipeline.fit_transform(housing_features)
    
    model=RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)
    
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model is trained.")
else:
    #Lets do inference
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)
    
    input_data=pd.read_csv("input1.csv") #test-data after removing the price column
    transformed_input=pipeline.transform(input_data)
    predictions=model.predict(transformed_input)
    input_data["Price"]=predictions #price prediction for test-data
    
    input_data.to_csv("output.csv",index=False)
    print("Inference completed, result saved into output.csv")
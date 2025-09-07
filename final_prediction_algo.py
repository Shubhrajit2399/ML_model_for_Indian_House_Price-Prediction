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
from flask import Flask, request, render_template, jsonify #type: ignore

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
    app = Flask(__name__)
    #Lets do inference
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)
    
    @app.route('/', methods=['GET','POST'])
    def prediction():
        if request.method == 'POST':
            carpet_area = float(request.form['Carpet Area(sqft)'])
            Title_BHK = request.form['Title(BHK)']
            location = request.form['location']
            Furnishing = request.form['Furnishing']
            floor_no = int(request.form['Floor No.'])
            total_floor = int(request.form['Total Floors'])
            Transaction = request.form['Transaction']
            facing = request.form['facing']
            overlooking = request.form['overlooking']
            Bathrooms = int(request.form['Bathroom'])
            Balcony = int(request.form['Balcony'])
            ParkingSpaces = int(request.form['Parking Spaces'])
            ParkingType = request.form['Parking Type']
            input_data = pd.DataFrame([[carpet_area, Title_BHK, location, Furnishing, floor_no, total_floor, Transaction, facing, overlooking, Bathrooms, Balcony, ParkingSpaces, ParkingType]],
                                      columns=['Carpet Area(sqft)', 'Title(BHK)', 'location', 'Furnishing', 'Floor No.', 'Total Floors', 'Transaction', 'facing', 'overlooking', 'Bathroom', 'Balcony', 'Parking Spaces', 'Parking Type'])
            transformed_input=pipeline.transform(input_data)
            predictions=model.predict(transformed_input)
            return(f'Estimated Price: ₹ {predictions[0]:,.2f}')
            #return jsonify({f'Estimated Price: ₹ {predictions[0]:,.2f}'})
        return render_template('index.html')
    if __name__ == '__main__':
        app.run(debug=True)
    
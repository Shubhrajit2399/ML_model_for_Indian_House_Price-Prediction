# ML_model_for_Indian_House_Price-Prediction
Price prediction of houses from different cities of india by RandomForest Regression.

1. #Un-organized data dump:
2. We have collected a data-dump of un-organized values of houses from different cities. (zipped file)

3. #Data Cleaning and Trimming:
4. We have trimmed, filttered, cleaned the data-dump and saved the formatted data in house_price_trim2.csv (house_price_data_cleaning.ipynb)
5. The new csv contains below attributes:
6.   <class 'pandas.core.frame.DataFrame'>
RangeIndex: 48452 entries, 0 to 48451
Data columns (total 14 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   Title(BHK)         48452 non-null  int64  
 1   Price              48452 non-null  float64
 2   location           48452 non-null  object 
 3   Carpet Area(sqft)  48452 non-null  int64  
 4   Floor No.          48452 non-null  int64  
 5   Total Floors       48452 non-null  int64  
 6   Transaction        48452 non-null  object 
 7   Furnishing         48452 non-null  object 
 8   facing             48452 non-null  object 
 9   overlooking        48452 non-null  object 
 10  Bathroom           48452 non-null  int64  
 11  Balcony            48452 non-null  int64  
 12  Parking Spaces     48452 non-null  int64  
 13  Parking Type       48452 non-null  object 
dtypes: float64(1), int64(7), object(6)
memory usage: 5.2+ MB

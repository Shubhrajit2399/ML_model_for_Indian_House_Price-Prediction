# ML_model_for_Indian_House_Price-Prediction
Price prediction of houses from different cities of india by RandomForest Regression.

1. #Un-organized data dump:
2. We have collected a data-dump of un-organized values of houses from different cities. (zipped file)

3. #Data Cleaning and Trimming:
4. We have trimmed, filttered, cleaned the data-dump and saved the formatted data in house_price_trim2.csv (house_price_data_cleaning.ipynb)
5. The new csv contains below attributes:
<img width="383" height="364" alt="Screenshot 2025-08-16 at 11 16 08 AM" src="https://github.com/user-attachments/assets/48b42840-8461-437f-90f9-8d2e9c834810" />

6. #Data analysis model:
7. Next we did a data analysis of the location-wise no. of properties.(house_price_trim_data_analysis.ipynb)
<img width="1241" height="1603" alt="download" src="https://github.com/user-attachments/assets/6fdb04e5-4697-489e-985b-810de1696f17" />

8. #Proper ML Model Selection:
9. Found out root-mean squared error by randomly testing from the stratified train-set and selected the proper algorithm to train the Machine Learning Model.(model_testing_selection.py)
10. runfile('/Users/shubhrajit/Documents/indian_house_price_pred/model_testing_selection.py', wdir='/Users/shubhrajit/Documents/indian_house_price_pred')
Linear-Regression RMSE thru cross-validation:
count    1.000000e+01
mean     7.126993e+06
std      4.649531e+05
min      6.551521e+06
25%      6.738311e+06
50%      7.125904e+06
75%      7.585341e+06
max      7.664270e+06
dtype: float64
DecisionTree-Regression RMSE thru cross-validation:
count    1.000000e+01
mean     5.201624e+06
std      7.230348e+05
min      4.454622e+06
25%      4.708382e+06
50%      4.955546e+06
75%      5.415303e+06
max      6.866222e+06
dtype: float64
SupportVector-Regression RMSE thru cross-validation:
count    1.000000e+01
mean     1.751398e+07
std      3.891398e+05
min      1.701806e+07
25%      1.726872e+07
50%      1.741737e+07
75%      1.759061e+07
max      1.826959e+07
dtype: float64
RandomForest-Regression RMSE thru cross-validation:
count    1.000000e+01
mean     4.011550e+06
std      5.885034e+05
min      3.511023e+06
25%      3.580048e+06
50%      3.839476e+06
75%      4.080388e+06
max      5.232342e+06
dtype: float64
11. #Final machine learning model training and data prediction:
12. Though the SVG gave the lowest mean value in RMSE but at the time of prediction it generalized all prediction values near the hyperplane which is not appropiate, so we took RandomForest Regression model to train our model and predicted the Price for random test-set data in input1.csv. (final_prediction_algo.py)
13. <img width="1188" height="426" alt="Screenshot 2025-08-16 at 12 10 35 PM" src="https://github.com/user-attachments/assets/e7355177-1271-4afd-83e5-18895b331b7e" />

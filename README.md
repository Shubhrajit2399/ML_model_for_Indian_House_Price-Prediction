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
10. <img width="399" height="616" alt="Screenshot 2025-08-16 at 12 22 49 PM" src="https://github.com/user-attachments/assets/0224395b-706a-4f06-ad1d-636b814940b6" />

11. #Final machine learning model training and data prediction:
12. Though the SVR gave the lowest mean value in RMSE but at the time of prediction it generalized all prediction values near the hyperplane which is not appropiate, so we took RandomForest Regression model to train our model and predicted the Price for random test-set data in input1.csv. (final_prediction_algo.py)
13. input.csv -> test-set data with Price column.
14. input1.csv -> test-set data without Price column.
15. <img width="1188" height="426" alt="Screenshot 2025-08-16 at 12 10 35 PM" src="https://github.com/user-attachments/assets/e7355177-1271-4afd-83e5-18895b331b7e" />

<!-End_of_File->

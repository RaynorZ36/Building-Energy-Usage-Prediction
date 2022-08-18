Project Title: Prediction of energy usage of new building.

Description: This AirFLow DAG will perform the following tasks:

             1. This AirFlow DAG work with S3 bucket named "cap-stone-data-lake"
             2. This AirFlow DAG will start when new building information is uploaded under "test" folder in the S3 bucket mentioned above.
             3. The new building information is assumed to be included in the CSV file which has the same schema as previous year building                   energy usage data sheet.
             4. Upon receipt of new building information, three File Transform Operators will be triggered. They will clean the original new                 building information and pick up columns to be used for prediction based on building property type. 
             5. Cleaned new building information will be uploaded to folder "home", "office" and "hotel" according to property type.
             6. The DAG will then wake a Python Operator to clean and process the csv file under "train" folder in S3 bucket, which will be                 used to create predictor for different type of building.
             7. Predictor generated will be uploaded to folder "home", "office" and "hotel" according to property type.
             8. Next operators will call the predictor in the sub folder "home", "office" and "hotel" and generate prediction for energy                     usage of new building accordingly.
             9. A table in the Postgresql database will be created, and be filled with new building info plus predicted energy usage.
             
             This S3 bucket working with AirFlow DAG will have following folders(keys):
             
             1. test, train, home, office, hotel, scripts.
             2. addition python syntax will be stored in .py file and uploaded to "script" folder.
             
Installation: Copy following Python files into DAG folder under folder where AirFlow was installed:

              1. CAPSTONE_PROJECT.py
              2. User_define.py
              3. home_transformer.py, office_transformer.py, hotel_transformer.py
              
             

Usage: CAPSTONE_PROJECT.py: Contains syntax for DAG procedure and operators.
       User_define.py: Contains user defined function to be used to clean the previous year energy usage of buildings and train predictor.
       home_transformer.py, office_transformer.py, hotel_transformer.py :Contains syntax to clean and substract data for prediction based on                                                                          new building property type

        
       Sequence of Operators in the DAG:
       s3_sensor_test >> home_folder_get_test_data >> office_folder_get_test_data >> hotel_folder_get_test_data >> read_from_train_bucket >>        train_predictor >> s3_sensor_home >> predict_home_bucket >> s3_sensor_office >> predict_office_test >> s3_sensor_hotel >>                    predict_hotel_test >> create_prediction_table >> update_prediction_table
       
       Function for each operator: 
       
       1. s3_sensor_test: trigger the next operator when new file was added into S3 buckets "test" folder.                                  
       2. *_folder_get_test_data: read the newly submitted file in the "test" folder. Do the data waggling and upload selected data to                                         designated folder based on building type.
       3. read_from_train_bucket: read the csv file stored in "train" folder.
       4. train_predictor: Use the file from "train" folder and generate 3 predictor for each building type.
       5. s3_sensor_*: trigger the next operator when new file was added into S3 buckets "home" or "office" or "hotel" folder.
       6. predict_*_bucket: Use the predictor and apply on the new building information based on building type.
       7. create_prediction_table: Create "Energy Prediction" table in Postgre Database.
       8. update_prediction_table: Update the "Energy Prediction" table with the energy usage prediction result and new building info.



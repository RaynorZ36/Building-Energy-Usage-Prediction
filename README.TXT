Project Title: Prediction of energy usage of new building.

Description: 

This project aims to provide help for utility provider like Con-Edison and National Grid to determine the increase of energy usage raised by new buildings.

City keeps growing, same as the need for more energy. To prepare ahead, utility provider needs to know to what extinct should they prepare to satisfy the newly constructed buildings. 


This project utilizes AirFLow DAG to perform the following tasks:

1.	This AirFlow DAG work with S3 bucket named "cap-stone-data-lake" created with  AWS online database storage.
2.	This project will be triggered when user upload new building information to "test” folder in the S3 bucket on AWS online database.
3.	The new building information is assumed to be included in the CSV file which follow the schema as required
4.	Upon receipt of new building information CSV file, three File Transform Operators will be triggered. These operators will clean the new building CSV file stored in the S3 bucket from AWS and pick up columns to be used for prediction based on building property type. 
5.	The transformer operators will generate 3 CSV files which contain information of building needed for prediction. These 3 CSV files will be uploaded to folder "home", "office" and "hotel" respectively according to property type.
6.	Next operator from AirFlow will wake a Python Operator to clean and process the CSV file under "train" folder in S3 bucket, which contains energy usage of buildings from previous years. The Python Operator will use the previous years energy usage data as training data to create predictor for different type of buildings.
7.	Predictor generated will be uploaded to folder "home", "office" and "hotel" according to property type.
8.	Python operators will reference the predictor in the sub folder "home", "office" and "hotel" and generate prediction for energy usage of new buildings based on the property type accordingly.
9.	The last two operators from AirFlow will create a table in the Postgresql database and update the table with new building information and predicted energy usage.

This S3 bucket working with AirFlow DAG will have following folders(keys):
             
1.	“test” – contain user uploaded new building information
2.	“train” – contain energy usage of existing buildings from previous years
3.	“home”/”office”/”hotel” – contains cleaned data and predictor by property type.
4.	“scripts” – contains py file storing Python codes.

             
Installation: Copy following Python files into DAG folder under folder where AirFlow was installed:

              1. CAPSTONE_PROJECT.py
              2. User_define.py
              3. home_transformer.py, office_transformer.py, hotel_transformer.py
              
             

Usage: 
CAPSTONE_PROJECT.py: Contains syntax for DAG procedure and operators.
User_define.py: Contains user defined function to be used to clean the previous year energy usage of buildings and train predictor.
home_transformer.py, office_transformer.py, hotel_transformer.py :Contains syntax to clean and substract data for prediction based on new building property type

        
Sequence of Operators in the DAG:
s3_sensor_test >> home_folder_get_test_data >> office_folder_get_test_data >> hotel_folder_get_test_data >> read_from_train_bucket 
>>train_predictor >> s3_sensor_home >> predict_home_bucket >> s3_sensor_office >> predict_office_test >> s3_sensor_hotel >> 
predict_hotel_test >> create_prediction_table >> update_prediction_table
       
Function for each operator: 
       
1. s3_sensor_test: trigger the next operator when new file was added into S3 buckets "test" folder.                                  
2. *_folder_get_test_data: read the newly submitted file in the "test" folder. Do the data waggling and upload selected data to designated folder based on building type.
3. read_from_train_bucket: read the csv file stored in "train" folder.
4. train_predictor: Use the file from "train" folder and generate 3 predictor for each building type.
5. s3_sensor_*: trigger the next operator when new file was added into S3 buckets "home" or "office" or "hotel" folder.
6. predict_*_bucket: Use the predictor and apply on the new building information based on building type.
7. create_prediction_table: Create "Energy Prediction" table in Postgre Database.
8. update_prediction_table: Update the "Energy Prediction" table with the energy usage prediction result and new building info.



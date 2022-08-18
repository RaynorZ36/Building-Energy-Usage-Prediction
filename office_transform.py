from textblob import TextBlob
import sys
import pandas as pd

"""
1, read the csv file as data_frame.
2, clean the data frame by keep only columns to be used for home_bldg model.
3, return the csv file

"""
input_data=sys.argv[1]
output_data=sys.argv[2]

def office_transform(input_data, output_data):
    with open(input_data, 'r') as input_file:
         df_test = pd.read_csv(input_file)
    
    cl_to_use_list = ['Property Id','Property Name','City','Primary Property Type - Self Selected',
                      'Property Floor Area (Building(s)) (ft²)',
                 'Year Built','ENERGY STAR Score','ENERGY STAR Certification - Eligibility','Latitude','Longitude',
                  'Office - Computer Density (Number per 1,000 ft²)','Office - Weekly Operating Hours',
                  'Office - Worker Density (Number per 1,000 ft²)',
                  'Multifamily Housing - Maximum Number of Floors',
                  'Multifamily Housing - Total Number of Residential Living Units','Multifamily Housing - Percent That Can Be Cooled',
                  'Hotel - Room Density (Number per 1,000 ft²)','Hotel - Worker Density (Number per 1,000 ft²)',
                   'Hotel - Percent That Can Be Cooled'
                     ]

    df_test = df_test[cl_to_use_list]

    #rename columns names

    df_test = df_test.rename(columns = {'Property Floor Area (Building(s)) (ft²)':'Total Floor Area -SF',
                                              'Office - Computer Density (Number per 1,000 ft²)':'Office - Computer Density',
                                             'Primary Property Type - Self Selected':'Primary Property Type',
                                             'Hotel - Room Density (Number per 1,000 ft²)':'Hotel Room Density',
                                             'Hotel - Worker Density (Number per 1,000 ft²)':'Hotel Worker Density',
                                             'Hotel - Percent That Can Be Cooled':'Hotel Perecent Area Cooled',
                                             'Office - Worker Density (Number per 1,000 ft²)':'Office - Worker Density',
                                              'Multifamily Housing - Maximum Number of Floors':'Multi Fami Max Floors',
                                              'Multifamily Housing - Total Number of Residential Living Units':'Multi Fami Resid Units',
                                              'Multifamily Housing - Percent That Can Be Cooled':'Multi Fami Area Cooled'
                                             })
    
    cl_to_numeric = ['Total Floor Area -SF','ENERGY STAR Score',
                    'Office - Computer Density','Office - Weekly Operating Hours','Office - Worker Density',
                   'Multi Fami Max Floors','Multi Fami Resid Units','Multi Fami Area Cooled',
                   'Hotel Room Density','Hotel Worker Density','Hotel Perecent Area Cooled']

    for col in cl_to_numeric:
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
        
     
    # put pipeline for different type of building
    df_process = df_test

    #Change to be made including following:
    #1, Energy Star Score: NaN change into 50, which is national median
    #2, Make a new col = 'Distance to Central Park' using "Latitude" & 'Longitude'
    #3, Fill NaN value in Distance to Central Pak with average 
    df_process['ENERGY STAR Score']=df_process['ENERGY STAR Score'].fillna(50)
    df_process['Distance to Central Park'] = (abs(df_process['Latitude'])- 40.4712) + (abs(df_process['Longitude'])- 73.9665)**2
    distance_avg = df_process['Distance to Central Park'].mean()
    df_process['Distance to Central Park'] = df_process['Distance to Central Park'].fillna(distance_avg)
    df_process['Year to now'] = abs(df_process['Year Built'])- 2020
    
    #pick columns only office type building:
    df_office = df_process[df_process['Primary Property Type'] == 'Office']
    
    cols = [x for x in df_office.columns]
    
    #remove Y column and columns whose data has been processed
    drop_list = ['Multi Fami Max Floors','Multi Fami Resid Units','Multi Fami Area Cooled',
                 'Hotel Room Density','Hotel Worker Density','Hotel Perecent Area Cooled','Latitude','Longitude','Year Built']

    for elem in drop_list:
        cols.remove(elem)
    
    
    df_office_cleaned = df_office[cols]

    with open(output_data, 'w') as output_file:
        output_file = df_office_cleaned.to_csv(index=False)


office_transform(input_data,output_data)
print("csv test file cleaned")
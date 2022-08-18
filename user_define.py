from collections import defaultdict
import json as json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import numpy as np
import dill

def turn_notavailable(x):
    if x == 'Not Available':
        x=np.nan
    return x


class Ensemble_predictor(BaseEstimator, RegressorMixin):
    def __init__(self, raw_pred, residue_pred):
        self.raw_pred = raw_pred
        self.residue_pred = residue_pred
        
    def fit(self, X, y):
        self.raw_pred.fit(X, y)
        """
        Build a custom predictor that takes as an argument two other predictors. 
        It should use the first to fit the raw data and the second to fit the residuals of the first.
        """
        self.residue_pred.fit(X, y - self.raw_pred.predict(X))  #self.raw_pred.predict(X) is what's predicted by first predictor
        
        return self
    
    def predict(self, X):
        
        return self.raw_pred.predict(X) + self.residue_pred.predict(X)


def train_data_predictor(data_frame):
    
    cl_to_use_list = ['Property Id','City','Primary Property Type - Self Selected','Property Floor Area (Building(s)) (ft²)',
                 'Year Built','ENERGY STAR Score','ENERGY STAR Certification - Eligibility','Latitude','Longitude',
                  'Electricity Use - Grid Purchase (kBtu)',
                  'Office - Computer Density (Number per 1,000 ft²)','Office - Weekly Operating Hours',
                  'Office - Worker Density (Number per 1,000 ft²)','Multifamily Housing - Maximum Number of Floors',
                  'Multifamily Housing - Total Number of Residential Living Units','Multifamily Housing - Percent That Can Be Cooled',
                  'Hotel - Room Density (Number per 1,000 ft²)','Hotel - Worker Density (Number per 1,000 ft²)',
                  'Hotel - Percent That Can Be Cooled']

    df_2013_work = data_frame[cl_to_use_list]

    #rename columns names

    df_2013_work = df_2013_work.rename(columns = {'Property Floor Area (Building(s)) (ft²)':'Total Floor Area -SF',
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

    #turn the energy use (Y) col into kwh UNIT and remove rows that doesn't have Y value
    df_2013_work = df_2013_work[df_2013_work['Electricity Use - Grid Purchase (kBtu)'] != 'Not Available']
    df_2013_work['Electricity Use - Grid Purchase (kBtu)'] = pd.to_numeric(df_2013_work['Electricity Use - Grid Purchase (kBtu)'],                                                                                                   errors='coerce')
    df_2013_work['Electricity Use - Grid Purchase (kwh)'] = df_2013_work['Electricity Use - Grid Purchase (kBtu)']*0.000293

    #sort out cols into wording and numeric columns

    df_work = df_2013_work
    
    # to change all columns with "not available" into 'nan'
    cl_not_available = ['ENERGY STAR Score', 'Electricity Use - Grid Purchase (kwh)',
                    'Total Floor Area -SF','Office - Computer Density','Office - Weekly Operating Hours','Office - Worker Density',
                   'Multi Fami Max Floors','Multi Fami Resid Units','Multi Fami Area Cooled',
                   'Hotel Room Density','Hotel Worker Density','Hotel Perecent Area Cooled']


    for col in cl_not_available:
        df_work[col] = df_work[col].apply(turn_notavailable)
    

    # to change all columns with dtype 'object' into 'int' or 'float'
    cl_to_numeric = ['Total Floor Area -SF','ENERGY STAR Score',
                    'Office - Computer Density','Office - Weekly Operating Hours','Office - Worker Density',
                   'Multi Fami Max Floors','Multi Fami Resid Units','Multi Fami Area Cooled',
                   'Hotel Room Density','Hotel Worker Density','Hotel Perecent Area Cooled']

    for col in cl_to_numeric:
        df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
        
     
    # put pipeline for different type of building
    df_process = df_work

    """#Change to be made including following:
    #1, Energy Star Score: NaN change into 50, which is national median
    #2, Make a new col = 'Distance to Central Park' using "Latitude" & 'Longitude'
    #3, Fill NaN value in Distance to Central Pak with average"""
    
    df_process['ENERGY STAR Score']=df_process['ENERGY STAR Score'].fillna(50)
    df_process['Distance to Central Park'] = (abs(df_process['Latitude'])- 40.4712) + (abs(df_process['Longitude'])- 73.9665)**2
    distance_avg = df_process['Distance to Central Park'].mean()
    df_process['Distance to Central Park'] = df_process['Distance to Central Park'].fillna(distance_avg)

    df_process['Multi Fami Max Floors'] = df_process['Multi Fami Max Floors'].fillna(df_process['Multi Fami Max Floors'].mean())
    df_process['Multi Fami Resid Units'] = df_process['Multi Fami Resid Units'].fillna(df_process['Multi Fami Resid Units'].mean())
    df_process['Multi Fami Area Cooled'] = df_process['Multi Fami Area Cooled'].fillna(df_process['Multi Fami Area Cooled'].mean())

    df_process['Year to now'] = abs(df_process['Year Built'])- 2020


    cols = [x for x in df_process.columns]
    
    #remove Y column and columns whose data has been processed
    drop_list = ['Property Id','Electricity Use - Grid Purchase (kwh)',
            'Electricity Use - Grid Purchase (kBtu)','Latitude','Longitude','Year Built']

    for elem in drop_list:
        cols.remove(elem)
    
    #remove columns which doesn't apply to certain type of
    df_home = df_process[df_process['Primary Property Type'] == 'Multifamily Housing']
    home_cols = cols.remove('Office - Computer Density','Office - Weekly Operating Hours','Office - Worker Density',
                       'Hotel Room Density','Hotel Worker Density','Hotel Perecent Area Cooled')
    
    df_office = df_process[df_process['Primary Property Type'] == 'Office']
    office_cols = cols.remove('Multi Fami Max Floors','Multi Fami Resid Units','Multi Fami Area Cooled',
                       'Hotel Room Density','Hotel Worker Density','Hotel Perecent Area Cooled')
    
    df_hotel = df_process[df_process['Primary Property Type'] == 'Hotel']
    hotel_cols = cols.remove('Multi Fami Max Floors','Multi Fami Resid Units','Multi Fami Area Cooled',
                       'Office - Computer Density','Office - Weekly Operating Hours','Office - Worker Density')


    #separate different columns for different type of buildings
    X_home_train= df_home[home_cols]
    X_hotel_train= df_hotel[hotel_cols]
    X_office_train= df_office[office_cols]

    y_train_home= df_home['Electricity Use - Grid Purchase (kwh)']
    y_train_office= df_office['Electricity Use - Grid Purchase (kwh)']
    y_train_hotel= df_hotel['Electricity Use - Grid Purchase (kwh)']

    home_linear_col = ['Year to now','ENERGY STAR Score','Total Floor Area -SF','Multi Fami Max Floors','Multi Fami Resid Units',
                       'Multi Fami Area Cooled']
    Onehotencoder_col =['City']
    Ordinalencoder_col = ['ENERGY STAR Certification - Eligibility']
    Standscaler_col = ['Total Floor Area -SF']

    transformer_common_cl = ColumnTransformer([
                                              ('Standscaler',StandardScaler(),Standscaler_col),
                                              ('Onehotencoder',OneHotEncoder(),Onehotencoder_col),
                                              ('Ordinalencoder',OrdinalEncoder(),Ordinalencoder_col)
                                                ])


    
    
    bldg_model = Pipeline([('Transform', transformer_common_cl),
                                ('Predictor', Ensemble_predictor(Ridge(alpha = 10), 
                                  RandomForestRegressor(n_estimators = 100, max_depth=10, min_samples_leaf = 5)))
                               ])

    #train the model on different data set based on building type
    home_bldg_model = bldg_model.fit(X_home_train , y_train_home)
    
    office_bldg_model = bldg_model.fit(X_office_train , y_train_office)

    hotel_bldg_model = bldg_model.fit(X_hotel_train , y_train_hotel)
    
    
    return home_bldg_model, office_bldg_model, hotel_bldg_model
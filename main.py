import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import ColumnTransformer
import joblib


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
def build_pipeline(num_attrbs,cat_attrbs):
    # for numerical data
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy='median')),
        ("scalar",StandardScaler()),
    ])
    
    # for categorical data
    cat_pipeline = Pipeline([
        ("one_hot",OneHotEncoder(handle_unknown='ignore')),
    ])
    
    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attrbs),
        ("cat",cat_pipeline,cat_attrbs),
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    housing  = pd.read_csv('housing.csv')
    housing['income_cat'] = pd.cut(housing['median_income'],bins=[0.0,1.5,3.0,4.5,5.5,np.inf],labels=[1,2,3,4,5])
    
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index, test_index in split.split(housing,housing['income_cat']):
        housing.loc[test_index].drop('income_cat',axis = 1).to_csv('input.csv',index = False)
        housing = housing.loc[train_index].drop('income_cat',axis=1)
    # seperate labels and features 
    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value',axis=1)
    
    #seperate num_attrbs and cat_attrbs
    num_attrbs = housing_features.drop('ocean_proximity',axis=1).columns.tolist()
    cat_attrbs = ['ocean_proximity']
    
    pipeline = build_pipeline(num_attrbs,cat_attrbs)
    housing_prepared = pipeline.fit_transform(housing_features)
    
    #train the model
    
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)
    
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    
    print('model runs successfull..')
else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    input_data  = pd.read_csv('input.csv')
    transformed_data = pipeline.transform(input_data)
    prediciton = model.predict(transformed_data)
    input_data['median_house_value'] = prediciton
    
    input_data.to_csv('output.csv',index=False)
    print('inference is complete...')
    
    
    
        
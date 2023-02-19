import pandas as pd
import numpy as np 
import os,sys 
import yaml
import dill
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.config import mongo_client

#Function for getting collection as DataFrame
def get_collection_as_dataframe(database_name:str, collection_name:str )-> pd.DataFrame :
    """
    Description: This function returns collections as dataframe
    Params:
    database_name: database name
    collection_name : collection name
    return Pandas dataframe of the collection
    """
    try:
        logging.info(f"Reading data from DataBase :{database_name} and collection : {collection_name}")
        df=pd.DataFrame(list(mongodb_client[database_name][collection_name].find()))
        logging.info(f"found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping column : _id")
            df=df.drop("_id",axis=1)
        logging.info(f"Rows and Columns in df: {df.shape}")
        return df
    except Exception as e:
        raise SensorException(e,sys)

#Function for Writting yaml file
def write_yaml_file (file_path,data:dict):
    """
    Description: Writing YAML file
    params:
    file_path: file path
    data: data #in dictionary form
    """
    try:
        file_dir=os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml_dump (data,file_writer)
    except Exception as e:
        raise SensorException (e,sys)

def convert_columns_float(df:pd.DataFrame,exclude_columns:list)-> pd.DataFrame:
    try:
        for columns in df.columns:
            if column not in exclude_columns:
                df[column]=df[column].astype('float')
        return df
    except Exception as e:
        raise SensorException (e,sys)

def save_object (file_path:str, obj:object)-> None:
    try:
        logging.info("Entered Saved_object method")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open (file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exiting the Save_object method")
    except Exception as e:
        raise SensorException (e,sys)

def load_object (file_path: str,) -> object :
    try:
        if not os.path.exist(file_path):
            raise Exception(f"The File : {file_path} doesn't exists")
        with open (file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise SensorException (e,sys)

def load_numpy_array_data (file_path: str) -> np.array:
    """
    Description: Load numpy array data from file
    params:
    file_path: str loaction of file to load
    return: np.array data loaded
    """
    try:
        with open (file_path,"rb") as file_obj:
            return np.load (file_obj)
    except Exception as e:
        raise SensorException (e,sys)

def save_numpy_array_data (file_path:str, array: np.array):
    """
    save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        with open (file_path, "wb") as file_obj :
            np.save(file_obj,array)
    except Exception as e:
        raise SensorException (e,sys)
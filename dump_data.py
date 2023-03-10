import pymongo
import pandas as pd
import json

from sensor.config import mongo_client

Data_File_Path = "/config/workspace/aps_failure_training_set1.csv"
Data_base_name = "aps"
collection_name = "sensor"

if __name__ == "__main__":
    df = pd.read_csv(Data_File_Path)
    # loading dataframe in csv
    print(f"Number of Rows and Columns: {df.shape}")
    # converting dataframe to json so that we can dump these record in mongodb
    df.reset_index(drop=True, inplace=True)
    json_record = list(json.loads(df.to_json(orient='records')))
    # convert record to json
    print(json_record[0])
    # insert converted json record to mongodb
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

import pymongo
import pandas as pd
import json

client = pymongo.MongoClient("mongodb://localhost:27017/")
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
    client[Data_base_name][collection_name].insert_many(json_record)

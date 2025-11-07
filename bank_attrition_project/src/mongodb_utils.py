from pymongo import MongoClient
from src.config import MONGODB_URI, MONGODB_DATABASE, MONGODB_COLLECTION
import pandas as pd

def get_mongo_client():
    client = MongoClient(MONGODB_URI)
    return client

def save_to_mongodb(df, database_name=MONGODB_DATABASE, collection_name=MONGODB_COLLECTION):
    client = get_mongo_client()
    db = client[database_name]
    collection = db[collection_name]
    
    pandas_df = df.toPandas()
    records = pandas_df.to_dict('records')
    
    collection.delete_many({})
    collection.insert_many(records)
    
    client.close()
    print(f"Saved {len(records)} records to MongoDB")

def load_from_mongodb(database_name=MONGODB_DATABASE, collection_name=MONGODB_COLLECTION):
    client = get_mongo_client()
    db = client[database_name]
    collection = db[collection_name]
    
    records = list(collection.find({}, {'_id': 0}))
    
    client.close()
    return records

def get_collection_stats(database_name=MONGODB_DATABASE, collection_name=MONGODB_COLLECTION):
    client = get_mongo_client()
    db = client[database_name]
    collection = db[collection_name]
    
    count = collection.count_documents({})
    
    client.close()
    return {"collection": collection_name, "document_count": count}

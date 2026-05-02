import os
from dotenv import load_dotenv
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

uri = os.getenv("MONGO_URI")

parsed = urlparse(uri)
username = parsed.username

print("MongoDB Username:", username)
from config import cfg

print("ACTIVE MONGO URI:", cfg.MONGO_URI)

# Load environment variables
load_dotenv()

# Get values from .env
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "cdvae")

# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)

# Select database
db = client[MONGO_DB]

print("✅ Connected to database:", MONGO_DB)

# List collections
collections = db.list_collection_names()
print("📁 Collections:", collections)

# If collections exist, fetch data
if collections:
    collection = db[collections[0]]  # take first collection
    docs = list(collection.find().limit(5))

    print("\n📄 Sample Data:")
    for doc in docs:
        print(doc)
else:
    print("⚠️ No collections found. Database is empty.")
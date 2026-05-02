from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))

# This will "create" DB when we insert
db = client["cdvae"]

# Insert sample document
result = db["structures"].insert_one({
    "test": "database initialized",
    "status": "working"
})

print("✅ Inserted ID:", result.inserted_id)

# Verify
print("📁 Collections:", db.list_collection_names())
print("📄 Data:", list(db["structures"].find()))
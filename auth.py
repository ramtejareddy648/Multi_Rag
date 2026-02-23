from fastapi import Depends, HTTPException, FastAPI
from pydantic import BaseModel
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi
load_dotenv()



app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(
    MONGO_URI, 
    tlsCAFile=certifi.where(),
    tlsAllowInvalidCertificates=True  
)

db=mongo_client['cag_db']
users_collection = db['users']

class UserAuth(BaseModel):
    username: str
    password: str
    email: str=None
    role:str='student'


@app.get("/")
async def health_check():
    return {"status": "Backend is running!"}

@app.post("/register")
async def register(user: UserAuth):
    hash_pwd=pwd_context.hash(user.password)
    
    
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    user_doc={
        'username': user.username,
        'email': user.email,
        'password_hash': hash_pwd,
        'role': user.role
        
    }
    
    try:
        users_collection.insert_one(user_doc)
        return {"message": "User registered successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
async def login(user: UserAuth):
    try:
        
        db_user = users_collection.find_one({"username": user.username})
    
        if db_user and pwd_context.verify(user.password, db_user['password_hash']):
            return {
                "username": db_user['username'], 
                "role": db_user.get('role', 'student')
                }
        else:
            raise HTTPException(status_code=401, detail="Invalid username or password")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    




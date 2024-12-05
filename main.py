import pickle
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, status, Depends
from datetime import datetime, timedelta
import time
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
import pandas as pd
import random
from database_conn import database
from helper import authenticate_user, create_access_token, get_current_active_user
from custom_data_type import Token, User, cv_json_1
from io import BytesIO
from CV_JSON import cv_json
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse
from fastapi import Request
from azure.storage.blob import BlobServiceClient
# from config import AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME, ACCESS_TOKEN_EXPIRE_MINUTES
from dotenv import load_dotenv
import logging
import os

load_dotenv(dotenv_path=".env") 

ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")  # Fetch the connection string
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")


# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not AZURE_CONNECTION_STRING:
#     raise ValueError("AZURE_CONNECTION_STRING is not set or is empty. Check your .env file.")

# if not ACCESS_TOKEN_EXPIRE_MINUTES:
#     raise ValueError("ACCESS_TOKEN_EXPIRE_MINUTES is not set or is empty. Check your .env file.")

# if not AZURE_CONTAINER_NAME:
#     raise ValueError("AZURE_CONTAINER_NAME is not set or is empty. Check your .env file.")

# Debug prints (remove or comment out in production)
# print(f"ACCESS_TOKEN_EXPIRE_MINUTES: {ACCESS_TOKEN_EXPIRE_MINUTES}")
# print(f"AZURE_CONNECTION_STRING: {AZURE_CONNECTION_STRING[:10]}...")  # Partial for validation
# print(f"AZURE_CONTAINER_NAME: {AZURE_CONTAINER_NAME}")


try:
    # Configure logging
    logging.basicConfig(filename="error.log", level=logging.ERROR)

    load_dotenv()
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def database_connect():
        await database.connect()

    @app.on_event("shutdown")
    async def database_disconnect():
        await database.disconnect()

    # Authentication
    @app.post("/token")
    async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
        user = await authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}

    user_data = {}
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

    CSV_BLOB_NAME = "files/uploaded_files.csv"
    PICKLE_FOLDER = "files/pickle_data/"

    # Helper functions
    def upload_blob(data: bytes, blob_name: str):
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(data, overwrite=True)

    def download_blob(blob_name: str) -> bytes:
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        return blob_data

    def initialize_csv_in_azure():
        try:
            download_blob(CSV_BLOB_NAME)
        except:
            df = pd.DataFrame(columns=["user_id", "pdf_name", "doc_id", "output_file_name"])
            csv_data = df.to_csv(index=False).encode('utf-8')
            upload_blob(csv_data, CSV_BLOB_NAME)

    def save_user_data_to_pickle(user_id: str, doc_id: str):
        pickle_data = pickle.dumps(user_data[user_id])
        pickle_blob_name = f"{PICKLE_FOLDER}/{doc_id}.pkl"
        upload_blob(pickle_data, pickle_blob_name)

    def load_user_data_from_pickle(doc_id: str):
        try:
            pickle_blob_name = f"{PICKLE_FOLDER}/{doc_id}.pkl"
            pickle_data = download_blob(pickle_blob_name)
            return pickle.loads(pickle_data)
        except:
            return None

    def update_csv_in_azure(user_id: str, pdf_name: str, doc_id: str, output_file_name: str):
        try:
            csv_data = download_blob(CSV_BLOB_NAME).decode('utf-8')
            df = pd.read_csv(BytesIO(csv_data.encode()))
        except Exception as e:
            print(f"Error downloading or reading the CSV, initializing new CSV: {e}")
            df = pd.DataFrame(columns=["user_id", "pdf_name", "doc_id", "output_file_name"])
        new_row = pd.DataFrame([{"user_id": user_id, "pdf_name": pdf_name, "doc_id": doc_id, "output_file_name": output_file_name}])
        df = pd.concat([df, new_row], ignore_index=True)
        updated_csv_data = df.to_csv(index=False).encode('utf-8')
        upload_blob(updated_csv_data, CSV_BLOB_NAME)

    def validate_user_doc_id_in_azure(user_id: str, doc_id: str):
        try:
            csv_data = download_blob(CSV_BLOB_NAME).decode('utf-8')
            df = pd.read_csv(BytesIO(csv_data.encode()))
            for _, row in df.iterrows():
                if row["user_id"] == user_id and str(row["doc_id"]) == str(doc_id):
                    return True
            return False
        except Exception as e:
            print(f"Error during validation: {e}")
            return False

    def doc_id_exists_in_azure(doc_id: str):
        try:
            csv_data = download_blob(CSV_BLOB_NAME).decode('utf-8')
            df = pd.read_csv(BytesIO(csv_data.encode()))
            return not df[df["doc_id"] == doc_id].empty
        except:
            return False

    def generate_unique_doc_id():
        while True:
            doc_id = str(random.randint(100000, 999999))
            if not doc_id_exists_in_azure(doc_id):
                return doc_id

    @app.post("/upload/")
    async def upload_pdf(request: Request, userinput: cv_json_1 = Depends(), file: UploadFile = File(...), background_tasks: BackgroundTasks = None, current_user: User = Depends(get_current_active_user)):
        
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only PDF files are allowed.",
            )
        

        current_time = datetime.now()
        print("Starting time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        
        user_id = userinput.UserId
        pdf_name = file.filename
        doc_id = generate_unique_doc_id()

        output_file_name = f"{user_id}_{doc_id}_output.txt"

        if user_id not in user_data:
            user_data[user_id] = {}
        
        file_content = await file.read()

        user_data[user_id] = {
            "file_content": file_content,
            "expiry": datetime.utcnow() + timedelta(minutes=15),
            "ready": False,
            "pdf_text": None,
            "doc_id": doc_id

        }
    
        save_user_data_to_pickle(user_id, doc_id)
        update_csv_in_azure(user_id, pdf_name, doc_id, output_file_name)

        background_tasks.add_task(run_background_tasks, user_id, doc_id, file_content)

        dynamic_url = request.url_for("get_pdf", user_id=user_id, doc_id=doc_id)
        
        return {"url": dynamic_url, "document id for the pdf": doc_id, "message": "You can access this after 15 minutes."}

    async def start_timer(user_id: str, doc_id: str):
        await asyncio.sleep(15 * 60)
        user_data[user_id]['ready'] = True
        save_user_data_to_pickle(user_id, doc_id)

    def extract_pdf_content_task(user_id: str, doc_id: str, file_content: bytes):
        file_stream = BytesIO(file_content)
        extracted_text = cv_json(file_stream)
        user_data[user_id]["pdf_text"] = extracted_text
        save_user_data_to_pickle(user_id, doc_id)

    async def run_background_tasks(user_id: str, doc_id: str, file_content: bytes):
        await asyncio.gather(
            start_timer(user_id, doc_id),
            asyncio.to_thread(extract_pdf_content_task, user_id, doc_id, file_content)
        )

    @app.get("/get-pdf/{user_id}/{doc_id}")
    async def get_pdf(user_id: str, doc_id: str):

        if not validate_user_doc_id_in_azure(user_id, doc_id):
            raise HTTPException(status_code=404, detail="Invalid user ID or document ID.")

        user_info = load_user_data_from_pickle(doc_id)

        if user_info is None:
            raise HTTPException(status_code=404, detail="No data found for the given document ID.")

        if datetime.utcnow() < user_info['expiry']:
            raise HTTPException(status_code=403, detail="The file is not ready yet. Please try after 15 minutes.")
        
        if user_info['pdf_text'] is None:
            raise HTTPException(status_code=403, detail="Extraction process is not completed yet.")
        
        return user_info["pdf_text"]


except Exception as e:
    logging.error("An error occurred", exc_info=True)
    raise e



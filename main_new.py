import uvicorn
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Depends,
    status,
    Form,
    Request,
)
from pydantic import BaseModel, Field
from typing import Optional, List
import base64
import openai
import json
from datetime import datetime
import os
import fitz  # PyMuPDF
import openai
from typing import Union, TypeVar
import base64
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from fastapi.responses import JSONResponse
import uuid  # Import for unique identifiers
import pymysql
import time
from langchain.output_parsers import PydanticOutputParser
import pandas as pd
import json
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import base64
import os
import requests
import httpx
from table_operations import insertExtractedData
from table_operations import getPrescriptionList
from table_operations import getPrescription
from table_operations import editPrescription
from table_operations import deletePrescription
from table_operations import signupUser
from table_operations import updateSignInTime
from table_operations import getAccountPassword
from table_operations import addMessage
from table_operations import getUserDetails
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.security import generate_password_hash, check_password_hash

os.environ["OPENAI_API_KEY"] = (
    "sk-a07Vs0y8Wxq2CY-NuI7ztLwqWc2C8QIKM6QNfznmxNT3BlbkFJlDQ4-KJBy3aFkEPUZfOS0Mm_EeeE2oTjjQPpLGEyUA"
)

import logging


# Initialize logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Define the Pydantic models


class PatientInfo(BaseModel):
    name: Optional[str] = Field(None, alias="Patient's Name")
    guardian_name: Optional[str] = Field(None, alias="S/o | D/o | W/o")
    dob: Optional[str] = Field(None, alias="Date of Birth")
    age: Optional[int] = Field(None)
    sex: Optional[str] = Field(None)
    occupation: Optional[str] = Field(None)
    insurance_no: Optional[str] = Field(None, alias="Health Insurance No")
    healthcare_provider: Optional[str] = Field(None, alias="Health Care Provider")
    health_card_no: Optional[str] = Field(None, alias="Health Card No")
    patient_id_no: Optional[str] = Field(None, alias="Patient ID No")
    address: Optional[str] = Field(None, alias="Patient's Address")
    cell_no: Optional[str] = Field(None, alias="Cell No")


class Diagnosis(BaseModel):
    diagnosed_with: Optional[str] = Field(None, alias="Diagnosed With")
    blood_pressure: Optional[str] = Field(None, alias="Blood Pressure")
    pulse_rate: Optional[Union[str, int]] = Field(None, alias="Pulse Rate")
    weight: Optional[Union[str, int]] = Field(None, alias="Weight")
    allergies: Optional[str] = Field(None, alias="Allergies")
    disabilities: Optional[str] = Field(None, alias="Disabilities If any")


class DrugInfo(BaseModel):
    name: Optional[str]
    unit: Optional[str] = Field(None, alias="Unit (Tablet / Syrup)")
    dosage: Optional[str] = Field(None, alias="Dosage (Per Day)")


class History(BaseModel):
    brief_history: Optional[str] = Field(None, alias="Brief History of Patient")
    follow_up_physician: Optional[str] = Field(None, alias="Follow Up Physician")


class Signature(BaseModel):
    signature: Optional[str] = Field(None, alias="Signature of Physician")


class DietToFollow(BaseModel):
    diet_to_follow: Optional[str] = Field(None, alias="Diet To Follow")


class PrescriptionMedical(BaseModel):
    patient_info: PatientInfo
    diagnosis: Diagnosis
    drugs: List[DrugInfo]
    diet_to_follow: DietToFollow
    history: History
    signature: Signature


# Initialize FastAPI app
app = FastAPI()

allowed_origins = [
    "*",  # Make it functional on d-day
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Allows CORS for these origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT"],  # Allowed HTTP methods
    allow_headers=["Authorization", "Content-Type"],  # Allowed headers
)


# # Define a fixed output directory for temporary images
# output_dir = "temp_images"
# os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# # Helper function to encode image as base64
# def encode_image(image_path: str) -> str:
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")

# # Convert PDF to images
# def pdf_to_images(pdf_content: bytes, output_dir: str = output_dir) -> List[str]:
#     pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
#     image_paths = []
#     for page_num in range(pdf_document.page_count):
#         page = pdf_document.load_page(page_num)
#         pix = page.get_pixmap()
#         image_path = os.path.join(output_dir, f"page_{page_num}.png")
#         pix.save(image_path)
#         image_paths.append(image_path)
#     return image_paths

# # Helper function to determine if file is a PDF or an image
# def is_pdf(file: UploadFile) -> bool:
#     return file.content_type == "application/pdf"


# Define a fixed output directory for temporary images
output_dir = "temp_imagess"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist


# Helper function to encode image as base64
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Convert PDF to images
# Convert PDF to images with unique naming
def pdf_to_images(pdf_content: bytes, output_dir: str = output_dir) -> List[str]:
    unique_id = str(uuid.uuid4())  # Generate a unique identifier for each upload
    pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
    image_paths = []
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        # Create a unique image path for each page
        image_path = os.path.join(output_dir, f"{unique_id}_page_{page_num}.png")
        pix.save(image_path)
        image_paths.append(image_path)
        logger.info(f"Saved page {page_num} as image: {image_path}")
    return image_paths


# Helper function to determine if file is a PDF
def is_pdf(file: UploadFile) -> bool:
    return file.content_type == "application/pdf"


# Helper function to get http_client session
async def get_http_client():
    return requests.Session()


# Endpoint to upload the file (PDF or image) and get structured data
@app.post("/upload-prescription/")
async def upload_prescription(doctor_id: str = Form(...), file: UploadFile = File(...)):
    try:
        # Read the uploaded file content
        file_content = await file.read()
        logger.info(f"Received file: {file.filename}")

        # Process as PDF or image
        if is_pdf(file):
            logger.info("Processing file as PDF.")
            image_paths = pdf_to_images(file_content)
        else:
            logger.info("Processing file as a single image.")
            timestamp = int(time.time())
            image_path = os.path.join(output_dir, f"{timestamp}_{file.filename}")
            with open(image_path, "wb") as img_file:
                img_file.write(file_content)
            image_paths = [image_path]
            logger.info(f"Saved image file: {image_path}")

        # Encode each image to base64
        encoded_images = [encode_image(img_path) for img_path in image_paths]

        # Call the function to extract text from each image and compile the results
        extracted_texts = [
            extract_text_from_image(img_base64) for img_base64 in encoded_images
        ]

        await insertExtractedData(extracted_texts, doctor_id)

        return JSONResponse(
            {"msg": "success", "status_code": 200, "pages": extracted_texts}
        )

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")


class EditPrescriptionModel(BaseModel):
    data: dict


@app.post("/edit-prescription/")
async def edit_prescription(request: EditPrescriptionModel):
    try:

        if request.data["prescription_id"] and len(request.data["prescription_id"]) > 0:
            await editPrescription(request.data)
            return JSONResponse(
                content={
                    "msg": "success",
                    "status_code": 200,
                    "data": request.data,
                }
            )
        else:
            raise HTTPException(status_code=401, detail="prescription_id is mandatory")
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail="Error processing the form data.")


@app.get("/list-extracted-prescriptions")
async def get_extracted_prescriptions(doctor_id: str):
    try:

        list_of_extracted_prescriptions_json = await getPrescriptionList(doctor_id)
        return JSONResponse(
            content={
                "msg": "list of all extracted prescriptions",
                "extracted_prescriptions": list_of_extracted_prescriptions_json,
            },
            status_code=200,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-prescription")
async def get_prescription(prescription_id: str):
    try:

        prescription_json = await getPrescription(prescription_id)

        return JSONResponse(
            content={
                "msg": "successfully fetched prescription",
                "prescription": prescription_json,
            },
            status_code=200,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete-prescription/")
async def delete_prescription(prescription_id: str):
    try:
        if not prescription_id:
            raise HTTPException(status_code=400, detail="prescription_id is required")

        await deletePrescription(prescription_id)

        return JSONResponse(
            content={"msg": "Prescription deleted successfully", "status_code": 200}
        )
    except Exception as e:
        logger.error(f"Error deleting prescription: {e}")
        raise HTTPException(status_code=500, detail="Error processing the request.")


# Function to simulate the GPT-based text extraction
def extract_text_from_image(image_base64: str) -> dict:
    try:
        logger.info("Starting text extraction from image.")
        # Your logic to interact with OpenAI API here
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    """ This is an image of a medical prescription written by a doctor. Extract all text and interpret it, correcting any errors or unclear parts using medical knowledge. Ensure all medication names, dosages, and instructions are accurate.""",
                    *map(lambda x: {"image": x, "resize": 768}, [image_base64]),
                ],
            },
        ]

        params = {
            "model": "gpt-4o",  # Adjust this as needed
            "messages": PROMPT_MESSAGES,
            "max_tokens": 3500,
        }

        result = openai.chat.completions.create(**params)
        extracted_text = result.choices[0].message.content

        # Set up prompt templates
        preamble = (
            "You are a medical expert AI chatbot having a conversation with a human. Your task is to provide accurate "
            "and helpful answers based on the extracted parts of a medical health report."
        )
        postamble = "Do not include any explaination Only include th only include  actual extracted data. Discard placeholder, dummy, or irrelevant information."

        # Formatting the prompt
        chat_prompt = PromptTemplate(
            input_variables=[
                "preamble",
                "format_instructions",
                "extracted_text",
                "postamble",
            ],
            template="{preamble}\n\n{format_instructions}\n\n{extracted_text}\n\n{postamble}",
        )

        parser = PydanticOutputParser(pydantic_object=PrescriptionMedical)
        request = chat_prompt.format_prompt(
            preamble=preamble,
            format_instructions=parser.get_format_instructions(),
            extracted_text=extracted_text,
            postamble=postamble,
        ).to_messages()

        chat = ChatOpenAI(temperature=0.0)
        response = chat(request)
        parsed_output = parser.parse(response.content)
        logger.info("Text extraction and parsing completed successfully.")
        return parsed_output.model_dump()

    except Exception as e:
        logger.error(f"Error during text extraction: {e}")
        raise e


@app.post("/signup")
async def signup(data: dict):
    try:
        if not all(
            [
                data.get("name"),
                data.get("email"),
                data.get("phone"),
                data.get("password"),
            ]
        ):
            raise HTTPException(status_code=400, detail="All data is required")
        password = data["password"]
        data["hashed_password"] = generate_password_hash(password)
        data["dated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await signupUser(data)
        return JSONResponse(
            content={"msg": "User signedup successfully", "status_code": 200}
        )
    except Exception as e:
        logger.error(f"Error signing up the user: {e}")
        raise HTTPException(status_code=500, detail="Error processing the request.")


@app.post("/login")
async def login(data: dict):
    try:
        if not all([data.get("email"), data.get("password")]):
            raise HTTPException(status_code=400, detail="All data is required")
        email = data["email"]
        password = data["password"]
        stored_password = await getAccountPassword(email)
        if check_password_hash(stored_password, password):
            await updateSignInTime(email)
            user_details = await getUserDetails(email)
            (
                hospital_id,
                id,
                email_id,
                phone_number,
                password,
                registration_timestamp,
                last_login_timestamp,
                active_session_token,
                active,
                session_ends,
                name,
            ) = user_details
            # Convert timedelta to timestamp
            if user_details[5] is not None:
                registration_timestamp = (
                    user_details[5].days * 24 * 60 * 60
                ) + user_details[5].seconds
            # Create a dictionary to store the user details
            user_details_dict = {
                "hospital_id": hospital_id,
                "id": id,
                "email_id": email_id,
                "phone_number": phone_number,
                "registration_timestamp": registration_timestamp,
                "last_login_timestamp": last_login_timestamp,
                "active_session_token": active_session_token,
                "active": active,
                "session_ends": session_ends,
                "name": name,
            }
            return JSONResponse(
                content={
                    "msg": "User signedin successfully",
                    "user_details": user_details_dict,
                    "status_code": 200,
                }
            )
    except Exception as e:
        logger.error(f"Error signing in the user: {e}")
        raise HTTPException(status_code=500, detail="Error processing the request.")


@app.post("/contact")
async def contact(data: dict):
    try:
        if not all(
            [
                data.get("email"),
                data.get("name"),
                data.get("subject"),
                data.get("message"),
            ]
        ):
            raise HTTPException(status_code=400, detail="All data is required")
        await addMessage(data)
        return JSONResponse(
            content={"msg": "Message added successfully", "status_code": 200}
        )
    except Exception as e:
        logger.error(f"Error signing in the user: {e}")
        raise HTTPException(status_code=500, detail="Error processing the request.")


if __name__ == "__main__":
    logger.info("Starting FastAPI application.")
    uvicorn.run(app, host="0.0.0.0", port=8000)


# (base) akashdesai@AkashDesai-Precision-3580:~/projects/ICU_LAB_REPORTS/handwritten_report_lab_pic/ip_linux_deep$ chmod 400 deepaarogya_api.pem
# (base) akashdesai@AkashDesai-Precision-3580:~/projects/ICU_LAB_REPORTS/handwritten_report_lab_pic/ip_linux_deep$ ls
# deepaarogya_api.pem
# (base) akashdesai@AkashDesai-Precision-3580:~/projects/ICU_LAB_REPORTS/handwritten_report_lab_pic/ip_linux_deep$ ssh -i deepaarogya_api.pem ubuntu@13.232.156.45The authenticity of host '13.232.156.45 (>
# ED25519 key fingerprint is SHA256:fMHr5m5nX/5mrS8FmD1rEIaqwKonYqPq3pyTKF3d1N0.
# This key is not known by any other names


# bash
# Copy code
# sudo apt install python3 python3-pip -y
# 4. Install Virtual Environment (optional but recommended)
# It’s recommended to use a virtual environment to manage your application’s dependencies. Install the venv package and create a virtual environment:

# bash
# Copy code
# sudo apt install python3-venv -y
# python3 -m venv fastapi-env
# source fastapi-env/bin/activate

# gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000

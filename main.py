import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import base64
import openai
import json


import openai
from typing import Union,TypeVar
import base64
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.output_parsers import PydanticOutputParser
import pandas as pd
import json  
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import base64
import os
os.environ['OPENAI_API_KEY'] = 'sk-a07Vs0y8Wxq2CY-NuI7ztLwqWc2C8QIKM6QNfznmxNT3BlbkFJlDQ4-KJBy3aFkEPUZfOS0Mm_EeeE2oTjjQPpLGEyUA'

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


# Helper function to encode image as base64
def encode_image(image_file) -> str:
    return base64.b64encode(image_file).decode('utf-8')


# Endpoint to upload the image and get the structured data
# Define a fixed output directory for temporary images
output_dir = "temp_images"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Helper function to encode image as base64
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Convert PDF to images
def pdf_to_images(pdf_content: bytes, output_dir: str = output_dir) -> List[str]:
    pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
    image_paths = []
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        image_path = os.path.join(output_dir, f"page_{page_num}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    return image_paths

# Helper function to determine if file is a PDF or an image
def is_pdf(file: UploadFile) -> bool:
    return file.content_type == "application/pdf"

# Endpoint to upload the file (PDF or image) and get structured data
@app.post("/upload-prescription/")
async def upload_prescription(file: UploadFile = File(...)):
    try:
        # Read the uploaded file content
        file_content = await file.read()

        # Process as PDF or image
        if is_pdf(file):
            # Convert PDF to images
            image_paths = pdf_to_images(file_content)
        else:
            # Handle as a single image file
            image_path = os.path.join(output_dir, file.filename)
            with open(image_path, "wb") as img_file:
                img_file.write(file_content)
            image_paths = [image_path]

        # Encode each image to base64
        encoded_images = [encode_image(img_path) for img_path in image_paths]

        # Call the function to extract text from each image and compile the results
        extracted_texts = [extract_text_from_image(img_base64) for img_base64 in encoded_images]

        # Return structured data for each page/image
        return {"pages": extracted_texts}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")




# Function to simulate the GPT-based text extraction
def extract_text_from_image(image_base64: str) -> dict:
    # Your logic to interact with OpenAI API here
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                """These is a image of medical prescription written by doctor. Get all the text from the image.""",
                *map(lambda x: {"image": x, "resize": 768}, [image_base64]),
            ],
        },
    ]

    params = {
        "model": "gpt-4o",  # Adjust this as needed
        "messages": PROMPT_MESSAGES,
        "max_tokens": 500,
    }

    result = openai.chat.completions.create(**params)
    extracted_text = result.choices[0].message.content


# Set up prompt templates
    preamble = ("You are a medical expert AI chatbot having a conversation with a human. Your task is to provide accurate "
                "and helpful answers based on the extracted parts of a medical health report. Pay close attention to the "
                "medical report's structure, language, and any cross-references to ensure comprehensive and precise "
                "extraction of information. Do not use prior knowledge or information from outside the context to answer "
                "the questions. Only use the information provided in the context to answer the questions.")

    postamble = "Do not include any explanation in the reply. Only include the extracted information in the reply."

    system_template = "{preamble}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{format_instructions}\n\n{extracted_text}\n\n{postamble}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = PromptTemplate(
        input_variables=["preamble", "format_instructions", "extracted_text", "postamble"],
        template=human_template
    )



    parser = PydanticOutputParser(pydantic_object=PrescriptionMedical)


    request = chat_prompt.format_prompt(
        preamble=preamble,
        format_instructions=parser.get_format_instructions(),
        extracted_text=extracted_text,
        postamble=postamble
    ).to_messages()

    chat = ChatOpenAI(temperature=0.0)
    response = chat(request)

    # Parse the response into the Pydantic model
    parsed_output = parser.parse(response.content)

    # Serialize using Pydantic's model_dump method and format using json.dumps
    parsed_output_dict = parsed_output.model_dump()
    parsed_output_json = json.dumps(parsed_output_dict, indent=2)  # Format with indentation

    return parsed_output_dict


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)






# (base) akashdesai@AkashDesai-Precision-3580:~/projects/ICU_LAB_REPORTS/handwritten_report_lab_pic/ip_linux_deep$ chmod 400 deepaarogya_api.pem 
# (base) akashdesai@AkashDesai-Precision-3580:~/projects/ICU_LAB_REPORTS/handwritten_report_lab_pic/ip_linux_deep$ ls
# deepaarogya_api.pem
# (base) akashdesai@AkashDesai-Precision-3580:~/projects/ICU_LAB_REPORTS/handwritten_report_lab_pic/ip_linux_deep$ ssh -i deepaarogya_api.pem ubuntu@13.232.156.45The authenticity of host '13.232.156.45 (13.232.156.45)' can't be established.
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


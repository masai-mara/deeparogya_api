import uvicorn
from datetime import datetime
import uuid
import json
import pymysql

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATABASE_CONFIG = {
    "host": "deeparogya.cnkm82668omh.ap-south-1.rds.amazonaws.com",
    "user": "admin",
    "password": "Deepaarogya_backend002",
    "db": "deeparogya_development",
}

def get_conn():
    return pymysql.Connection(
        host=DATABASE_CONFIG["host"],
        user=DATABASE_CONFIG["user"],
        passwd=DATABASE_CONFIG["password"],
        db=DATABASE_CONFIG["db"],
    )
    
    
def rows_to_json_transform(rows, fields):
    json_data = []
    for row in rows:
        row_dict = dict(zip(fields, row))
        for key, value in row_dict.items():
            if isinstance(value, datetime):
                row_dict[key] = value.isoformat()
        json_data.append(row_dict)
    return json_data
    

def row_to_json_transform_single(row, schema_fields):
    result = {}
    for key, value in zip(schema_fields, row):
        if isinstance(value, datetime):  
            result[key] = value.isoformat()  
        else:
            result[key] = value  
    return result



db = get_conn()

 
async def insertExtractedData(extracted_texts,doctor_id):
    cursor = db.cursor()
    for text in extracted_texts:
        patient_id =  "gcs_id" # TODO this needs to be an actual id, and not something hardcoded
  
        prescription_extraction_id = uuid.uuid4()
        time_now_str = str(datetime.now())
        cursor.execute(
        "INSERT INTO ExtractedPrescription VALUES (%s, % s, % s, % s, % s, % s, % s, % s, %s, %s)",
        (
            patient_id,
            json.dumps(text["patient_info"]),
            json.dumps(text["diagnosis"]),
            json.dumps(text["drugs"]),
            json.dumps(text["diet_to_follow"]),
            json.dumps(text["history"]),
            json.dumps(text["signature"]),
            time_now_str,
            prescription_extraction_id,
            doctor_id,
        ),
        )
    cursor.close()
    db.commit()
    

async def getPrescriptionList(id):
    cursor = db.cursor()
    cursor.execute(
            "SELECT * FROM ExtractedPrescription where doctor_id = %s", (id,)
        )
    extracted_prescriptions = cursor.fetchall()
    cursor.close()
    db.commit()
    extracted_prescription_schema_fields = [
        "patient_id",
        "patient_info",
        "diagnosis",
        "drugs",
        "diet_to_follow",
        "history",
        "signature",
        "created_at",
        "prescription_extraction_id",
        "doctor_id",
    ]

    list_of_extracted_prescriptions_json = rows_to_json_transform(
        extracted_prescriptions, extracted_prescription_schema_fields
    )
    return list_of_extracted_prescriptions_json
    
    

async def getPrescription(prescription_id):
    cursor = db.cursor()
    
    try:
        cursor.execute(
            "SELECT * FROM ExtractedPrescription WHERE prescription_extraction_id = %s", (prescription_id,)
        )
        extracted_prescription = cursor.fetchone() 
        
        if not extracted_prescription:
            return []

        extracted_prescription_schema_fields = [
            "patient_id",
            "patient_info",
            "diagnosis",
            "drugs",
            "diet_to_follow",
            "history",
            "signature",
            "created_at",
            "prescription_extraction_id",
            "doctor_id",
        ]

        prescription_json = row_to_json_transform_single(
            extracted_prescription, extracted_prescription_schema_fields
        )
        
        return prescription_json

    except Exception as e:
        return []
    finally:
        cursor.close()
        db.commit()

    


async def editPrescription(data):
    cursor = db.cursor()
    time_now_str = str(datetime.now())
    try:
        cursor.execute(
            """
            UPDATE ExtractedPrescription
            SET 
                patient_id = %s,
                patient_info = %s,
                diagnosis = %s,
                drugs = %s,
                diet_to_follow = %s,
                history = %s,
                signature = %s
            WHERE 
                prescription_extraction_id = %s
            """,
            (
                data["patient_id"],
                json.dumps(data["patient_info"]),
                json.dumps(data["diagnosis"]),
                json.dumps(data["drugs"]),
                json.dumps(data["diet_to_follow"]),
                json.dumps(data["history"]),
                json.dumps(data["signature"]),
                data["prescription_id"],
            )
        )

        db.commit()

    except Exception as e:
        db.rollback() 
        print(f"Error: {e}")  
        raise e

    finally:
        cursor.close()
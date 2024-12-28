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


async def insertExtractedData(extracted_texts, doctor_id):
    cursor = db.cursor()
    for text in extracted_texts:
        patient_id = (
            "gcs_id"  # TODO this needs to be an actual id, and not something hardcoded
        )

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
    cursor.execute("SELECT * FROM ExtractedPrescription where doctor_id = %s", (id,))
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
            "SELECT * FROM ExtractedPrescription WHERE prescription_extraction_id = %s",
            (prescription_id,),
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
            ),
        )

        db.commit()

    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise e

    finally:
        cursor.close()


async def deletePrescription(prescription_id):
    """
    Deletes a prescription from the database.
    Args:
        prescription_id: The ID of the prescription to delete.
    Raises:
        Exception: If an error occurs during the deletion process.
    """

    cursor = db.cursor()
    try:
        cursor.execute(
            """
            DELETE FROM ExtractedPrescription
            WHERE prescription_extraction_id = %s
            """,
            (prescription_id,),
        )
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error deleting prescription: {e}")
        raise e
    finally:
        cursor.close()


async def signupUser(data):
    """
    Sings up a user.
    Args:
        data: The object containing name, email, phone, password of the user to signup.
    Raises:
        Exception: If an error occurs during the singup process.
    """
    name = data["name"]
    email = data["email"]
    phone = data["phone"]
    hashed_password = data["hashed_password"]
    dated = data["dated"]
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO Signup (name, email, phone, password, dated) VALUES (%s, %s, %s, %s, %s)",
            (name, email, phone, hashed_password, dated),
        )
        db.commit()
    except Exception as e:
        db.rollback()
        raise (e)
    finally:
        cursor.close()


async def getAccountPassword(email: str) -> str:
    """
    Returns password of a user.
    Args:
        email: Registered email of the user at signup.
    Raises:
        Exception: If an error occurs during the get password process.
    """
    cursor = db.cursor()
    try:
        cursor.execute("SELECT password FROM Signup WHERE email = %s", (email,))
        account = cursor.fetchone()
        stored_password = account[0]
        return stored_password
    except Exception as e:
        raise (e)
    finally:
        cursor.close()


async def updateSignInTime(email: str):
    """
    Updates last login time for a user.
    Args:
        email: Registered email of the user at signup.
    Raises:
        Exception: If an error occurs during the update lastlogin time process.
    """
    cursor = db.cursor()
    try:
        cursor.execute(
            "UPDATE Signup SET last_login_timestamp = NOW() WHERE email = %s", (email,)
        )
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error updating last login time: {e}")
        raise (e)
    finally:
        cursor.close()


async def addMessage(data: dict):
    """
    Updates last login time for a user.
    Args:
        data: Registered email of the user at signup.
    Raises:
        Exception: If an error occurs during the add message process.
    """
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO Message (name, email, subject, message) VALUES (%s, %s, %s, %s)",
            (data["name"], data["email"], data["subject"], data["message"]),
        )
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error adding the message: {e}")
        raise (e)
    finally:
        cursor.close()


async def getUserDetails(email: str) -> dict:
    """
    Returns details of a user.
    Args:
        email: Registered email of the user at signup.
    Raises:
        Exception: If an error occurs during the get password process.
    """
    cursor = db.cursor()
    try:
        cursor.execute("SELECT * FROM User WHERE email_id = %s", (email,))
        user = cursor.fetchone()
        return user
    except Exception as e:
        raise (e)
    finally:
        cursor.close()

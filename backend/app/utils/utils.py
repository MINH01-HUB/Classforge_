# app/utils.py
import pandas as pd
from app.models.model import RawStudentData
from app import db

def upload_raw_csv(filepath: str):
    """Reads a CSV and inserts records into the RawStudentData table."""
    df = pd.read_csv(filepath)

    for _, row in df.iterrows():
        student = RawStudentData(
            student_id=row['student_id'],
            gender=row['gender'],
            school_type=row['school_type'],
            parental_education=row['parental_education'],
            exam_score=row['exam_score']
        )
        db.session.add(student)

    db.session.commit()
    print("✅ Uploaded raw CSV data into PostgreSQL.")

from app.models.model import TransformedStudentData

def preprocess_raw_to_transformed():
    """Reads all raw data, encodes and normalizes it, and stores results in TransformedStudentData."""
    raw_data = RawStudentData.query.all()
    for entry in raw_data:
        transformed = TransformedStudentData(
            student_id=entry.student_id,
            encoded_gender=1 if entry.gender.lower() == 'female' else 0,
            encoded_school_type=0 if 'public' in entry.school_type.lower() else 1,
            encoded_parental_education=len(entry.parental_education.split()),
            normalized_score=entry.exam_score / 100.0
        )
        db.session.add(transformed)

    db.session.commit()
    print("✅ Transformed and stored encoded student data.")














# import torch
# import pandas as pd

# def preprocess_data(df):
#     df = df.copy()
#     df['gender'] = df['gender'].map({'male': 0, 'female': 1})
#     df['immigrant_status'] = df['immigrant_status'].astype(int)
#     df['SES'] = df['SES'].astype(float)
#     df['achievement'] = df['achievement'].astype(float)
#     df['psychological_distress'] = df['psychological_distress'].astype(float)
#     return df

# def allocate_students(df, model, criterion, num_classes):
#     if criterion == "academic":
#         features = ['achievement']
#     elif criterion == "wellbeing":
#         features = ['psychological_distress']
#     else:
#         features = ['achievement', 'psychological_distress']
    
#     inputs = torch.tensor(df[features].values, dtype=torch.float32)
#     with torch.no_grad():
#         outputs = model(inputs)
#         predicted_classes = torch.argmax(outputs, dim=1).numpy()

#     df['class'] = predicted_classes % num_classes
#     return df[['student_id', 'class']]

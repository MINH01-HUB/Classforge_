# app/utils/csv_handler.py

import os


import pandas as pd
from app.models.model import RawStudentData, TransformedStudentData
from app import db
from sklearn.preprocessing import LabelEncoder, StandardScaler




def upload_raw_csv(filepath: str):
    """Upload raw CSV data to SQLAlchemy database"""
    df = pd.read_csv(filepath)
    
    # Convert student_id to string
    df['student_id'] = df['student_id'].astype(str)

    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        existing = RawStudentData.query.filter_by(student_id=str(row['student_id'])).first()
        if existing:
            skipped += 1
            continue

        student = RawStudentData(
            student_id=str(row['student_id']),  # Ensure student_id is string
            gender=str(row['gender']),
            immigrant_status=str(row['immigrant_status']),
            SES=float(row['SES']),
            achievement=float(row['achievement']),
            psychological_distress=float(row['psychological_distress'])
        )
        db.session.add(student)
        inserted += 1

    db.session.commit()
    print(f"✅ Upload complete: {inserted} new rows inserted, {skipped} skipped (duplicates).")

    # Immediately preprocess and insert into TransformedStudentData
    preprocess_raw_to_transformed()

def preprocess_raw_to_transformed():
    """Preprocess raw data to transformed data"""
    raw_students = RawStudentData.query.all()
    if not raw_students:
        print("❌ No raw student data found.")
        return

    # Build DataFrame from raw student objects
    df = pd.DataFrame([{
        "student_id": str(s.student_id),
        "gender": s.gender,
        "immigrant_status": s.immigrant_status,
        "SES": s.SES,
        "achievement": s.achievement,
        "psychological_distress": s.psychological_distress
    } for s in raw_students])

    # Encode categorical features
    df['encoded_gender'] = LabelEncoder().fit_transform(df['gender'])
    df['encoded_immigrant_status'] = LabelEncoder().fit_transform(df['immigrant_status'])

    # Normalize continuous features
    scaler = StandardScaler()
    df[['SES', 'achievement', 'psychological_distress']] = scaler.fit_transform(
        df[['SES', 'achievement', 'psychological_distress']]
    )

    inserted = 0
    skipped = 0

    # Insert transformed records into DB
    for _, row in df.iterrows():
        if TransformedStudentData.query.filter_by(student_id=str(row['student_id'])).first():
            skipped += 1
            continue

        transformed = TransformedStudentData(
            student_id=str(row['student_id']),
            encoded_gender=int(row['encoded_gender']),
            encoded_immigrant_status=int(row['encoded_immigrant_status']),
            ses=float(row['SES']),
            achievement=float(row['achievement']),
            psychological_distress=float(row['psychological_distress'])
        )
        db.session.add(transformed)
        inserted += 1

    db.session.commit()
    print(f"✅ Preprocessing complete: {inserted} inserted, {skipped} skipped (already transformed).")









#version 1.0.0
# def preprocess_raw_to_transformed():
#     raw_students = RawStudentData.query.all()
#     if not raw_students:
#         print("❌ No raw student data found.")
#         return

#     df = pd.DataFrame([{
#         "student_id": s.student_id,
#         "gender": s.gender,
#         "immigrant_status": s.school_type,
#         "SES": float(s.parental_education),
#         "achievement": float(s.exam_score),
#         "psychological_distress": 0.0  # Add this field if your dataset supports it
#     } for s in raw_students])

#     # Encode categorical features
#     gender_enc = LabelEncoder()
#     immigrant_enc = LabelEncoder()
#     df['encoded_gender'] = gender_enc.fit_transform(df['gender'])
#     df['encoded_immigrant_status'] = immigrant_enc.fit_transform(df['immigrant_status'])

#     # Normalize continuous features
#     scaler = StandardScaler()
#     df[['SES', 'achievement', 'psychological_distress']] = scaler.fit_transform(
#         df[['SES', 'achievement', 'psychological_distress']]
#     )

#     for _, row in df.iterrows():
#         transformed = TransformedStudentData(
#             student_id=row['student_id'],
#             encoded_gender=row['encoded_gender'],
#             encoded_immigrant_status=row['encoded_immigrant_status'],
#             ses=row['SES'],
#             achievement=row['achievement'],
#             psychological_distress=row['psychological_distress']
#         )
#         db.session.add(transformed)

#     db.session.commit()
#     print("✅ Transformed data inserted with scaled features.")
from app import db
from datetime import datetime

# ------------------------------------------
# TABLE 1: RawStudentData
# Stores the original (unprocessed) student data uploaded from CSV
# ------------------------------------------
class RawStudentData(db.Model):
    __tablename__ = 'raw_student_data'
    student_id = db.Column(db.String(50), primary_key=True, unique=True, nullable=False)
    gender = db.Column(db.String(10))
    immigrant_status = db.Column(db.String(20))
    SES = db.Column(db.Float)
    achievement = db.Column(db.Float)
    psychological_distress = db.Column(db.Float)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<RawStudentData {self.student_id}>"

# ------------------------------------------
# TABLE 2: TransformedStudentData
# Stores the encoded and normalized data for ML/GAT model processing
# ------------------------------------------
class TransformedStudentData(db.Model):
    __tablename__ = 'transformed_student_data'
    student_id = db.Column(db.String(50), primary_key=True, unique=True, nullable=False)
    
    # Encoded features
    encoded_gender = db.Column(db.Integer)
    encoded_immigrant_status = db.Column(db.Integer)
    ses = db.Column(db.Float)
    achievement = db.Column(db.Float)
    psychological_distress = db.Column(db.Float)

    # GAT output
    class_id = db.Column(db.String(20), db.ForeignKey('classroom.id'), nullable=True)

    transformed_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<TransformedStudentData {self.student_id}>"

# ------------------------------------------
# TABLE 3: Classroom
# Optional if you want normalized class table for foreign key (future proof)
# ------------------------------------------
class Classroom(db.Model):
    __tablename__ = 'classroom'
    id = db.Column(db.String(20), primary_key=True)
    description = db.Column(db.String(100))
    students = db.relationship('TransformedStudentData', backref='classroom', lazy=True)

    def __repr__(self):
        return f"<Classroom {self.id}>"

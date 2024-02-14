import json, os
import boto3
from sqlalchemy import create_engine, Column, DateTime, String, Array, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy import func

database_name = os.environ['DATABASE_NAME']
video_table_name = os.environ['VIDEO_TABLE_NAME']
secret_name = os.environ['SECRET_NAME']
writer_endpoint = os.environ['DB_WRITER_ENDPOINT']

secrets_manager = boto3.client('secretsmanager')
credentials = json.loads(secrets_manager.get_secret_value(SecretId=self.secret_name)["SecretString"])
username = credentials["username"]
password = credentials["password"]

engine = create_engine(f'postgresql://{username}:{password}@{writer_endpoint}:5432/{database_name}')
Base = declarative_base()

class Video(Base):
    __tablename__ = video_table_name
    
    name = Column(String, primary_key=True)
    uploaded_at = Column(DateTime(timezone=True))
    summary = Column(String)
    summary_embedding = Column(Array(Float))
    
Session = sessionmaker(bind=engine)  
session = Session()
    
def handler(event, context):
    print("received event:")
    print(event)

    video_s3_path = ""
    for payload in event:
        if 'videoS3Path' in payload: video_s3_path = payload["videoS3Path"]

    video_name = os.path.basename(video_s3_path)

    # Validate video file extension. Only .mp4, .MP4, .mov, and .MOV are allowed.
    if video_name[-4:] not in [".mp4", ".MP4", ".mov", ".MOV"]:
        return {
            'statusCode': 400,
            'body': json.dumps({"preprocessing": "Unsupported video file extension. Only .mp4, .MP4, .mov, and .MOV are allowed."})
        }

    # Insert new video to database
    video = Video(name=video_name, uploaded_at=func.now(tz="UTC"))
    session.add(video)
    session.commit()

    return {
        'statusCode': 200,
        'body': json.dumps({"preprocessing": "success"})
    }

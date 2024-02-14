import json, os
import boto3
from sqlalchemy import create_engine, Column, Text, DateTime, String, func, bindparam
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import mapped_column, sessionmaker
from sqlalchemy.sql import bindparam
from sqlalchemy.dialects.postgresql import insert as db_insert


from pgvector.sqlalchemy import Vector

database_name = os.environ['DATABASE_NAME']
video_table_name = os.environ['VIDEO_TABLE_NAME']
secret_name = os.environ['SECRET_NAME']
writer_endpoint = os.environ['DB_WRITER_ENDPOINT']
embedding_dimension = os.environ['EMBEDDING_DIMENSION']

secrets_manager = boto3.client('secretsmanager')
credentials = json.loads(secrets_manager.get_secret_value(SecretId=self.secret_name)["SecretString"])
username = credentials["username"]
password = credentials["password"]

engine = create_engine(f'postgresql://{username}:{password}@{writer_endpoint}:5432/{database_name}')
Base = declarative_base()

class Videos(Base):
    __tablename__ = video_table_name
    
    name = Column(String(200), primary_key=True, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), nullable=False)
    summary = Column(Text)
    summary_embedding = mapped_column(Vector(int(embedding_dimension)))
    
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

    # Parameterize
    video_name_param = bindparam('name') 
    uploaded_at_param = bindparam('uploaded_at')

    upsert = insert(Videos).values(
        name=video_name_param,
        uploaded_at=uploaded_at_param
    )
    
    upsert = upsert.on_conflict_do_update(
        constraint=f"{videos_table_name}_pkey",
        set_={
            Videos.uploaded_at: uploaded_at_param
        }
    )

    with session.begin():
        session.execute(upsert, {
            "name": video_name,  
            "uploaded_at": func.now(tz="UTC")
        })
        session.commit()

    return {
        'statusCode': 200,
        'body': json.dumps({"preprocessing": "success"})
    }

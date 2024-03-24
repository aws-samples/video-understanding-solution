import os, json
import boto3
import urllib.parse
from sqlalchemy import create_engine, Column, DateTime, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, mapped_column
from sqlalchemy.sql import bindparam
from pgvector.sqlalchemy import Vector
from datetime import datetime

bedrock = boto3.client("bedrock-runtime")
secrets_manager = boto3.client('secretsmanager')

reader_endpoint = os.environ['DB_READER_ENDPOINT']
database_name = os.environ['DATABASE_NAME']
video_table_name = os.environ['VIDEO_TABLE_NAME']
secret_name = os.environ['SECRET_NAME']
embedding_model_id = os.environ["EMBEDDING_MODEL_ID"]
embedding_dimension = int(os.environ['EMBEDDING_DIMENSION'])
acceptable_embedding_distance = float(os.environ['ACCEPTABLE_EMBEDDING_DISTANCE'])
display_page_size = int(os.environ['DISPLAY_PAGE_SIZE'])

credentials = json.loads(secrets_manager.get_secret_value(SecretId=secret_name)["SecretString"])
username = credentials["username"]
password = credentials["password"]

engine = create_engine(f'postgresql://{username}:{password}@{reader_endpoint}:5432/{database_name}')
Base = declarative_base()

Session = sessionmaker(bind=engine)  
session = Session()

class Videos(Base):
    __tablename__ = video_table_name
    
    name = Column(String(200), primary_key=True, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), nullable=False)
    summary = Column(Text)
    summary_embedding = mapped_column(Vector(int(embedding_dimension)))


def handler(event, context):
    params = event["queryStringParameters"]
    
    page = int(params["page"])
    video_name_contains = urllib.parse.unquote(params["videoNameContains"]) if "videoNameContains" in params else None
    uploaded_between= urllib.parse.unquote(params["uploadedBetween"]) if "uploadedBetween" in params else None
    about = urllib.parse.unquote(params["about"]) if "about" in params else None
   
    # Use SQLAlchemy to search videos with the 3 filters above.
    videos = session.query(Videos.name)
    if video_name_contains is not None:
        video_name_contains_param = bindparam("name")
        videos = videos.filter(Videos.name.like(video_name_contains_param))
    if uploaded_between is not None:
        # Assume uploaded_between is like 2024-02-07T16:00:00.000Z|2024-02-15T16:00:00.000Z
        start, stop = uploaded_between.split("|")
        start = datetime.strptime(start[:-5], "%Y-%m-%dT%H:%M:%S")
        stop = datetime.strptime(stop[:-5], "%Y-%m-%dT%H:%M:%S")
        start_param = bindparam("start")
        stop_param = bindparam("stop")
        videos = videos.filter(Videos.uploaded_at.between(start_param, stop_param))
    if about is not None:
        # Get the embedding for the video topic
        body = json.dumps({
            "texts":[about],
            "input_type": "search_query",
        })
        call_done = False
        while(not call_done):
            try:
                response = bedrock.invoke_model(body=body, modelId=embedding_model_id)
                call_done = True
            except ThrottlingException:
                print("Amazon Bedrock throttling exception")
                time.sleep(60)
            except Exception as e:
                raise e

        # Disabling semgrep rule for checking data size to be loaded to JSON as the source is from Amazon Bedrock
        # nosemgrep: python.aws-lambda.deserialization.tainted-json-aws-lambda.tainted-json-aws-lambda
        about_embedding = json.loads(response.get("body").read().decode())['embeddings'][0]

        videos = videos.filter(Videos.summary_embedding.cosine_distance(about_embedding) < acceptable_embedding_distance)
        videos = videos.order_by(Videos.summary_embedding.cosine_distance(about_embedding))
    
    videos = videos.order_by(Videos.uploaded_at.desc())

    if video_name_contains is not None:
       videos = videos.params(name=f"%{video_name_contains}%")
    if uploaded_between is not None:
       videos = videos.params(start=start, stop=stop)

    videos = videos.offset(page*display_page_size).limit(display_page_size+1)
    
    video_names = videos.all()
    video_names = [v.name for v in video_names]

    next_page = None
    if len(video_names) > display_page_size:
        video_names = video_names[:display_page_size]
        next_page = page + 1

    response_payload = {
        "videos": video_names,
        "pageSize": display_page_size,
    }
    if next_page is not None: response_payload['nextPage'] = next_page

    response = {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(response_payload)
    }

    return response
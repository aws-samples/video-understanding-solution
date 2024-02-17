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
embedding_dimension = os.environ['EMBEDDING_DIMENSION']

page_size = 25
acceptable_embedding_distance = 50

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
    video_name_starts_with = urllib.parse.unquote(params["video_name_starts_with"]) if "video_name_starts_with" in params else None
    uploaded_between= urllib.parse.unquote(params["uploaded_between"]) if "uploaded_between" in params else None
    about = urllib.parse.unquote(params["about"]) if "about" in params else None
   
    # Use SQLAlchemy to search videos with the 3 filters above.
    videos = session.query(Videos.name)
    if video_name_starts_with is not None:
        print(video_name_starts_with)
        video_name_starts_with_param = bindparam("name")
        videos = videos.filter(Videos.name.like(f"{video_name_starts_with}%"))#video_name_starts_with_param))
    if uploaded_between is not None:
        # Assume uploaded_between is like 2024-02-07T16:00:00.000Z|2024-02-15T16:00:00.000Z
        start, stop = uploaded_between.split("|")
        start = datetime.strptime(start[:-5], "%Y-%m-%dT%H:%M:%S")
        stop = datetime.strptime(stop[:-5], "%Y-%m-%dT%H:%M:%S")
        #start_param = bindparam("start")
        #stop_param = bindparam("stop")
        print(start)
        print(stop)
        videos = videos.filter(Videos.uploaded_at.between(start, stop))#start_param, stop_param))
    if about is not None:
        print(about)
        # Get the embedding for the video topic
        body = json.dumps(
            {
                "inputText": about,
            }
        )
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
        about_embedding = json.loads(response.get("body").read())["embedding"]
        print(about_embedding)
        #summary_embedding_param = bindparam("summary_embedding")
        videos = videos.filter(Videos.summary_embedding.l2_distance(about_embedding) < acceptable_embedding_distance)

    #if video_name_starts_with is not None:
    #   videos = videos.params(name=f"{video_name_starts_with}%")
    #if uploaded_between is not None:
    #    videos = videos.params(start=start, stop=stop)
    #if about is not None:
    #    videos = videos.params(summary_embedding=about_embedding)

    videos = videos.offset(page*page_size).limit(page_size+1)
    print("actual query")
    print(str(videos))
    

    video_names = videos.all()
    video_names = [v.name for v in video_names]
    print(video_names)
    next_page = None
    if len(video_names) > page_size:
        video_names = video_names[:page_size]
        next_page = page + 1

    response_payload = {
        "videos": video_names,
        "page_size": page_size,
    }
    if next_page is not None: response_payload['next_page'] = next_page
    print(response_payload)

    response = {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*"
        },
        "body": json.dumps(response_payload)
    }

    return response
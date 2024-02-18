import json, os
import boto3
import psycopg2

database_name = os.environ['DATABASE_NAME']
video_table_name = os.environ['VIDEO_TABLE_NAME']
entities_table_name = os.environ['ENTITIES_TABLE_NAME']
content_table_name = os.environ['CONTENT_TABLE_NAME']
secret_name = os.environ['SECRET_NAME']
embedding_dimension = os.environ["EMBEDDING_DIMENSION"]
writer_endpoint = os.environ['DB_WRITER_ENDPOINT']

class Database():
    def __init__(self, writer, database_name, embedding_dimension, port=5432):
        self.writer_endpoint = writer
        self.username = None
        self.password = None
        self.port = port
        self.database_name = database_name
        self.video_table_name = video_table_name
        self.entities_table_name = entities_table_name 
        self.content_table_name = content_table_name 
        self.secret_name = secret_name
        self.embedding_dimension = embedding_dimension
        self.conn = None
    
    def fetch_credentials(self):
        secrets_manager = boto3.client('secretsmanager')
        credentials = json.loads(secrets_manager.get_secret_value(SecretId=self.secret_name)["SecretString"])
        self.username = credentials["username"]
        self.password = credentials["password"]
    
    def connect_for_writing(self):
        if self.username is None or self.password is None: self.fetch_credentials()
            
        conn = psycopg2.connect(host=self.writer_endpoint, port=self.port, user=self.username, password=self.password, database=self.database_name)
        self.conn = conn
        return conn
    
    def close_connection(self):
        self.conn.close()
        self.conn = None
    
    def setup_vector_db(self, embedding_dimension):
        if self.conn is None:
            self.connect_for_writing()
        
        cur = self.conn.cursor()

        # Create pgVector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # For the statements below, semgrep rule for flagging formatted query is disabled as this Lambda is to be invoked in deployment phase by CloudFormation, not user facing.

        # Create videos table and set indexes
        # nosemgrep: python.lang.security.audit.formatted-sql-query.formatted-sql-query, python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
        cur.execute(f"CREATE TABLE {self.video_table_name} (name varchar(200) PRIMARY KEY NOT NULL, uploaded_at timestamp without time zone NOT NULL DEFAULT (current_timestamp AT TIME ZONE 'UTC'), summary text, summary_embedding vector({str(embedding_dimension)}));")
        # nosemgrep: python.lang.security.audit.formatted-sql-query.formatted-sql-query, python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
        cur.execute(f"CREATE INDEX name_index ON {self.video_table_name} (name);")
        # nosemgrep: python.lang.security.audit.formatted-sql-query.formatted-sql-query, python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
        cur.execute(f"CREATE INDEX uploaded_at_index ON {self.video_table_name} (uploaded_at);")
        # nosemgrep: python.lang.security.audit.formatted-sql-query.formatted-sql-query, python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
        cur.execute(f"CREATE INDEX name_and_uploaded_at_index ON {self.video_table_name} (name, uploaded_at);")
        
        # Create entities table
        # nosemgrep: python.lang.security.audit.formatted-sql-query.formatted-sql-query, python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
        cur.execute(f"CREATE TABLE {self.entities_table_name} (id bigserial PRIMARY KEY NOT NULL, name VARCHAR(100) NOT NULL, sentiment VARCHAR(20), reason text, video_name varchar(200) NOT NULL REFERENCES {self.video_table_name}(name));")
       
        # Create content table
        # nosemgrep: python.lang.security.audit.formatted-sql-query.formatted-sql-query, python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
        cur.execute(f"CREATE TABLE {self.content_table_name} (id bigserial PRIMARY KEY NOT NULL, chunk text NOT NULL, chunk_embedding vector({str(embedding_dimension)}), video_name varchar(200) NOT NULL REFERENCES {self.video_table_name}(name));")

        self.conn.commit()
        cur.close()
        return True
    
db = Database(writer=writer_endpoint, database_name = database_name, embedding_dimension = embedding_dimension)
def on_event(event, context):
    request_type = event['RequestType'].lower()
    if request_type == 'create':
        return on_create(event)
    if request_type == 'update':
        return on_update(event)
    if request_type == 'delete':
        return on_delete(event)
    raise Exception(f'Invalid request type: {request_type}')


def on_create(event):
    try:
        db.setup_vector_db(embedding_dimension)
        db.close_connection()
    except Exception as e:
        print(e)
    
    return {'PhysicalResourceId': "VectorDBDatabaseSetup"}


def on_update(event):
    physical_id = event["PhysicalResourceId"]
    print("no op")
    return {'PhysicalResourceId': physical_id}

def on_delete(event):
    physical_id = event["PhysicalResourceId"]
    print("no op")
    return {'PhysicalResourceId': physical_id}

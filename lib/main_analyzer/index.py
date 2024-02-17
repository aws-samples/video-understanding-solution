import os, time, json, copy, math, re
from abc import ABC, abstractmethod
import boto3
from botocore.config import Config
from sqlalchemy import create_engine, Column, DateTime, String, Text, Integer, func, ForeignKey, update
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import mapped_column, sessionmaker
from pgvector.sqlalchemy import Vector

truncation_for_text = 50
truncation_for_scene = 50
video_script_storage_chunk_size = 10000 # characters

config = Config(read_timeout=1000) # Extends botocore read timeout to 1000 seconds

rekognition = boto3.client('rekognition')
transcribe = boto3.client("transcribe")
#sagemaker = boto3.client("sagemaker-runtime")
secrets_manager = boto3.client('secretsmanager')
bedrock = boto3.client(service_name="bedrock-runtime", config=config)
s3 = boto3.client("s3")

model_id = os.environ["MODEL_ID"]
embedding_model_id = os.environ["EMBEDDING_MODEL_ID"]
embedding_dimension = os.environ['EMBEDDING_DIMENSION']
bucket_name = os.environ["BUCKET_NAME"]
raw_folder = os.environ["RAW_FOLDER"]
video_script_folder = os.environ["VIDEO_SCRIPT_FOLDER"]
transcription_folder = os.environ["TRANSCRIPTION_FOLDER"]
entity_sentiment_folder = os.environ["ENTITY_SENTIMENT_FOLDER"]
summary_folder = os.environ["SUMMARY_FOLDER"]

database_name = os.environ['DATABASE_NAME']
video_table_name = os.environ['VIDEO_TABLE_NAME']
entities_table_name = os.environ['ENTITIES_TABLE_NAME']
content_table_name = os.environ['CONTENT_TABLE_NAME']
secret_name = os.environ['SECRET_NAME']
writer_endpoint = os.environ['DB_WRITER_ENDPOINT']

video_s3_path = os.environ['VIDEO_S3_PATH']
labels_job_id = os.environ['LABEL_DETECTION_JOB_ID']
texts_job_id = os.environ['TEXT_DETECTION_JOB_ID']
transcription_job_name = os.environ['TRANSCRIPTION_JOB_NAME']

credentials = json.loads(secrets_manager.get_secret_value(SecretId=secret_name)["SecretString"])
username = credentials["username"]
password = credentials["password"]

engine = create_engine(f'postgresql://{username}:{password}@{writer_endpoint}:5432/{database_name}')
Base = declarative_base()

Session = sessionmaker(bind=engine)  
session = Session()

print("1")

def handler():
    print("handler called")

    video_name = os.path.basename(video_s3_path)
    video_path= '/'.join(video_s3_path.split('/')[1:])

    video_transcript_s3_path = f"{transcription_folder}/{video_path}.txt"

    try:
        wait_for_rekognition_label_detection(labels_job_id=labels_job_id, sort_by='TIMESTAMP')
        print("waiting rekognition visual done")
        wait_for_rekognition_text_detection(texts_job_id=texts_job_id)
        print("waiting rekognition text done")
        wait_for_transcription_job(transcription_job_name=transcription_job_name)
        print("waiting done")
        response = analyze_video(labels_job_id=labels_job_id, texts_job_id=texts_job_id, video_transcript_s3_path=video_transcript_s3_path)
        print("analyzing done")
        store_summary_result(summary=response['summary'], video_name=video_name, video_path=video_path)
        print("summary store done")
        store_sentiment_result(sentiment_string=response['sentiment'], video_name=video_name, video_path=video_path)
        print("sentiment store done")
        store_video_script_result(video_script=response['video_script'], video_name=video_name, video_path=video_path)
        print("chunk store done")
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise

    return {
        'statusCode': 200,
        'body': json.dumps({"main_analyzer": "success"})
    }

def store_summary_result(summary, video_name, video_path):
    # Store summary in S3
    s3.put_object(
        Body=summary, 
        Bucket=bucket_name, 
        Key=f"{summary_folder}/{video_path}.txt"
    )

    # Get summary embedding
    body = json.dumps(
        {
            "inputText": summary,
        }
    )
    response = bedrock.invoke_model(body=body, modelId=embedding_model_id)
    # Disabling semgrep rule for checking data size to be loaded to JSON as the source is from Amazon Bedrock
    # nosemgrep: python.aws-lambda.deserialization.tainted-json-aws-lambda.tainted-json-aws-lambda
    summary_embedding = json.loads(response.get("body").read())["embedding"]

    # Store summary in database
    update_stmt = (
        update(Videos).
        where(Videos.name == video_path).
        values(summary = summary, summary_embedding = summary_embedding)
    )
    
    session.execute(update_stmt)
    session.commit()


    print("summary embedding stored")

def store_sentiment_result(sentiment_string, video_name, video_path):
    print("sentiment string is")
    print(sentiment_string)
    # Extract entities and sentiment from the string
    entities_dict = {}
    entity_regex = r"\n*([\w\s\'\.]+?)\s*:\s*(positive|negative|neutral|mixed)"
    for match in re.finditer(entity_regex, sentiment_string):
        entity = match.group(1).strip()
        sentiment = match.group(2)
        entities_dict[entity] = sentiment
    print("entities dict")
    print(entities_dict)

    s3.put_object(
        Body="\n".join(f"{e} : {s}" for e, s in entities_dict.items()), 
        Bucket=bucket_name, 
        Key=f"{entity_sentiment_folder}/{video_path}.txt"
    )

    entities = []
    for entity, sentiment in entities_dict.items():
        # Store entity in database
        entities.append(Entities(
            name=entity,
            sentiment=sentiment,
            video_name=video_path
        ))

    # Store into database
    session.add_all(entities)
    session.commit()

    print("entities done")
    print(entities_dict)
    
def store_video_script_result(video_script, video_name, video_path):
    s3.put_object(
        Body=video_script, 
        Bucket=bucket_name, 
        Key=f"{video_script_folder}/{video_path}.txt"
    )

    # Chunking the video script for storage in DB while converting them to embedding
    video_script_length = len(video_script)
    number_of_chunks = math.ceil( (video_script_length + 1) / video_script_storage_chunk_size )
    print("num chunks video script storage")
    print(number_of_chunks)

    chunks = []
    for chunk_number in range(0, number_of_chunks):
        is_last_chunk = (chunk_number == (number_of_chunks - 1))
        is_first_chunk = (chunk_number == 0)

        start = 0 if is_first_chunk else int(chunk_number*video_script_storage_chunk_size)
        stop = video_script_length if is_last_chunk else (chunk_number+1)*video_script_storage_chunk_size
        chunk_string = video_script[start:stop]
    
        # So long as this is not the first chunk, remove whatever before first \n since likely the chunk cutting is not done exactly at the \n
        if not is_first_chunk:
            chunk_string = chunk_string[chunk_string.index("\n"):] 
        
        # Get the embedding for the chunk
        body = json.dumps(
            {
                "inputText": chunk_string,
            }
        )
        call_done = False
        while(not call_done):
            try:
                response = bedrock.invoke_model(body=body, modelId=embedding_model_id)
                print("embedding response")
                print(response)
                call_done = True
            except ThrottlingException:
                print("Amazon Bedrock throttling exception")
                time.sleep(60)
            except Exception as e:
                raise e

        # Disabling semgrep rule for checking data size to be loaded to JSON as the source is from Amazon Bedrock
        # nosemgrep: python.aws-lambda.deserialization.tainted-json-aws-lambda.tainted-json-aws-lambda
        chunk_embedding = json.loads(response.get("body").read())["embedding"]
        
        # Create database object
        chunks.append(Contents(
            chunk=chunk_string,
            chunk_embedding=chunk_embedding,
            video_name=video_path
        ))

    # Store in database
    session.add_all(chunks)
    session.commit()

def analyze_video(labels_job_id, texts_job_id, video_transcript_s3_path):
    video_scenes, video_length = visual_scenes_iterate_pages(labels_job_id)
    video_texts = visual_texts_iterate_pages(texts_job_id)
    video_transcription_file = s3.get_object(Bucket=bucket_name, Key=video_transcript_s3_path)
    transcript = json.loads(video_transcription_file['Body'].read().decode('utf-8'))
    
    video_analyzer = VideoAnalyzerBedrock(model_id, video_length, video_scenes, video_texts, transcript)
    response = video_analyzer.run()
    return response

def wait_for_rekognition_label_detection(labels_job_id, sort_by):
        getObjectDetection = rekognition.get_label_detection(JobId=labels_job_id, SortBy=sort_by)
        while(getObjectDetection['JobStatus'] == 'IN_PROGRESS'):
            time.sleep(5)
            print('.', end='')
            getObjectDetection = rekognition.get_label_detection(JobId=labels_job_id, SortBy=sort_by)

def wait_for_rekognition_text_detection(texts_job_id):
    getTextDetection = rekognition.get_text_detection(JobId=texts_job_id)
    while(getTextDetection['JobStatus'] == 'IN_PROGRESS'):
        time.sleep(5)
        print('.', end='')
        getTextDetection = rekognition.get_text_detection(JobId=texts_job_id)

def wait_for_transcription_job(transcription_job_name):
    getTranscription = transcribe.get_transcription_job(TranscriptionJobName=transcription_job_name)
    while(getTranscription ["TranscriptionJob"]["TranscriptionJobStatus"] == 'IN_PROGRESS'):
        time.sleep(5)
        print('.', end='')
        getTranscription = transcribe.get_transcription_job(TranscriptionJobName=transcription_job_name)

def extract_visual_scenes(scenes, getObjectDetection):
    for l in getObjectDetection['Labels']:
        scene = []
        timestamp = int(l['Timestamp']/1000)
        if timestamp in scenes:
            scene = scenes[timestamp]
        else:
            scenes[timestamp] =scene
        detected_labels = l['Label']['Name']

        # To avoid having too many detected scenes, cut the detected scene per frame to a certain number (default = 25)
        if (detected_labels not in scene) and (len(scene) <= truncation_for_scene):
            scene.append(detected_labels)
    return scenes

def visual_scenes_iterate_pages(labels_job_id):
    scenes = {}
    getObjectDetection = rekognition.get_label_detection(
        JobId=labels_job_id,
        MaxResults=1000,
        SortBy='TIMESTAMP'
    )
    video_duration = float(getObjectDetection["VideoMetadata"]["DurationMillis"])/1000
    scenes = extract_visual_scenes(scenes, getObjectDetection)

    while("NextToken" in getObjectDetection):
        getObjectDetection = rekognition.get_label_detection(
            JobId=labels_job_id,
            MaxResults=1000,
            NextToken=getObjectDetection["NextToken"]
        )
        scenes = extract_visual_scenes(scenes, getObjectDetection)
    
    return scenes, video_duration

def extract_visual_texts(texts, getTextDetection):
    for l in getTextDetection['TextDetections']:
        if l['TextDetection']["Type"] == "WORD": continue
        
        text = []
        timestamp = int(l['Timestamp']/1000)
        if timestamp in texts:
            text = texts[timestamp]
        else:
            texts[timestamp] = text

        detected_texts = l['TextDetection']['DetectedText']

        # To avoid having too many detected text, cut the detected text per frame to a certain number (default = 25)
        if len(text) <= truncation_for_text: 
            text.append(detected_texts)
    return texts
    
def visual_texts_iterate_pages(texts_job_id):
    texts = {}
    getTextDetection = rekognition.get_text_detection(
        JobId=texts_job_id,
        MaxResults=1000
    )
    texts = extract_visual_texts(texts, getTextDetection)
    while("NextToken" in getTextDetection):
        getTextDetection = rekognition.get_text_detection(
            JobId=texts_job_id,
            MaxResults=1000,
            NextToken=getTextDetection["NextToken"]
        )
        texts = extract_visual_texts(texts, getTextDetection)
    
    return texts

class Videos(Base):
    __tablename__ = video_table_name
    
    name = Column(String(200), primary_key=True, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), nullable=False)
    summary = Column(Text)
    summary_embedding = mapped_column(Vector(int(embedding_dimension)))

class Entities(Base):
    __tablename__ = entities_table_name
    
    id = Column(Integer, primary_key=True) 
    name = Column(String(100), nullable=False)
    sentiment = Column(String(20), nullable=False)
    video_name = Column(ForeignKey(f"{video_table_name}.name"), nullable=False)
    
class Contents(Base):
    __tablename__ = content_table_name
    
    id = Column(Integer, primary_key=True)
    chunk = Column(Text, nullable=False)
    chunk_embedding = Column(Vector(int(embedding_dimension))) 
    video_name = Column(ForeignKey(f"{video_table_name}.name"), nullable=False)
    
class VideoAnalyzer(ABC):
    def __init__(self, video_length, video_scenes, video_texts, transcript):
        self.original_scenes = video_scenes
        self.original_visual_texts = video_texts
        self.original_transcript = transcript
        self.scenes = []
        self.visual_texts = []
        self.transcript = []
        self.video_script_chunk_size = 300000 # characters
        self.video_script_chunk_overlap = 1000 # characters
        self.scene_similarity_score = 0.5
        self.text_similarity_score = 0.5
        self.video_rolling_summary = ""
        self.video_rolling_sentiment = ""
        self.combined_video_script = ""
        self.all_combined_video_script = ""
        self.llm_parameters = {}
  
    @abstractmethod
    def call_llm(self, prompt):
        pass
  
    def preprocess_scenes(self):
        scenes = dict(sorted(copy.deepcopy(self.original_scenes).items()))
        prev_objects = []
        for k,v in list(scenes.items()):
            # The loop below is to check how much of this scene resembling previous scene
            num_of_matches_with_prev_objects = 0
            for i in v:
                if i in prev_objects:
                    num_of_matches_with_prev_objects += 1
            #  Delete scene entry if there is not detected object
            if v == []: 
                del scenes[k]
            # Delete scene entry if the detected objects are the same as the previous scene
            elif v == prev_objects:
                del scenes[k]
            # Delete scene entry if the detected objects are too similar with the previous scene
            elif float(num_of_matches_with_prev_objects) > len(prev_objects)*self.scene_similarity_score:
                del scenes[k]
            else:
                prev_objects = v
      
        self.scenes = sorted(scenes.items())
      
    def preprocess_visual_texts(self):
        visual_texts = dict(sorted(self.original_visual_texts.items()))
        prev_texts = []
        for k,v in list(visual_texts.items()):
            # The loop below is to check how much of this text resembling previous text
            num_of_matches_with_prev_texts = 0
            for i in v:
                if i in prev_texts:
                    num_of_matches_with_prev_texts += 1
            #  Delete text entry if there is not detected text
            if v == []: 
                del visual_texts[k]
            # Delete text entry if the detected texts are the same as the previous text
            elif v == prev_texts:
                del visual_texts[k]
            # Delete text entry if the detected texts are too similar with the previous scene
            elif float(num_of_matches_with_prev_texts) > len(prev_texts)*self.text_similarity_score:
                del visual_texts[k]
            else:
                prev_texts = v
      
        self.visual_texts = sorted(visual_texts.items())

    def preprocess_transcript(self):
        transcript= {}
        word_start_time = None
        sentence_start_time = -1
        sentence = ""
        for item in self.original_transcript["results"]["items"]:
            if item['type'] == 'punctuation':
                # Add punctuation to sentence without heading space
                sentence += f"{ item['alternatives'][0]['content'] }"
                # Add sentence to transcription
                transcript[word_start_time] = sentence
                # Reset the sentence and sentence start time
                sentence = ""
                sentence_start_time  = -1
            else:
                word_start_time = int(float(item['start_time']))
                if sentence_start_time  == -1:
                    # Add word to sentence with heading space
                    sentence += f"{ item['alternatives'][0]['content'] }"
                    # Set the start time of the sentence to be the start time of this first word in the sentence
                    sentence_start_time  = word_start_time
                else:
                    # Add word to sentence with heading space
                    sentence += f" { item['alternatives'][0]['content'] }"

                self.transcript = sorted(transcript.items())
      
    def generate_combined_video_script(self):
        def transform_scenes(x):
            timestamp = x[0]
            objects = ",".join(x[1])
            return (timestamp, f"Scene:{objects}")
        scenes  = list(map(transform_scenes, self.scenes))

        def transform_texts(x):
            timestamp = x[0]
            texts = ",".join(x[1])
            return (timestamp, f"Text:{texts}")
        visual_texts  = list(map(transform_texts, self.visual_texts))

        def transform_transcript(x):
            timestamp = x[0]
            transcript = x[1]
            return (timestamp, f"Voice:{transcript}")
        transcript  = list(map(transform_transcript, self.transcript))
 
        # Combine all inputs
        combined_video_script = sorted( scenes + visual_texts + transcript)
        
        combined_video_script = "\n".join(list(map(lambda x: f"{x[0]}:{x[1]}", combined_video_script)))
        
        self.combined_video_script = combined_video_script
        self.all_combined_video_script += combined_video_script
  
    def generate_summary(self):
        prompt_prefix = "You are an expert video analyst who reads a VIDEO SCRIPT and creates summary of the video and why it is interesting.\n" \
                        "The VIDEO SCRIPT contains the visual scenes, the visual texts, and human voice in the video."
        
        video_script_length = len(self.combined_video_script)

        # When the video is short enough to fit into 1 chunk
        if video_script_length <= self.video_script_chunk_size:
            core_prompt = f"The VIDEO SCRIPT has format below.\n" \
                            "timestamp in seconds:scene / text / voice\n" \
                            "<VideoScript>\n" \
                            f"{self.combined_video_script}\n" \
                            "</VideoScript>\n"
          
            prompt = f"{prompt_prefix}\n\n" \
            f"{core_prompt}\n\n" \
            "Given the VIDEO SCRIPT above, decribe the summary of the video and why it is interesting. DO NOT make up anything you do not know. DO NOT mention about \"video script\" as your audience might not be aware of its presence.\n" \
            "Give the summary directly without any intro.\n" \
            "Summary: "
          
            self.video_rolling_summary = self.call_llm(prompt)
        # When the video is long enough to be divided into multiple chunks to fit within LLM's context length
        else:
            number_of_chunks = math.ceil( (video_script_length + 1) / (self.video_script_chunk_size - self.video_script_chunk_overlap) )

            for chunk_number in range(0, number_of_chunks):
                is_last_chunk = (chunk_number == (number_of_chunks - 1))
                is_first_chunk = (chunk_number == 0)

                start = 0 if is_first_chunk else int(chunk_number*self.video_script_chunk_size - self.video_script_chunk_overlap)
                stop = video_script_length if is_last_chunk else (chunk_number+1)*self.video_script_chunk_size
                chunk_combined_video_script = self.combined_video_script[start:stop]
            
                # So long as this is not the first chunk, remove whatever before first \n since likely the chunk cutting is not done exactly at the \n
                if not is_first_chunk:
                    chunk_combined_video_script = chunk_combined_video_script[chunk_combined_video_script.index("\n"):]
                    
                core_prompt = f"The VIDEO SCRIPT has format below.\n" \
                        "timestamp in seconds:scene / text / voice\n" \
                        "<VideoScript>\n" \
                        f"{chunk_combined_video_script}\n" \
                        "</VideoScript>\n"
                
                if is_last_chunk:
                    prompt = f"{prompt_prefix}\n\n" \
                    f"The video has {number_of_chunks} parts.\n\n" \
                    f"Below is the summary of all previous parts of the video::\n\n" \
                    f"{self.video_rolling_summary}\n\n" \
                    f"The below VIDEO SCRIPT is only for the LAST video part.\n\n" \
                    f"{core_prompt}\n\n" \
                    "Given the previous summary and the VIDEO SCRIPT above, decribe the summary of the whole video and why it is interesting. DO NOT make up anything you do not know. DO NOT mention about \"video script\" as your audience might not be aware of its presence.\n" \
                    "Give the summary directly without any intro.\n" \
                    "Summary: "
                    
                    chunk_summary = self.call_llm(prompt)
                    self.video_rolling_summary = chunk_summary
                elif is_first_chunk:
                    prompt = f"{prompt_prefix}\n\n" \
                    f"The video has {number_of_chunks} parts. The below VIDEO SCRIPT is only for the first part.\n\n" \
                    f"{core_prompt}\n\n" \
                    f"Given VIDEO SCRIPT above, decribe the summary of the video so far. DO NOT make up anything you do not know.\n" \
                    "Summary: "
                    
                    chunk_summary = self.call_llm(prompt)
                    self.video_rolling_summary = chunk_summary
                else:
                    prompt = f"{prompt_prefix}\n\n" \
                    f"The video has {number_of_chunks} parts.\n\n" \
                    f"Below is the summary of all previous parts of the video:\n\n" \
                    f"{self.video_rolling_summary}\n\n" \
                    f"The below VIDEO SCRIPT is only for part {chunk_number} of the video.\n\n" \
                    f"{core_prompt}\n\n" \
                    "Given the previous summary and the VIDEO SCRIPT above, decribe the summary of the whole video so far. DO NOT make up anything you do not know.\n" \
                    "Summary: "
                    
                    chunk_summary = self.call_llm(prompt)
                    self.video_rolling_summary = chunk_summary

        return self.video_rolling_summary
    
    def extract_sentiment(self):
        prompt_prefix = "You are an expert video analyst who reads a VIDEO SCRIPT and extract entities and their associated sentiment from the video.\n" \
                        "The VIDEO SCRIPT contains the visual scenes, the visual texts, and human voice in the video."
        
        video_script_length = len(self.combined_video_script)

        # When the video is short enough to fit into 1 chunk
        if video_script_length <= self.video_script_chunk_size:
            core_prompt = f"The VIDEO SCRIPT has format below.\n" \
                            "timestamp in seconds:scene / text / voice\n" \
                            "<VideoScript>\n" \
                            f"{self.combined_video_script}\n" \
                            "</VideoScript>\n"
          
            prompt = f"{prompt_prefix}\n\n" \
            f"{core_prompt}\n\n" \
            "Now your job is to list the entities you found in the video and their sentiment [positive, negative, mixed, neutral].\n" \
            "Entities can be person, company, country, concept, brand, terms, or anything where audience may be interested in knowing the trend.\n" \
            "Your answer MUST consist ONLY pairs of entity and sentiment with : as delimiter. Follow the below format.\n\n" \
            "Entities:\n" \
            "mathematic:negative\nteacher:positive\nplaying in classroom:positive\nexamination:mixed\n\n" \
            "DO NOT make up anything you do not know. DO NOT mention about \"video script\" as your audience might not be aware of its presence.\n" \
            "STRICTLY FOLLOW the format above.\n" \
            "Give the entities and sentiment directly without any intro.\n" \
            "Entities:\n"
          
            self.video_rolling_sentiment = self.call_llm(prompt)
        else:
            number_of_chunks = math.ceil( (video_script_length + 1) / (self.video_script_chunk_size - self.video_script_chunk_overlap) )
            
            for chunk_number in range(0, number_of_chunks):
                is_last_chunk = (chunk_number == (number_of_chunks - 1))
                is_first_chunk = (chunk_number == 0)
                start = 0 if is_first_chunk else int(chunk_number*self.video_script_chunk_size - self.video_script_chunk_overlap)
                stop = video_script_length if is_last_chunk else (chunk_number+1)*self.video_script_chunk_size
                chunk_combined_video_script = self.combined_video_script[start:stop]
            
                # So long as this is not the first chunk, remove whatever before first \n since likely the chunk cutting is not done exactly at the \n
                if not is_first_chunk:
                    chunk_combined_video_script = chunk_combined_video_script[chunk_combined_video_script.index("\n"):]
                    
                core_prompt = f"The VIDEO SCRIPT has format below.\n" \
                        "timestamp in seconds:scene / text / voice\n" \
                        "<VideoScript>\n" \
                        f"{chunk_combined_video_script}\n" \
                        "</VideoScript>\n"
                
                if is_last_chunk:
                    prompt = f"{prompt_prefix}\n\n" \
                    f"The video has {number_of_chunks} parts.\n\n" \
                    "Below are the entities and sentiment you extracted from the previous parts of the video, with reasoning.\n\n" \
                    f"Entities:\n{self.video_rolling_sentiment}\n\n" \
                    f"The below VIDEO SCRIPT is only for the LAST video part.\n\n" \
                    f"{core_prompt}\n\n" \
                    "Now your job is to list the entities you found in the video and their sentiment [positive, negative, mixed, neutral]. Also provide the reasoning.\n" \
                    "Entities can be person, company, country, concept, brand, terms, or anything where audience may be interested in knowing the trend.\n" \
                    "Your answer MUST consist ONLY pairs of entity and sentiment with : as delimiter. NO reasoning. Follow the below format.\n\n" \
                    "Entities:\n" \
                    "mathematic:negative\n" \
                    "teacher:positive\n" \
                    "playing in classroom:positive\n" \
                    "examination:mixed\n\n" \
                    "DO NOT make up anything you do not know. DO NOT mention about \"video script\" as your audience might not be aware of its presence.\n" \
                    "DO NOT list the same entity twice. Instead, you can modify your previous extracted entities and sentiment to combine the findings.\n" \
                    "STRICTLY FOLLOW the format above. DELETE any reasoning of the sentiment. DELETE unnecessary information. \n" \
                    "Give the entities and sentiment directly without any intro.\n" \
                    "Entities:\n"
                    
                    chunk_sentiment = self.call_llm(prompt)
                    self.video_rolling_sentiment = chunk_sentiment
                elif is_first_chunk:
                    prompt = f"{prompt_prefix}\n\n" \
                    f"The video has {number_of_chunks} parts. The below VIDEO SCRIPT is only for the first part.\n\n" \
                    f"{core_prompt}\n\n" \
                    "Now your job is to list the entities you found in the video and their sentiment [positive, negative, mixed, neutral]. Also provide the reasoning.\n" \
                    "Entities can be person, company, country, concept, brand, terms, or anything where audience may be interested in knowing the trend.\n" \
                    "Your answer MUST consist ONLY pairs of entity and sentiment with : as delimiter, and the reasoning after - sign. Follow the below format.\n\n" \
                    "Entities:\n" \
                    "mathematic:negative - The kid being interviewed said he dreads math.\n" \
                    "teacher:positive - The kid really respects his teacher as the teacher is patient to him.\n" \
                    "playing in classroom:positive - The kid also likes to play in the classroom, the only thing motivates him to go to school.\n" \
                    "examination:mixed - The kid dreads examination for fear of bad result, but he enjoys the challenge.\n\n" \
                    "DO NOT make up anything you do not know.\n" \
                    "DO NOT list the same entity twice. Instead, you can modify your previous extracted entities and sentiment to combine the findings.\n" \
                    "STRICTLY FOLLOW the format above.\n" \
                    "Entities:\n"
                    
                    chunk_sentiment = self.call_llm(prompt)
                    self.video_rolling_sentiment = chunk_sentiment
                else:
                    prompt = f"{prompt_prefix}\n\n" \
                    f"The video has {number_of_chunks} parts.\n\n" \
                    "Below are the entities and sentiment you extracted from the previous parts of the video, with reasoning.\n\n" \
                    f"Entities:\n{self.video_rolling_sentiment}\n\n" \
                    f"The below VIDEO SCRIPT is only for part {chunk_number} of the video.\n\n" \
                    f"{core_prompt}\n\n" \
                    "Now your job is to list the entities you found in the whole video so far and their sentiment [positive, negative, mixed, neutral]. Also provide the reasoning.\n" \
                    "Entities can be person, company, country, concept, brand, terms, or anything where audience may be interested in knowing the trend.\n" \
                    "Your answer MUST consist ONLY pairs of entity and sentiment with : as delimiter, and the reasoning after - sign. Follow the below format.\n\n" \
                    "Entities:\n" \
                    "mathematic:negative - The kid being interviewed said he dreads math.\n" \
                    "teacher:positive - The kid really respects his teacher as the teacher is patient to him.\n" \
                    "playing in classroom:positive - The kid also likes to play in the classroom, the only thing motivates him to go to school.\n" \
                    "examination:mixed - The kid dreads examination for fear of bad result, but he enjoys the challenge.\n\n" \
                    "DO NOT make up anything you do not know.\n" \
                    "DO NOT list the same entity twice. Instead, you can modify your previous extracted entities and sentiment to combine the findings.\n" \
                    "STRICTLY FOLLOW the format above.\n" \
                    "Entities:\n"
                    
                    chunk_sentiment = self.call_llm(prompt)
                    self.video_rolling_sentiment = chunk_sentiment
        print("video rolling sentiment")
        print(self.video_rolling_sentiment)
                
        return self.video_rolling_sentiment
      
    def run(self):
        self.preprocess_scenes()
        self.preprocess_visual_texts()
        self.preprocess_transcript()
        self.generate_combined_video_script()

        summary = self.generate_summary()
        sentiment = self.extract_sentiment()
        video_script = self.all_combined_video_script
        
        return {
            'summary': summary,
            'video_script': video_script,
            'sentiment': sentiment
        }

class VideoAnalyzerBedrock(VideoAnalyzer):    
    def __init__(self, endpoint_name, video_length, video_scenes, video_texts, transcript):
        super().__init__(video_length, video_scenes, video_texts, transcript)
        self.endpoint_name = endpoint_name
        self.llm_parameters = {
        "temperature": 0.7,
        "top_k": 10,
        "max_tokens_to_sample": 2000,
        "stop_sequences": ["\n\nHuman:"]
    }
  
    def call_llm(self, prompt, temperature=None, top_k=None, stop_sequences=[]):
        print("Prompt to LLM")
        print(prompt.replace("\n",""))

        self.llm_parameters['prompt'] = f"\n\nHuman:{prompt}\n\nAssistant:"
        if temperature is not None:
            self.llm_parameters['temperature'] = temperature
        if top_k is not None:
            self.llm_parameters['top_k'] = top_k
        if stop_sequences is not None:
            self.llm_parameters['stop_sequences'] += stop_sequences

        input_str = json.dumps(self.llm_parameters)
        encoded_input = input_str.encode("utf-8")
        call_done = False
        while(not call_done):
            try:
                bedrock_response = bedrock.invoke_model(body=encoded_input, modelId=self.endpoint_name)
                call_done = True
            except bedrock.exceptions.ThrottlingException as e:
                print("Amazon Bedrock throttling exception")
                time.sleep(60)
            except Exception as e:
                raise e

        response = json.loads(bedrock_response.get("body").read())["completion"]
        print("Response from LLM")
        print(response)
        return response

# Legacy
class VideoAnalyzerSageMaker(VideoAnalyzer):
    def __init__(self, endpoint_name, video_length, video_scenes, video_texts, transcript):
        super().__init__(video_length, video_scenes, video_texts, transcript)
        self.endpoint_name = endpoint_name
        self.llm_parameters = {
            "max_new_tokens": 500,
            "num_return_sequences": 1,
            "do_sample": False,
            "return_full_text": False,
            "temperature": 0.8,
            "stop": ["\n\n\n"]
        }

    def query_endpoint_with_json_payload(self, encoded_json, endpoint_name, content_type="application/json"):
        response = sagemaker.invoke_endpoint(
            EndpointName=endpoint_name, ContentType=content_type, Body=encoded_json
        )
        return response

    def parse_response_model(self, query_response):
        model_predictions = json.loads(query_response["Body"].read())
        return [gen["generated_text"] for gen in model_predictions]
  
    def call_llm(self, prompt):
        input_str = json.dumps({"inputs": prompt, "parameters": self.llm_parameters})
        encoded_input = input_str.encode("utf-8")

        response = self.query_endpoint_with_json_payload(encoded_input, endpoint_name, content_type="application/json")
        response = self.parse_response_model(response)[0]
        return response


if __name__ == "__main__":
    print("called")
    handler()
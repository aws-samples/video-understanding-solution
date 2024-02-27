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
video_script_storage_chunk_size = 512 #10000 # characters

config = Config(read_timeout=1000) # Extends botocore read timeout to 1000 seconds

rekognition = boto3.client('rekognition')
transcribe = boto3.client("transcribe")
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

def handler():
    video_name = os.path.basename(video_s3_path)
    video_path= '/'.join(video_s3_path.split('/')[1:])

    video_transcript_s3_path = f"{transcription_folder}/{video_path}.txt"

    try:
        wait_for_rekognition_label_detection(labels_job_id=labels_job_id, sort_by='TIMESTAMP')
        wait_for_rekognition_text_detection(texts_job_id=texts_job_id)
        wait_for_transcription_job(transcription_job_name=transcription_job_name)
        response = analyze_video(labels_job_id=labels_job_id, texts_job_id=texts_job_id, video_transcript_s3_path=video_transcript_s3_path)
        store_summary_result(summary=response['summary'], video_name=video_name, video_path=video_path)
        store_sentiment_result(sentiment_string=response['sentiment'], video_name=video_name, video_path=video_path)
        store_video_script_result(video_script=response['video_script'], video_name=video_name, video_path=video_path)
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
    body = json.dumps({
        "texts":[summary],
        "input_type": "search_document",
    })
    response = bedrock.invoke_model(body=body, modelId=embedding_model_id)
    # Disabling semgrep rule for checking data size to be loaded to JSON as the source is from Amazon Bedrock
    # nosemgrep: python.aws-lambda.deserialization.tainted-json-aws-lambda.tainted-json-aws-lambda
    summary_embedding = json.loads(response.get("body").read().decode())["embeddings"][0] #["embedding"]

    # Store summary in database
    update_stmt = (
        update(Videos).
        where(Videos.name == video_path).
        values(summary = summary, summary_embedding = summary_embedding)
    )
    
    session.execute(update_stmt)
    session.commit()

def store_sentiment_result(sentiment_string, video_name, video_path):
    # Extract entities and sentiment from the string
    entities_dict = {}
    entity_regex = r"\n*([^|\n]+?)\|\s*(positive|negative|neutral|mixed|N\/A)\s*\|([^|\n]+?)\n"
    for match in re.finditer(entity_regex, sentiment_string):
        entity = match.group(1).strip()
        sentiment = match.group(2)
        reason = match.group(3)
        entities_dict[entity] = {
            "sentiment": sentiment,
            "reason": reason
        }

    s3.put_object(
        Body="\n".join(f"{e}|{s['sentiment']}|{s['reason']}" for e, s in entities_dict.items()), 
        Bucket=bucket_name, 
        Key=f"{entity_sentiment_folder}/{video_path}.txt"
    )

    entities = []
    for entity, value in entities_dict.items():
        # Store entity in database
        entities.append(Entities(
            name=entity,
            sentiment=value['sentiment'],
            reason=value['reason'],
            video_name=video_path
        ))

    # Store into database
    session.add_all(entities)
    session.commit()
    
def store_video_script_result(video_script, video_name, video_path):
    s3.put_object(
        Body=video_script, 
        Bucket=bucket_name, 
        Key=f"{video_script_folder}/{video_path}.txt"
    )

    # Chunking the video script for storage in DB while converting them to embedding
    video_script_length = len(video_script)
    number_of_chunks = math.ceil( (video_script_length + 1) / video_script_storage_chunk_size )

    chunks = []
    for chunk_number in range(0, number_of_chunks):
        is_last_chunk = (chunk_number == (number_of_chunks - 1))
        is_first_chunk = (chunk_number == 0)

        start = 0 if is_first_chunk else int(chunk_number*video_script_storage_chunk_size)
        stop = video_script_length if is_last_chunk else (chunk_number+1)*video_script_storage_chunk_size
        chunk_string = video_script[start:stop]
    
        # So long as this is not the first chunk, remove whatever before first \n since likely the chunk cutting is not done exactly at the \n
        if not is_first_chunk:
            try:
                chunk_string = chunk_string[chunk_string.index("\n"):] 
            except:
                pass
        
        # Get the embedding for the chunk
        body = json.dumps({
            "texts":[chunk_string],
            "input_type": "search_document",
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
        chunk_embedding = json.loads(response.get("body").read().decode())['embeddings'][0] #["embedding"]
        
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
        getObjectDetection = rekognition.get_label_detection(JobId=labels_job_id, SortBy=sort_by)

def wait_for_rekognition_text_detection(texts_job_id):
    getTextDetection = rekognition.get_text_detection(JobId=texts_job_id)
    while(getTextDetection['JobStatus'] == 'IN_PROGRESS'):
        time.sleep(5)
        getTextDetection = rekognition.get_text_detection(JobId=texts_job_id)

def wait_for_transcription_job(transcription_job_name):
    getTranscription = transcribe.get_transcription_job(TranscriptionJobName=transcription_job_name)
    while(getTranscription ["TranscriptionJob"]["TranscriptionJobStatus"] == 'IN_PROGRESS'):
        time.sleep(5)
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
    sentiment = Column(String(20))
    reason = Column(Text)
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
        self.video_script_chunk_size_for_summary_generation = 100000 # characters
        self.video_script_chunk_overlap_for_summary_generation = 500 # characters
        self.video_script_chunk_size_for_entities_extraction = 50000 #10000 # characters
        self.video_script_chunk_overlap_for_entities_extraction = 500 #200 # characters
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
        prompt_prefix = "You are an expert video analyst who reads a Video Timeline and creates summary of the video.\n" \
                        "The Video Timeline is a text representation of a video.\n" \
                        "The Video Timeline contains the visual scenes, the visual texts, and human voice in the video.\n" \
                        "Visual scenes (scene) represents what objects are visible in the video at that second. This can be the objects seen in camera, or objects from a screen sharing, or any other visual scenarios. You can infer how the visualization look like, so long as you are confident.\n" \
                        "Visual texts (text) are the text visible in the video. It can be texts in real world objects as recorded in video camera, or those from screen sharing, or those from presentation recording, or those from news, movies, or others. \n" \
                        "Human voice (voice) is the transcription of the video.\n"
        
        video_script_length = len(self.combined_video_script)

        # When the video is short enough to fit into 1 chunk
        if video_script_length <= self.video_script_chunk_size_for_summary_generation:
            core_prompt = f"The VIDEO TIMELINE has format below.\n" \
                            "timestamp in seconds:scene / text / voice\n" \
                            "<Video Timeline>\n" \
                            f"{self.combined_video_script}\n" \
                            "</Video Timeline>\n"
          
            prompt = f"{prompt_prefix}\n\n" \
            f"{core_prompt}\n\n" \
            "<Task>\n" \
            "Describe the summary of the video in paragraph format.\n" \
            "You can make reasonable extrapolation of the actual video given the Video Timeline.\n" \
            "DO NOT mention 'Video Timeline' or 'video timeline'.\n" \
            "Give the summary directly without any other sentence.\n" \
            "</Task>\n" \
            "Summary:\n"
          
            self.video_rolling_summary = self.call_llm(prompt, stop_sequences=["<Task>"])
        # When the video is long enough to be divided into multiple chunks to fit within LLM's context length
        else:
            number_of_chunks = math.ceil( (video_script_length + 1) / (self.video_script_chunk_size_for_summary_generation - self.video_script_chunk_overlap_for_summary_generation) )

            for chunk_number in range(0, number_of_chunks):
                is_last_chunk = (chunk_number == (number_of_chunks - 1))
                is_first_chunk = (chunk_number == 0)

                start = 0 if is_first_chunk else int(chunk_number*self.video_script_chunk_size_for_summary_generation - self.video_script_chunk_overlap_for_summary_generation)
                stop = video_script_length if is_last_chunk else (chunk_number+1)*self.video_script_chunk_size_for_summary_generation
                chunk_combined_video_script = self.combined_video_script[start:stop]
            
                # So long as this is not the first chunk, remove whatever before first \n since likely the chunk cutting is not done exactly at the \n
                if not is_first_chunk:
                    try:
                        chunk_combined_video_script = chunk_combined_video_script[chunk_combined_video_script.index("\n"):]
                    except:
                        pass
                    
                core_prompt = f"The VIDEO TIMELINE has format below.\n" \
                        "timestamp in seconds:scene / text / voice\n" \
                        "<Video Timeline>\n" \
                        f"{chunk_combined_video_script}\n" \
                        "</Video Timeline>\n"
                
                if is_last_chunk:
                    prompt = f"{prompt_prefix}\n\n" \
                    f"The video has {number_of_chunks} parts.\n\n" \
                    f"The below Video Timeline is only for the part {chunk_number+1} of the video, which is the LAST part.\n\n" \
                    f"{core_prompt}\n\n" \
                    "Below is the summary of all previous part/s of the video:\n\n" \
                    f"{self.video_rolling_summary}\n\n" \
                    "<Task>\n" \
                    "Describe the summary of the whole video in paragraph format.\n" \
                    "You can make reasonable extrapolation of the actual video given the Video Timeline.\n" \
                    "DO NOT mention 'Video Timeline' or 'video timeline'.\n" \
                    "Give the summary directly without any other sentence.\n" \
                    "</Task>\n" \
                    "Summary:\n"
                    
                    chunk_summary = self.call_llm(prompt, stop_sequences=["<Task>"])
                    self.video_rolling_summary = chunk_summary
                elif is_first_chunk:
                    prompt = f"{prompt_prefix}\n\n" \
                    f"The video has {number_of_chunks} parts. The below Video Timeline is only for the first part.\n\n" \
                    f"{core_prompt}\n\n" \
                    "<Task>\n" \
                    "Describe the summary of the first part of the video in paragraph format.\n" \
                    "You can make reasonable extrapolation of the actual video given the Video Timeline.\n" \
                    "DO NOT mention 'Video Timeline' or 'video timeline'.\n" \
                    "Give the summary directly without any other sentence.\n" \
                    "</Task>\n" \
                    "Summary:\n"
                    
                    chunk_summary = self.call_llm(prompt, stop_sequences=["<Task>"])
                    self.video_rolling_summary = chunk_summary
                else:
                    prompt = f"{prompt_prefix}\n\n" \
                    f"The video has {number_of_chunks} parts.\n\n" \
                    f"The below Video Timeline is only for part {chunk_number+1} of the video.\n\n" \
                    f"{core_prompt}\n\n" \
                    "Below is the summary of all previous part/s of the video:\n\n" \
                    f"{self.video_rolling_summary}\n\n" \
                    "<Task>\n" \
                    "Describe the summary of the video so far in paragraph format.\n" \
                    "In your summary, retain important details from the previous part/s summaries. Your summary MUST include the summary of all parts of the video so far.\n"\
                    "You can make reasonable extrapolation of the actual video given the Video Timeline.\n" \
                    "DO NOT mention 'Video Timeline' or 'video timeline'.\n" \
                    "Give the summary directly without any other sentence.\n" \
                    "</Task>\n" \
                    "Summary:\n"
                    
                    chunk_summary = self.call_llm(prompt, stop_sequences=["<Task>"])
                    self.video_rolling_summary = chunk_summary

        return self.video_rolling_summary
    
    def extract_sentiment(self):
        prompt_prefix = "You are an expert video analyst who reads a Video Timeline and extract entities and their associated sentiment.\n" \
                        "The Video Timeline is a text representation of a video.\n" \
                        "The Video Timeline contains the visual scenes, the visual texts, and human voice in the video.\n" \
                        "Visual scenes (scene) represents what objects are visible in the video at that second. This can be the objects seen in camera, or objects from a screen sharing, or any other visual scenarios. You can infer how the visualization look like, so long as you are confident.\n" \
                        "Visual texts (text) are the text visible in the video. It can be texts in real world objects as recorded in video camera, or those from screen sharing, or those from presentation recording, or those from news, movies, or others. \n" \
                        "Human voice (voice) is the transcription of the video.\n"
                        
        
        video_script_length = len(self.combined_video_script)

        # When the video is short enough to fit into 1 chunk
        if video_script_length <= self.video_script_chunk_size_for_entities_extraction:
            core_prompt = f"The Video Timeline has a format below.\n" \
                            "timestamp in seconds:scene / text / voice\n" \
                            "<Video Timeline>\n" \
                            f"{self.combined_video_script}\n" \
                            "</Video Timeline>\n"
          
            prompt = f"{prompt_prefix}\n\n" \
            f"{core_prompt}\n\n" \
            f"To help you, here is the summary of the whole video.\n\n{self.video_rolling_summary}\n\n" \
            "<Task>\n" \
            "Now your job is to infer and list the entities, their sentiment [positive, negative, mixed, neutral], and the sentiment's reason.\n" \
            "You will never see the video. This video timeline is the best you get. You can make reasonable extrapolation of the actual video given the Video Timeline.\n" \
            "Entities can be a person, company, country, concept, brand, or anything where audience may be interested in knowing the trend.\n" \
            "You MUST ONLY list important entities of interest, not every entity you found.\n" \
            "For person or individual, DO NOT give sentiment rating. Put N/A for the sentiment and reason fields.\n" \
            "Sentiment's reason MUST justify the sentiment. For no meaningful reason, just put N/A for the sentiment's reason.\n" \
            "Each row of your answer MUST be of this format entity|sentiment|sentiment's reason. Follow the below example.\n\n" \
            "Entities:\n" \
            "mathematic|negative|The kid interviewed in the video seems to be really afraid of mathematics as his grade is always struggling.\n" \
            "Rudy Donna|N/A|Rudy Donna is the interviewee's friend. No sentiment score is provided for person/individual.\n" \
            "teacher|positive|Despite the kid having not so good grades, he seems to like all his teachers in school as they are all patient.\n" \
            "extracurricular activities|neutral|N/A.\n" \
            "examination|mixed|While the interviewee dreads examination, he likes the challenge of it.\n\n" \
            "STRICTLY FOLLOW the format above. DO NOT mention Video Timeline or video timeline.\n" \
            "</Task>\n" \
            "Entities:\n"
          
            self.video_rolling_sentiment = self.call_llm(prompt, temperature=0.8, stop_sequences=["<Task>"])
        else:
            number_of_chunks = math.ceil( (video_script_length + 1) / (self.video_script_chunk_size_for_entities_extraction - self.video_script_chunk_overlap_for_entities_extraction) )
            
            for chunk_number in range(0, number_of_chunks):
                is_last_chunk = (chunk_number == (number_of_chunks - 1))
                is_first_chunk = (chunk_number == 0)
                start = 0 if is_first_chunk else int(chunk_number*self.video_script_chunk_size_for_entities_extraction - self.video_script_chunk_overlap_for_entities_extraction)
                stop = video_script_length if is_last_chunk else (chunk_number+1)*self.video_script_chunk_size_for_entities_extraction
                chunk_combined_video_script = self.combined_video_script[start:stop]
            
                # So long as this is not the first chunk, remove whatever before first \n since likely the chunk cutting is not done exactly at the \n
                if not is_first_chunk:
                    try:
                        chunk_combined_video_script = chunk_combined_video_script[chunk_combined_video_script.index("\n"):]
                    except:
                        pass
                    
                core_prompt = f"The VIDEO TIMELINE has format below.\n" \
                        "timestamp in seconds:scene / text / voice\n" \
                        "<Video Timeline>\n" \
                        f"{chunk_combined_video_script}\n" \
                        "</Video Timeline>\n"
                
                if is_last_chunk:
                    prompt = f"{prompt_prefix}\n\n" \
                    f"The video has {number_of_chunks} parts. The below Video Timeline is only for part {chunk_number+1} of the video, which is the LAST part.\n\n" \
                    f"{core_prompt}\n\n" \
                    "Below are the entities, sentiment, and reason you extracted from the previous part/s of the video.\n\n" \
                    f"Entities:\n{self.video_rolling_sentiment}\n\n" \
                    f"To help you, here is the summary of the whole video.\n\n{self.video_rolling_summary}\n\n" \
                    "<Task>\n" \
                    "Now your job is to infer and list the entities, their sentiment [positive, negative, mixed, neutral], and the sentiment's reason.\n" \
                    "You will never see the video. This video timeline is the best you get. You can make reasonable extrapolation of the actual video given the Video Timeline.\n" \
                    "Entities can be a person, company, country, concept, brand, or anything where audience may be interested in knowing the trend.\n" \
                    "You MUST ONLY list important entities of interest, not every entity you found.\n" \
                    "For person or individual, DO NOT give sentiment rating. Put N/A for the sentiment and reason fields.\n" \
                    "Sentiment's reason MUST justify the sentiment. For no meaningful reason, just put N/A for the sentiment's reason.\n" \
                    "Each row of your answer MUST be of this format entity|sentiment|sentiment's reason. Follow the below example.\n\n" \
                    "Entities:\n" \
                    "mathematic|negative|The kid interviewed in the video seems to be really afraid of mathematics as his grade is always struggling.\n" \
                    "Rudy Donna|N/A|Rudy Donna is the interviewee's friend. No sentiment score is provided for person/individual.\n" \
                    "teacher|positive|Despite the kid having not so good grades, he seems to like all his teachers in school as they are all patient.\n" \
                    "extracurricular activities|neutral|N/A.\n" \
                    "examination|mixed|While the interviewee dreads examination, he likes the challenge of it.\n\n" \
                    "STRICTLY FOLLOW the format above. DO NOT mention Video Timeline or video timeline.\n" \
                    "Your answer should include entities found from previous part/s as well. DO NOT duplicate entities.\n" \
                    "</Task>\n" \
                    "Entities:\n"
                    
                    chunk_sentiment = self.call_llm(prompt, temperature=0.8, stop_sequences=["<Task>"])
                    self.video_rolling_sentiment = chunk_sentiment
                elif is_first_chunk:
                    prompt = f"{prompt_prefix}\n\n" \
                    f"The video has {number_of_chunks} parts. The below Video Timeline is only for the first part.\n\n" \
                    f"{core_prompt}\n\n" \
                    f"To help you, here is the summary of the whole video.\n\n{self.video_rolling_summary}\n\n" \
                    "<Task>\n" \
                    "Now your job is to infer and list the entities, their sentiment [positive, negative, mixed, neutral], and the sentiment's reason.\n" \
                    "You will never see the video. This video timeline is the best you get. You can make reasonable extrapolation of the actual video given the Video Timeline.\n" \
                    "Entities can be a person, company, country, concept, brand, or anything where audience may be interested in knowing the trend.\n" \
                    "You MUST ONLY list important entities of interest, not every entity you found.\n" \
                    "For person or individual, DO NOT give sentiment rating. Put N/A for the sentiment and reason fields.\n" \
                    "Sentiment's reason MUST justify the sentiment. For no meaningful reason, just put N/A for the sentiment's reason.\n" \
                    "Each row of your answer MUST be of this format entity|sentiment|sentiment's reason. Follow the below example.\n\n" \
                    "Entities:\n" \
                    "mathematic|negative|The kid interviewed in the video seems to be really afraid of mathematics as his grade is always struggling.\n" \
                    "Rudy Donna|N/A|Rudy Donna is the interviewee's friend. No sentiment score is provided for person/individual.\n" \
                    "teacher|positive|Despite the kid having not so good grades, he seems to like all his teachers in school as they are all patient.\n" \
                    "extracurricular activities|neutral|N/A.\n" \
                    "examination|mixed|While the interviewee dreads examination, he likes the challenge of it.\n\n" \
                    "STRICTLY FOLLOW the format above. DO NOT mention Video Timeline or video timeline.\n" \
                    "I know you need all parts of the video to come with the entity list. Therefore, for now you only need to give an INTERMEDIATE results by listing the entities you have in this first part of the video.\n" \
                    "</Task>\n" \
                    "Entities:\n"
                    
                    chunk_sentiment = self.call_llm(prompt, temperature=0.8, stop_sequences=["<Task>"])
                    self.video_rolling_sentiment = chunk_sentiment
                else:
                    prompt = f"{prompt_prefix}\n\n" \
                    f"The video has {number_of_chunks} parts. The below Video Timeline is only for part {chunk_number+1} of the video.\n\n" \
                    f"{core_prompt}\n\n" \
                    "Below are the entities, sentiment, and reason you extracted from the previous part/s of the video.\n\n" \
                    f"Entities:\n{self.video_rolling_sentiment}\n\n" \
                    f"To help you, here is the summary of the whole video.\n\n{self.video_rolling_summary}\n\n" \
                    "<Task>\n" \
                    "Now your job is to infer and list the entities, their sentiment [positive, negative, mixed, neutral], and the sentiment's reason.\n" \
                    "You will never see the video. This video timeline is the best you get. You can make reasonable extrapolation of the actual video given the Video Timeline.\n" \
                    "Entities can be a person, company, country, concept, brand, or anything where audience may be interested in knowing the trend.\n" \
                    "You MUST ONLY list important entities of interest, not every entity you found.\n" \
                    "For person or individual, DO NOT give sentiment rating. Put N/A for the sentiment and reason fields.\n" \
                    "Sentiment's reason MUST justify the sentiment. For no meaningful reason, just put N/A for the sentiment's reason.\n" \
                    "Each row of your answer MUST be of this format entity|sentiment|sentiment's reason. Follow the below example.\n\n" \
                    "Entities:\n" \
                    "mathematic|negative|The kid interviewed in the video seems to be really afraid of mathematics as his grade is always struggling.\n" \
                    "Rudy Donna|N/A|Rudy Donna is the interviewee's friend. No sentiment score is provided for person/individual.\n" \
                    "teacher|positive|Despite the kid having not so good grades, he seems to like all his teachers in school as they are all patient.\n" \
                    "extracurricular activities|neutral|N/A.\n" \
                    "examination|mixed|While the interviewee dreads examination, he likes the challenge of it.\n\n" \
                    "STRICTLY FOLLOW the format above. DO NOT mention Video Timeline or video timeline.\n" \
                    "I know you need all parts of the video to come with the entity list. Therefore, for now you only need to give an INTERMEDIATE results by listing the entities you have so far. DO NOT duplicate entities.\n" \
                    "</Task>\n" \
                    "Entities:\n"
                    
                    chunk_sentiment = self.call_llm(prompt, temperature=0.8, stop_sequences=["<Task>"])
                    self.video_rolling_sentiment = chunk_sentiment
                
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
        return response


if __name__ == "__main__":
    handler()
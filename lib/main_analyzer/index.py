import os, time, json, copy, math, re, io
from abc import ABC, abstractmethod
import boto3, botocore
from botocore.config import Config
from sqlalchemy import create_engine, Column, DateTime, String, Text, Integer, func, ForeignKey, update
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import mapped_column, sessionmaker
from pgvector.sqlalchemy import Vector
from typing import Union, Self
import cv2
import base64
from PIL import Image

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
        # Initiate class for video preprocessing
        video_preprocessor = VideoPreprocessor( 
            labels_job_id=labels_job_id, 
            texts_job_id=texts_job_id, 
            transcription_job_name=transcription_job_name,
            bucket_name=bucket_name,
            video_s3_path=video_s3_path,
            video_transcript_s3_path=video_transcript_s3_path
        )
        # Wait for extraction jobs to finish
        video_preprocessor.wait_for_dependencies()
        
        # Preprocess and extract information
        visual_objects, visual_texts, transcript, celebrities, faces = video_preprocessor.run()
        
        # Initiate class for video analysis
        video_analyzer = VideoAnalyzerBedrock(
            model_name=model_id, 
            embedding_model_name=embedding_model_id,
            bucket_name=bucket_name,
            video_name=video_name,
            video_path=video_path,
            visual_objects=visual_objects,
            visual_texts=visual_texts, 
            transcript=transcript,
            celebrities=celebrities,
            faces=faces,
            summary_folder=summary_folder,
            entity_sentiment_folder=entity_sentiment_folder,
            video_script_folder=video_script_folder
        )
        # Run video analysis
        video_analyzer.run()

        # Store results to S3 and database
        video_analyzer.store()

    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise

    return {
        'statusCode': 200,
        'body': json.dumps({"main_analyzer": "success"})
    }

class CelebrityFinding():
    celebrity_match_confidence_threshold: int = 97
    face_bounding_box_overlap_threshold: float = 0.1
    celebrity_emotion_confidence_threshold: int = 85
    celebrity_feature_confidence_threshold: int = 90

    def __init__(self, celebrity_dict: dict):
        self.name: str = celebrity_dict['Name']
        self.match_confidence: int = int(celebrity_dict['MatchConfidence'])
        self.smile: bool = bool(celebrity_dict['Face']['Smile']['Value'])
        self.smile_confidence: int = int(celebrity_dict['Face']['Smile']['Confidence'])
        self.emotions: list[str] = filter(lambda em: (len(em) > 0),[emo['Type'].lower() if int(emo['Confidence']) >= celebrity_emotion_confidence_threshold else '' for emo in celebrity_dict['Face']['Emotions']])
        self.top: float = float(celebrity_dict['Face']['BoundingBox']['Top'])
        self.left: float = float(celebrity_dict['Face']['BoundingBox']['Left'])
        self.height: float = float(celebrity_dict['Face']['BoundingBox']['Height'])
        self.width: float = float(celebrity_dict['Face']['BoundingBox']['Width'])

    def is_matching_face(self, bb_top:float , bb_left: float, bb_height: float, bb_width: float) -> bool:
        return (abs(self.top - bb_top) <= face_bounding_box_overlap_threshold) and
        (abs(self.left - bb_left) <= face_bounding_box_overlap_threshold) and
        (abs(self.height - bb_height) <= face_bounding_box_overlap_threshold) and
        (abs(self.width - bb_width) <= face_bounding_box_overlap_threshold)

    def display(self) -> str:
        display_string = self.name
        if len(self.emotions) > 0:
            display_string += "|" + "-".join(self.emotions)
        if self.smile_confidence >= celebrity_feature_confidence_threshold: 
            display_string += "|smiling" if self.smile else "|not smiling"
        return display_string

class FaceFinding():
    face_detection_confidence_threshold: int = 97
    face_feature_confidence_threshold: int = 97
    face_age_range_match_threshold: float = 3
    face_emotion_confidence_threshold: int = 85

    def __init__(self, face_dict: dict):
        self.confidence: int = int(face_dict['Confidence'])
        self.age_low: int = int(face_dict['AgeRange']['Low'])
        self.age_high: int = int(face_dict['AgeRange']['High'])
        self.top: float = float(face_dict['FaceDetails']['BoundingBox']['Top'])
        self.top_display: str = str(round(self.top, 2))
        self.left: float = float(face_dict['FaceDetails']['BoundingBox']['Left'])
        self.left_display: str = str(round(self.left, 2))
        self.height: float = float(face_dict['FaceDetails']['BoundingBox']['Height'])
        self.height_display: str = str(round(self.height, 2))
        self.width: float = float(face_dict['FaceDetails']['BoundingBox']['Width'])
        self.width_display: str = str(round(self.width, 2))
        self.beard: bool = bool(face_dict['Beard']['Value'])
        self.beard: int = int(face_dict['Beard']['Confidence'])
        self.eyeglasses: bool = bool(face_dict['Eyeglasses']['Value'])
        self.eyeglasses_confidence: int = int(face_dict['Eyeglasses']['Confidence'])
        self.eyesopen: bool = bool(face_dict['EyesOpen']['Value'])
        self.eyesopen_confidence: int = int(face_dict['EyesOpen']['Confidence'])
        self.sunglasses: bool = bool(face_dict['Sunglasses']['Value'])
        self.sunglasses_confidence: int = int(face_dict['Sunglasses']['Confidence'])
        self.smile: bool = bool(face_dict['Smile']['Value'])
        self.smile_confidence: int = int(face_dict['Smile']['Confidence'])
        self.mouthopen: bool = bool(face_dict['MouthOpen']['Value'])
        self.mouthopen_confidence: int = int(face_dict['MouthOpen']['Confidence'])
        self.mustache: bool = bool(face_dict['Mustache']['Value'])
        self.mustache_confidence: int = int(face_dict['Mustache']['Confidence'])
        self.gender: str = str(face_dict['Gender']['Value']).lower()
        self.gender_confidence: int = int(face_dict['Gender']['Confidence'])
        self.emotions: list[str] = filter(lambda em: (len(em) > 0),[emo['Type'].lower() if int(emo['Confidence']) >= face_emotion_confidence_threshold else '' for emo in face_dict['Emotions']])

        def is_duplicate(self, face_list: list[Self]) -> bool:
            found = False
            face_finding: Self
            for face_finding in face_list:
                if (abs(self.age_low - face_finding.age_low) <= face_age_range_match_threshold) and (abs(self.age_high - face_finding.age_high) <= face_age_range_match_threshold): found = True
            return found

        def display(self) -> str:
            display_string = f"{self.age_low}-{self.age_high} years old"
            if self.gender_confidence >= face_feature_confidence_threshold:
                display_string += f"|{self.gender}"
            if len(self.emotions) > 0:
                display_string += "|" + "-".join(self.emotions)
            if self.smile_confidence >= face_feature_confidence_threshold:
                display_string += "|smiling" if self.smile else "|not smiling"
            if self.beard_confidence >= face_feature_confidence_threshold:
                display_string += "|has beard" if self.beard else "|no beard"
            if self.mustache_confidence >= face_feature_confidence_threshold:
                display_string += "|has mustache" if self.mustache else "|no mustache"
            if self.sunglasses_confidence >= face_feature_confidence_threshold:
                display_string += "|wears sunglasses" if self.sunglasses else "|no sunglasses"
            if self.eyeglasses_confidence >= face_feature_confidence_threshold:
                display_string += "|wears eyeglasses" if self.eyeglasses else "|no eyeglasses"
            if self.mouthopen_confidence >= face_feature_confidence_threshold:
                display_string += "|mouth is open" if self.mouthopen else "|mouth is closed"
            if self.eyesopen_confidence >= face_feature_confidence_threshold:
                display_string += "|eyes are open" if self.eyesopen else "|eyes is closed"
            display_string += f"|face is located {self.left_display} from left - {self.top_display} from top - with height {self.height_display} and width {self.width_display} of the video frame"
            
            return display_string

class VideoPreprocessor(ABC):
    def __init__(self, 
        labels_job_id: str, 
        texts_job_id: str,
        transcription_job_name: str,
        bucket_name: str,
        video_s3_path: str,
        video_transcript_s3_path: str):

        self.rekognition = boto3.client("rekognition")
        self.s3 = boto3.client("s3")
        self.transcribe = boto3.client("transcribe")
        self.labels_job_id: str = labels_job_id
        self.texts_job_id: str = texts_job_id
        self.transcription_job_name: str = transcription_job_name
        self.bucket_name: str = bucket_name
        self.video_s3_path: str = video_s3_path
        self.video_transcript_s3_path: str = video_transcript_s3_path
        self.video_duration_seconds: float  = 0.0
        self.video_duration_millis: int = 0
        self.visual_objects: dict[int, list[str]] = {}
        self.visual_texts: dict[int, list[str]] = {}
        self.transcript: dict
        self.celebrities: dict[int, list[str]] = {}
        self.faces: dict[int, list[str]] = {}
        self.person_timestamps_millis: list[int] = []
        self.maximum_number_of_objects_per_timestamp_second: int = 50
        self.maximum_number_of_texts_per_timestamp_second: int = 50

    def wait_for_rekognition_label_detection(self, sort_by):
        get_object_detection = self.rekognition.get_label_detection(JobId=self.labels_job_id, SortBy=sort_by)
        while(get_object_detection['JobStatus'] == 'IN_PROGRESS'):
            time.sleep(5)
            get_object_detection = self.rekognition.get_label_detection(JobId=self.labels_job_id, SortBy=sort_by)

    def wait_for_rekognition_text_detection(self):
        get_text_detection = rekognition.get_text_detection(JobId=self.texts_job_id)
        while(get_text_detection['JobStatus'] == 'IN_PROGRESS'):
            time.sleep(5)
            get_text_detection = self.rekognition.get_text_detection(JobId=self.texts_job_id)

    def wait_for_transcription_job(self):
        get_transcription = self.transcribe.get_transcription_job(TranscriptionJobName=self.transcription_job_name)
        while(get_transcription["TranscriptionJob"]["TranscriptionJobStatus"] == 'IN_PROGRESS'):
            time.sleep(5)
            get_transcription = self.transcribe.get_transcription_job(TranscriptionJobName=self.transcription_job_name)
    
    def extract_visual_objects(self, get_object_detection_result: dict):
        label: dict
        for label in get_object_detection_result['Labels']:
            objects_at_this_timestamp: list = []
            timestamp_millis:int = int(label['Timestamp'])
            timestamp_second: int = int(timestamp_millis/1000)

            # If this timestamp second is already in self.visual_objects dictionary, then use it and append. Otherwise, add a new key to the dict.
            if timestamp_second in self.visual_objects:
                objects_at_this_timestamp = self.visual_objects[timestamp_second]
            else:
                self.visual_objects[timestamp_second] = objects_at_this_timestamp

            object_name: str = label['Label']['Name']

            # If this is a Face object, then register this into the list of timetamps
            if object_name == "Person":
                self.person_timestamps_millis.append(timestamp_millis)

            # To avoid having too many detected visual objects, cut the detected object per frame to a certain number (default = 25)
            if (object_name not in objects_at_this_timestamp) and (len(objects_at_this_timestamp) <= self.maximum_number_of_objects_per_timestamp_second):
                # Append the object name to the list of objects for timestamp second
                objects_at_this_timestamp.append(object_name)
  
    def iterate_object_detection_result(self):
        get_object_detection_result: dict = self.rekognition.get_label_detection(
            JobId=self.labels_job_id,
            MaxResults=1000,
            SortBy='TIMESTAMP'
        )
        self.video_duration_millis = float(get_object_detection_result["VideoMetadata"]["DurationMillis"])
        self.video_duration_seconds = self.video_duration_millis/1000

        # Extract visual scenes and populate self.visual_objects
        self.extract_visual_objects(get_object_detection_result)

        # In case results is large, iterate the next pages until no more page left.
        while("NextToken" in get_object_detection_result):
            get_object_detection_result: dict = self.rekognition.get_label_detection(
                JobId=self.labels_job_id,
                MaxResults=1000,
                NextToken=get_object_detection_result["NextToken"]
            )
            self.extract_visual_objects(get_object_detection_result)

    def extract_visual_texts(self, get_text_detection_result: dict):
        text_detection: dict
        for text_detection in get_text_detection_result['TextDetections']:
            if text_detection['TextDetection']["Type"] == "WORD": continue
            
            texts_at_this_timestamp: list = []
            timestamp_second = int(int(text_detection['Timestamp'])/1000)

            # If this timestamp second is already in self.visual_texts dictionary, then use it and append. Otherwise, add a new key to the dict.
            if timestamp_second in self.visual_texts:
                texts_at_this_timestamp = self.visual_texts[timestamp_second]
            else:
                self.visual_texts[timestamp_second] = texts_at_this_timestamp

            text: str = text_detection['TextDetection']['DetectedText']

            # To avoid having too many detected text, cut the detected text per frame to a certain number (default = 25)
            if len(texts_at_this_timestamp) <= self.maximum_number_of_texts_per_timestamp_second: 
                texts_at_this_timestamp.append(text)

    def iterate_text_detection_result(self):
        get_text_detection_result: dict = self.rekognition.get_text_detection(
            JobId=self.texts_job_id,
            MaxResults=1000
        )

        # Extract visual texts and populate self.visual_texts
        self.extract_visual_texts(get_text_detection_result)

        # In case results is large, iterate the next pages until no more page left.
        while("NextToken" in get_text_detection_result):
            get_text_detection_result = self.rekognition.get_text_detection(
                JobId=self.texts_job_id,
                MaxResults=1000,
                NextToken=get_text_detection_result["NextToken"]
            )
            self.extract_visual_texts(get_text_detection_result)
    
    def fetch_transcription(self):
        video_transcription_file: dict = self.s3.get_object(Bucket=self.bucket_name, Key=self.video_transcript_s3_path)
        self.transcript = json.loads(video_transcription_file['Body'].read().decode('utf-8'))

    def download_video(self) -> str:
        filename: str = os.path.basename(self.video_s3_path)
        self.s3.download_file(self.bucket_name, self.video_s3_path, filename)
        return filename

    def load_video(self, video_filename: str) -> cv2.VideoCapture:
        return cv2.VideoCapture(video_filename)

    def detect_faces_and_celebrities(self):
        if len(self.visual_objects) == 0: raise AssertionError("Face and celebrity detection is called before label detection is parsed")
        if len(self.person_timestamps_millis) == 0: return # No face object detected in the video, returning

        video: cv2.VideoCapture = self.load_video( self.download_video() )
        
        timestamp_millis: int
        for timestamp_millis in self.person_timestamps_millis:
            timestamp_second = int(timestamp_millis/1000)
            video.set(cv2.CAP_PROP_POS_MSEC, int(timestamp_millis))
            success, frame = video.read()
            
            if success:
                image = Image.fromarray(frame)
                io_stream = io.BytesIO()
                image.save(io_stream, format='JPEG')
                image = io_stream.getvalue()

                # Call Rekognition to detect celebrity
                recognize_celebrity_response: dict = self.rekognition.recognize_celebrities(Image={'Bytes': image})
                celebrity_findings: list[dict] = recognize_celebrity_response["CelebrityFaces"]
                unrecognized_faces: list[dict] = recognize_celebrity_response["UnrecognizedFaces"]
                
                # Parse Rekognition celebrity detection data and add to dictionary as appropriate
                if len(celebrity_findings) > 0:
                    if timestamp_second not in self.celebrities: self.celebrities[timestamp_second] = []
                    celebrity_finding_dict: dict
                    for celebrity_finding_dict in celebrity_findings:
                        if int(celebrity_finding_dict["MatchConfidence"]) < CelebrityFinding.celebrity_match_confidence_threshold: continue
                        celebrity_finding = CelebrityFinding(celebrity_finding_dict)
                        self.celebrities[timestamp_second].append(celebrity_finding)

                # Only call the detect face APU if there are other faces beside the recognized celebrity in this frame
                # This also applies when there is 0 celebrity detected, but there are more faces in the frame.
                if len(unrecognized_faces) > 0:
                    # Call Rekognition to detect faces
                    face_findings: dict = self.rekognition.detect_faces(Image={'Bytes': image},Attributes=['ALL'])['FaceDetails']

                    if len(face_findings) > 0:
                        if timestamp_second not in self.faces: self.faces[timestamp_second] = []
                        face_finding_dict: dict
                        for face_finding_dict in face_findings:
                            if int(face_finding_dict["Confidence"]) < FaceFinding.face_detection_confidence_threshold : continue
                            face_finding = FaceFinding(face_finding_dict)

                            # The below code checks if this face is already captured as celebrity by checking the bounding box for the detected celebrities at this frame
                            face_found_in_celebrities_list = False
                            for celebrity_finding in self.celebrities[timestamp_second]:
                                if celebrity_finding.is_matching_face(face_finding.top, face_finding.left, face_finding.height, face_finding.width): 
                                    face_found_in_celebrities_list = True

                            # Only add if the face is not found in the celebrity list
                            if not face_found_in_celebrities_list:
                                # Only add if there is no other face with similar age range at the same second.
                                if not self.is_duplicate(self.faces[timestamp_second]):
                                    self.faces[timestamp_second].append(face_finding)

    def wait_for_dependencies(self):
        self.wait_for_rekognition_label_detection(sort_by='TIMESTAMP')
        self.wait_for_rekognition_text_detection()
        self.wait_for_transcription_job()

    def run(self):
        self.iterate_object_detection_result()
        self.iterate_text_detection_result()
        self.fetch_transcription()
        self.detect_faces_and_celebrities()
        return self.visual_objects, self.visual_texts, self.transcript, self.celebrities, self.faces

    
class VideoAnalyzer(ABC):
    def __init__(self, 
        bucket_name: str,
        video_name: str, 
        video_path: str, 
        visual_objects: dict[int, list[str]], 
        visual_texts: dict[int, list[str]], 
        transcript: dict,
        celebrities: dict[int, list[str]],
        faces: dict[int, list[str]],
        summary_folder: str,
        entity_sentiment_folder: str,
        video_script_folder: str
        ):

        self.s3_client = boto3.client("s3")
        self.bucket_name: str = bucket_name
        self.summary_folder: str = summary_folder
        self.entity_sentiment_folder: str = entity_sentiment_folder
        self.video_name: str = video_name
        self.video_path: str = video_path
        self.original_visual_objects: dict[int, list[str]] = visual_objects
        self.original_visual_texts: dict[int, list[str]] = visual_texts
        self.original_transcript: dict = transcript
        self.original_celebrities: dict[int, list[CelebrityFinding]] = celebrities
        self.original_faces: dict[int, list[FaceFinding]] = faces
        self.visual_objects: list[list[Union[int, list[str]]]] = []
        self.visual_texts: list[list[Union[int, list[str]]]] = []
        self.transcript: list[list[Union[int, str]]] = []
        self.celebrities:list[list[Union[int, list[str]]]]  = []
        self.faces: list[list[Union[int, list[str]]]]  = []
        self.video_script_chunk_size_for_summary_generation: int = 100000 # characters
        self.video_script_chunk_overlap_for_summary_generation: int = 500 # characters
        self.video_script_chunk_size_for_entities_extraction: int = 50000 #10000 # characters
        self.video_script_chunk_overlap_for_entities_extraction: int = 500 #200 # characters
        self.video_script_storage_chunk_size: int = 768 #10000 # Number of characters, which depends on the embedding model
        self.scene_similarity_score: float = 0.9
        self.text_similarity_score: float = 0.9
        self.video_rolling_summary: str = ""
        self.video_rolling_sentiment: str = ""
        self.combined_video_script: str = ""
        self.all_combined_video_script: str = ""
        self.llm_parameters: dict = {}
        self.summary: str = ""
        self.entities: str = ""
        self.video_script: str = ""
    
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
  
    @abstractmethod
    def call_llm(self, prompt):
        pass
    
    @abstractmethod
    def call_embedding_llm(self, document):
        pass
  
    def preprocess_visual_objects(self):
        visual_objects_across_timestamps = dict(sorted(copy.deepcopy(self.original_visual_objects).items()))
        prev_objects: list = []

        timestamp_second: int
        visual_objects_at_particular_timestamp: list
        for timestamp_second, visual_objects_at_particular_timestamp in list(visual_objects_across_timestamps.items()):
            # The loop below is to check how much of this scene resembling previous scene
            num_of_matches_with_prev_objects: int = 0
            visual_object: str
            for visual_object in visual_objects_at_particular_timestamp:
                if visual_object in prev_objects: num_of_matches_with_prev_objects += 1
            # Delete scene entry if there is no detected object
            if len(visual_objects_at_particular_timestamp) == 0: 
                del visual_objects_across_timestamps[timestamp_second]
            # Delete scene entry if the detected objects are too similar with the previous scene
            elif float(num_of_matches_with_prev_objects) > len(prev_objects)*self.scene_similarity_score:
                del visual_objects_across_timestamps[timestamp_second]
            else:
                prev_objects = visual_objects_at_particular_timestamp
      
        self.visual_objects = sorted(visual_objects_across_timestamps.items())
      
    def preprocess_visual_texts(self):
        visual_texts_across_timestamps = dict(sorted(self.original_visual_texts.items()))
        prev_texts: list = []

        timestamp_second: int
        visual_texts_at_particular_timestamp: list
        for timestamp_second, visual_texts_at_particular_timestamp in list(visual_texts_across_timestamps.items()):
            # The loop below is to check how much of this text resembling previous text
            num_of_matches_with_prev_texts: int = 0
            text: str
            for text in visual_texts_at_particular_timestamp:
                if text in prev_texts: num_of_matches_with_prev_texts += 1
            #  Delete text entry if there is not detected text
            if len(visual_texts_at_particular_timestamp) == 0: 
                del visual_texts_across_timestamps[timestamp_second]
            # Delete text entry if the detected texts are too similar with the previous scene
            elif float(num_of_matches_with_prev_texts) > len(prev_texts)*self.text_similarity_score:
                del visual_texts_across_timestamps[timestamp_second]
            else:
                prev_texts = visual_texts_at_particular_timestamp
      
        self.visual_texts = sorted(visual_texts_across_timestamps.items())

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
    
    def preprocess_celebrities(self):
        self.celebrities = sorted(dict([(timestamp_second, celebrity_finding.display()) for timestamp_second, celebrity_finding in self.original_celebrities.items()]))

    def preprocess_faces(self):
        self.faces = sorted(dict([(timestamp_second, face_finding.display()) for timestamp_second,face_finding in self.original_faces.items()]))
      
    def generate_combined_video_script(self):
        def transform_scenes(x):
            timestamp = x[0]
            objects = ",".join(x[1])
            return (timestamp, f"Scene:{objects}")
        scenes  = list(map(transform_scenes, self.visual_objects))

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

        def transform_celebrities(x):
            timestamp = x[0]
            celebrities = ",".join(x[1])
            return (timestamp, f"Celebrity:{celebrities}")
        visual_celebrities = list(map(transform_celebrities, self.celebrities))

        def transform_faces(x):
            timestamp = x[0]
            faces = ",".join(x[1])
            return (timestamp, f"Face:{faces}")
        visual_faces = list(map(transform_faces, self.faces))
 
        # Combine all inputs
        combined_video_script = sorted( scenes + visual_texts + transcript + visual_celebrities + visual_faces )
        
        combined_video_script = "\n".join(list(map(lambda x: f"{x[0]}:{x[1]}", combined_video_script)))
        
        self.combined_video_script = combined_video_script
        self.all_combined_video_script += combined_video_script
  
    def generate_summary(self):
        prompt_prefix = "You are an expert video analyst who reads a Video Timeline and creates summary of the video.\n" \
                        "The Video Timeline is a text representation of a video.\n" \
                        "The Video Timeline contains the visual scenes, the visual texts, and human voice in the video.\n" \
                        "Visual scenes (scene) represents what objects are visible in the video at that second. This can be the objects seen in camera, or objects from a screen sharing, or any other visual scenarios. You can infer how the visualization look like, so long as you are confident.\n" \
                        "Visual texts (text) are the text visible in the video. It can be texts in real world objects as recorded in video camera, or those from screen sharing, or those from presentation recording, or those from news, movies, or others. \n" \
                        "Human voice (voice) is the transcription of the video.\n" \
                        "Celebrities (celebrity) provides information about the celebrity detected in the video at that second and (if any) their visible detected emotions. For example: Jeff Bezos|happy-surprised,Andy Jassy|calm,Jeff Barr \n" \
                        "Human faces (face) lists the face seen in the video at that second, the gender (if detected), and whether the person smiles or not (if detected). For example: 23-29 years old|male|smiling,31-33 years old|not smiling,44-49 years old. \n"

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
                        "Human voice (voice) is the transcription of the video.\n" \
                        "Celebrities (celebrity) provides information about the celebrity detected in the video at that second and (if any) their visible detected emotions. For example: Jeff Bezos|happy-surprised,Andy Jassy|calm,Jeff Barr \n" \
                        "Human faces (face) lists the face seen in the video at that second, the gender (if detected), and whether the person smiles or not (if detected). For example: 23-29 years old|male|smiling,31-33 years old|not smiling,44-49 years old. \n"
                        
        
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
    
    def store_summary_result(self):
        # Store summary in S3
        self.s3_client.put_object(
            Body=self.summary, 
            Bucket=self.bucket_name, 
            Key=f"{self.summary_folder}/{self.video_path}.txt"
        )

        summary_embedding = self.call_embedding_llm(self.summary)

        # Store summary in database
        update_stmt = (
            update(self.Videos).
            where(self.Videos.name == self.video_path).
            values(summary = self.summary, summary_embedding = summary_embedding)
        )
        
        session.execute(update_stmt)
        session.commit()

    def store_sentiment_result():
        # Extract entities and sentiment from the string
        entities_dict: dict[str, dict[str, str]] = {}
        entity_regex = r"\n*([^|\n]+?)\|\s*(positive|negative|neutral|mixed|N\/A)\s*\|([^|\n]+?)\n"
        for match in re.finditer(entity_regex, self.entities):
            entity = match.group(1).strip()
            sentiment = match.group(2)
            reason = match.group(3)
            entities_dict[entity] = {
                "sentiment": sentiment,
                "reason": reason
            }

        self.s3_client.put_object(
            Body="\n".join(f"{e}|{s['sentiment']}|{s['reason']}" for e, s in entities_dict.items()), 
            Bucket=self.bucket_name, 
            Key=f"{self.entity_sentiment_folder}/{self.video_path}.txt"
        )

        entities: list[self.Entities] = []
        for entity, value in entities_dict.items():
            # Store entity in database
            entities.append(self.Entities(
                name=entity,
                sentiment=value['sentiment'],
                reason=value['reason'],
                video_name=self.video_path
            ))

        # Store into database
        session.add_all(entities)
        session.commit()
        
    def store_video_script_result(self):
        self.s3_client.put_object(
            Body=self.video_script, 
            Bucket=self.bucket_name, 
            Key=f"{self.video_script_folder}/{self.video_path}.txt"
        )

        # Chunking the video script for storage in DB while converting them to embedding
        video_script_length = len(self.video_script)
        number_of_chunks = math.ceil( (video_script_length + 1) / self.video_script_storage_chunk_size )

        chunks: list[self.Contents] = []
        for chunk_number in range(0, number_of_chunks):
            is_last_chunk = (chunk_number == (number_of_chunks - 1))
            is_first_chunk = (chunk_number == 0)

            start = 0 if is_first_chunk else int(chunk_number*self.video_script_storage_chunk_size)
            stop = video_script_length if is_last_chunk else (chunk_number+1)*self.video_script_storage_chunk_size
            chunk_string = video_script[start:stop]
        
            # So long as this is not the first chunk, remove whatever before first \n since likely the chunk cutting is not done exactly at the \n
            if not is_first_chunk:
                try:
                    chunk_string = chunk_string[chunk_string.index("\n"):] 
                except:
                    pass
            
            # Get the embedding for the chunk
            chunk_embedding = call_embedding_llm(chunk_string)
            
            # Create database object
            chunks.append(self.Contents(
                chunk=chunk_string,
                chunk_embedding=chunk_embedding,
                video_name=self.video_path
            ))

        # Store in database
        session.add_all(chunks)
        session.commit()

      
    def run(self):
        self.preprocess_visual_objects()
        self.preprocess_visual_texts()
        self.preprocess_transcript()
        self.preprocess_celebrities()
        self.preprocess_faces()

        self.generate_combined_video_script()

        self.summary = self.generate_summary()
        self.entities = self.extract_sentiment()
        self.video_script = self.all_combined_video_script
    
    def store(self):
        self.store_summary_result()
        self.store_sentiment_result()
        self.store_video_script_result()

class VideoAnalyzerBedrock(VideoAnalyzer):    
    def __init__(self, 
        model_name: str,
        embedding_model_name: str,
        bucket_name: str,
        video_name: str, 
        video_path: str, 
        visual_objects: dict[int, list[str]], 
        visual_texts: dict[int, list[str]], 
        transcript: dict,
        celebrities: dict[int, list[str]],
        faces: dict[int, list[str]],
        summary_folder: str,
        entity_sentiment_folder: str,
        video_script_folder: str
        ):
        super().__init__(bucket_name, video_name, video_path, visual_objects, visual_texts, transcript, celebrities, faces,summary_folder, entity_sentiment_folder, video_script_folder)
        self.bedrock_client = boto3.client("bedrock-runtime")
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
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
                bedrock_response = self.bedrock_client.invoke_model(body=encoded_input, modelId=self.model_name)
                call_done = True
            except bedrock.exceptions.ThrottlingException as e:
                print("Amazon Bedrock throttling exception")
                time.sleep(60)
            except Exception as e:
                raise e

        response = json.loads(bedrock_response.get("body").read())["completion"]
        return response
    
    def call_embedding_llm(self, document):
        # Get summary embedding
        body = json.dumps({
            "texts":[document],
            "input_type": "search_document",
        })
        call_done = False
        while(not call_done):
            try:
                response = self.bedrock_client.invoke_model(body=body, modelId=self.embedding_model_name)
                call_done = True
            except ThrottlingException:
                print("Amazon Bedrock throttling exception")
                time.sleep(60)
            except Exception as e:
                raise e

        # Disabling semgrep rule for checking data size to be loaded to JSON as the source is from Amazon Bedrock
        # nosemgrep: python.aws-lambda.deserialization.tainted-json-aws-lambda.tainted-json-aws-lambda
        embedding = json.loads(response.get("body").read().decode())["embeddings"][0] #["embedding"]
        return embedding


if __name__ == "__main__":
    handler()
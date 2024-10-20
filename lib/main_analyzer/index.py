import os, time, json, copy, math, re, io
from abc import ABC, abstractmethod
import boto3, botocore
from botocore.config import Config
from sqlalchemy import create_engine, Column, DateTime, String, Text, Integer, func, ForeignKey, update
from sqlalchemy.orm import mapped_column, sessionmaker,  declarative_base
from pgvector.sqlalchemy import Vector
from typing import Union, Self
import cv2
import base64
from PIL import Image
import concurrent.futures
from multiprocessing import Pool
import itertools
import logging

CONFIG_LABEL_DETECTION_ENABLED = "label_detection_enabled"
CONFIG_TRANSCRIPTION_ENABLED = "transcription_enabled"
CONFIG_NUMBER_OF_FRAMES_TO_LLM = "number_of_frames_to_llm"
CONFIG_VIDEO_SAMPLING_INTERVAL_MS = "video_sampling_interval_ms"
LLM_MODEL = "llm_model"

fast_model_id = "anthropic.claude-3-haiku-20240307-v1:0"
balanced_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"


secrets_manager = boto3.client('secretsmanager')

model_id = fast_model_id if os.environ[LLM_MODEL] == "fast" else balanced_model_id
vqa_model_id = model_id
frame_interval = os.environ[CONFIG_VIDEO_SAMPLING_INTERVAL_MS]
number_of_frames_to_llm = os.environ[CONFIG_NUMBER_OF_FRAMES_TO_LLM]
embedding_model_id = os.environ["EMBEDDING_MODEL_ID"]
embedding_dimension = os.environ['EMBEDDING_DIMENSION']
bucket_name = os.environ["BUCKET_NAME"]
raw_folder = os.environ["RAW_FOLDER"]
video_script_folder = os.environ["VIDEO_SCRIPT_FOLDER"]
transcription_folder = os.environ["TRANSCRIPTION_FOLDER"]
entity_sentiment_folder = os.environ["ENTITY_SENTIMENT_FOLDER"]
summary_folder = os.environ["SUMMARY_FOLDER"]
video_caption_folder = os.environ["VIDEO_CAPTION_FOLDER"]

database_name = os.environ['DATABASE_NAME']
video_table_name = os.environ['VIDEO_TABLE_NAME']
entities_table_name = os.environ['ENTITIES_TABLE_NAME']
content_table_name = os.environ['CONTENT_TABLE_NAME']
secret_name = os.environ['SECRET_NAME']
writer_endpoint = os.environ['DB_WRITER_ENDPOINT']

video_s3_path = os.environ['VIDEO_S3_PATH'] 
transcription_job_name = os.environ['TRANSCRIPTION_JOB_NAME']
label_detection_job_id = os.environ['LABEL_DETECTION_JOB_ID']
label_detection_enabled = True if os.environ[CONFIG_LABEL_DETECTION_ENABLED] == "1" else False
transcription_enabled = True if os.environ[CONFIG_TRANSCRIPTION_ENABLED] == "1" else False


credentials = json.loads(secrets_manager.get_secret_value(SecretId=secret_name)["SecretString"])
username = credentials["username"]
password = credentials["password"]

engine = create_engine(f'postgresql://{username}:{password}@{writer_endpoint}:5432/{database_name}')
Base = declarative_base()

Session = sessionmaker(bind=engine)  
session = Session()   

class CelebrityFinding():
    celebrity_match_confidence_threshold: int = 97
    face_bounding_box_overlap_threshold: float = 0.1
    celebrity_emotion_confidence_threshold: int = 85
    celebrity_feature_confidence_threshold: int = 90

    def __init__(self, celebrity_dict: dict):
        self.name: str = celebrity_dict['Name']
        self.match_confidence: int = int(celebrity_dict['MatchConfidence'])
        # These emotion related attributes were disabled on July 29, 2024 to respect this AUP https://www.anthropic.com/legal/aup
        #self.smile: bool = bool(celebrity_dict['Face']['Smile']['Value'])
        #self.smile_confidence: int = int(celebrity_dict['Face']['Smile']['Confidence'])
        #self.emotions: list[str] = list(filter(lambda em: (len(em) > 0),[emo['Type'].lower() if int(emo['Confidence']) >= self.celebrity_emotion_confidence_threshold else '' for emo in celebrity_dict['Face']['Emotions']]))
        self.top: float = float(celebrity_dict['Face']['BoundingBox']['Top'])
        self.top_display: str = str(round(self.top, 2))
        self.left: float = float(celebrity_dict['Face']['BoundingBox']['Left'])
        self.left_display: str = str(round(self.left, 2))
        self.height: float = float(celebrity_dict['Face']['BoundingBox']['Height'])
        self.height_display: str = str(round(self.height, 2))
        self.width: float = float(celebrity_dict['Face']['BoundingBox']['Width'])
        self.width_display: str = str(round(self.width, 2))
        
        logging.basicConfig(level=logging.DEBUG)

    def is_matching_face(self, bb_top:float , bb_left: float, bb_height: float, bb_width: float) -> bool:
        return (
            (abs(self.top - bb_top) <= self.face_bounding_box_overlap_threshold) and
            (abs(self.left - bb_left) <= self.face_bounding_box_overlap_threshold) and
            (abs(self.height - bb_height) <= self.face_bounding_box_overlap_threshold) and
            (abs(self.width - bb_width) <= self.face_bounding_box_overlap_threshold)
        )

    def display(self) -> str:
        display_string = f"Name is {self.name}. "
        # These emotion related attributes were disabled on July 29, 2024 to respect this AUP https://www.anthropic.com/legal/aup
        #if len(self.emotions) > 0:
        #    display_string += "Emotions appear to be " + ", ".join(self.emotions) + ". "
        #if self.smile_confidence >= self.celebrity_feature_confidence_threshold: 
        #    display_string += "The celebrity is smiling. " if self.smile else "The celebrity is not smiling at this moment. "
        display_string += f"The celeb's face is located at {self.left_display} of frame's width from left and {self.top_display} of frame's height from top, with face height of {self.height_display} of frame's height and face width of {self.width_display} of the frame's width. "

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
        self.top: float = float(face_dict['BoundingBox']['Top'])
        self.top_display: str = str(round(self.top, 2))
        self.left: float = float(face_dict['BoundingBox']['Left'])
        self.left_display: str = str(round(self.left, 2))
        self.height: float = float(face_dict['BoundingBox']['Height'])
        self.height_display: str = str(round(self.height, 2))
        self.width: float = float(face_dict['BoundingBox']['Width'])
        self.width_display: str = str(round(self.width, 2))
        self.beard: bool = bool(face_dict['Beard']['Value'])
        self.beard_confidence: int = int(face_dict['Beard']['Confidence'])
        self.eyeglasses: bool = bool(face_dict['Eyeglasses']['Value'])
        self.eyeglasses_confidence: int = int(face_dict['Eyeglasses']['Confidence'])
        self.eyesopen: bool = bool(face_dict['EyesOpen']['Value'])
        self.eyesopen_confidence: int = int(face_dict['EyesOpen']['Confidence'])
        self.sunglasses: bool = bool(face_dict['Sunglasses']['Value'])
        self.sunglasses_confidence: int = int(face_dict['Sunglasses']['Confidence'])
        # These emotion related attributes were disabled on July 29, 2024 to respect this AUP https://www.anthropic.com/legal/aup
        #self.smile: bool = bool(face_dict['Smile']['Value'])
        #self.smile_confidence: int = int(face_dict['Smile']['Confidence'])
        self.mouthopen: bool = bool(face_dict['MouthOpen']['Value'])
        self.mouthopen_confidence: int = int(face_dict['MouthOpen']['Confidence'])
        self.mustache: bool = bool(face_dict['Mustache']['Value'])
        self.mustache_confidence: int = int(face_dict['Mustache']['Confidence'])
        self.gender: str = str(face_dict['Gender']['Value']).lower()
        self.gender_confidence: int = int(face_dict['Gender']['Confidence'])
        # This emotion related attributes were disabled on July 29, 2024 to respect this AUP https://www.anthropic.com/legal/aup
        #self.emotions: list[str] = list(filter(lambda em: (len(em) > 0),[emo['Type'].lower() if int(emo['Confidence']) >= self.face_emotion_confidence_threshold else '' for emo in face_dict['Emotions']]))

    def is_duplicate(self, face_list: list[Self]) -> bool:
        found = False
        face_finding: Self
        for face_finding in face_list:
            if (abs(self.age_low - face_finding.age_low) <= self.face_age_range_match_threshold) and (abs(self.age_high - face_finding.age_high) <= self.face_age_range_match_threshold): found = True
        return found

    def display(self) -> str:
        display_string = f"The person is about {self.age_low} to {self.age_high} years old. "
        if self.gender_confidence >= self.face_feature_confidence_threshold:
            display_string += f"Identifed gender is {self.gender}. "
        # These emotion related attributes were disabled on July 29, 2024 to respect this AUP https://www.anthropic.com/legal/aup
        #if len(self.emotions) > 0:
        #    display_string += "Emotion appears to be " + ", ".join(self.emotions) + ". "
        #if self.smile_confidence >= self.face_feature_confidence_threshold:
        #    display_string += "Seems to be smiling. " if self.smile else "Seems to be not smiling. "
        if self.beard_confidence >= self.face_feature_confidence_threshold:
            display_string += "Person has beard. " if self.beard else "Person has no beard. "
        if self.mustache_confidence >= self.face_feature_confidence_threshold:
            display_string += "Person has mustache. " if self.mustache else "Person has no mustache. "
        if self.sunglasses_confidence >= self.face_feature_confidence_threshold:
            display_string += "Person wears sunglasses. " if self.sunglasses else "No sunglasses is identified. "
        if self.eyeglasses_confidence >= self.face_feature_confidence_threshold:
            display_string += "Person wears eyeglasses. " if self.eyeglasses else "Person does not wear eyeglasses. "
        if self.mouthopen_confidence >= self.face_feature_confidence_threshold:
            display_string += "At this moment, person's mouth is open, might be speaking. " if self.mouthopen else "At this moment, person's mouth is not opened. "
        if self.eyesopen_confidence >= self.face_feature_confidence_threshold:
            display_string += "Person's eyes are open. " if self.eyesopen else "At this moment, person's eyes are closed. "
        display_string += f"Person's face is located {self.left_display} of frame's width from left and {self.top_display} of frame's height from top, with face height of {self.height_display} of frame's height and face width {self.width_display} of frame's width. "
        
        return display_string

class ObjectFinding():
    confidence_score_threshold: float = 80.0
    top_n_threshold: int = 10
    def __init__(self, label: str, confidence_score: float):
        self.label: str = label
        self.confidence_score: float = confidence_score
    
    def display(self) -> str:
        return self.label

class VideoPreprocessor(ABC):
    s3_client = boto3.client("s3")
    transcribe_client = boto3.client("transcribe")
    rekognition_client = boto3.client("rekognition")
    
    def __init__(self, 
        label_detection_job_id: str,
        transcription_job_name: str,
        bucket_name: str,
        video_s3_path: str,
        video_transcript_s3_path: str,
        frame_interval: str):

        self.label_detection_job_id: str = label_detection_job_id
        self.transcription_job_name: str = transcription_job_name
        self.bucket_name: str = bucket_name
        self.video_s3_path: str = video_s3_path
        self.video_transcript_s3_path: str = video_transcript_s3_path
        self.video_duration_seconds: float  = 0.0
        self.video_duration_millis: int = 0
        self.visual_objects: dict[int, list[ObjectFinding]] = {}
        self.visual_scenes: dict[int, str] = {}
        self.visual_captions: dict[int, str] = {}
        self.visual_texts: dict[int, list[str]] = {}
        self.transcript: dict = {}
        self.celebrities: dict[int, list[CelebrityFinding]] = {}
        self.faces: dict[int, list[FaceFinding]] = {}
        self.person_timestamps_millis: list[int] = []
        self.text_timestamps_millis: list[int] = []
        self.frame_interval: int = int(frame_interval) # Millisecond
        self.frame_interval_tolerance: int = int(0.25*self.frame_interval) # In millisecond. This means, any frame located within this tolerance in the timeline will be considered the same as the main frame being taken at regular interval
        self.frame_dim_for_vqa: tuple(int) = (512, 512)
        self.video_filename = ""
        self.frame_bytes: list[list[Union[int, bytes]]] = []
        self.person_frame_bytes: list[list[Union[int, bytes]]] = []
        self.parallel_degree = os.cpu_count()
    
    @abstractmethod
    def call_vqa(self, image_data: str) ->str:
        pass

    def wait_for_rekognition_label_detection(self, sort_by):
        get_object_detection = self.rekognition_client.get_label_detection(JobId=self.label_detection_job_id, SortBy=sort_by)
        while(get_object_detection['JobStatus'] == 'IN_PROGRESS'):
            time.sleep(5)
            get_object_detection = self.rekognition_client.get_label_detection(JobId=self.label_detection_job_id, SortBy=sort_by)

    def wait_for_transcription_job(self):
        get_transcription = self.transcribe_client.get_transcription_job(TranscriptionJobName=self.transcription_job_name)
        job_status = get_transcription["TranscriptionJob"]["TranscriptionJobStatus"]
        while(job_status == 'IN_PROGRESS' or job_status == "QUEUED"):
            time.sleep(5)
            get_transcription = self.transcribe_client.get_transcription_job(TranscriptionJobName=self.transcription_job_name)
    
    def extract_visual_objects(self, get_object_detection_result: dict):
        person_timestamps_seconds: list(int) = []
        label: dict
        for label in get_object_detection_result['Labels']:
            objects_at_this_timestamp: list = []
            timestamp_millis: int = int(label['Timestamp'])
            timestamp_second: int = round(timestamp_millis/1000, 1)

            # If this timestamp millis is already in self.visual_objects dictionary, then use it and append. Otherwise, add a new key to the dict.
            if timestamp_millis in self.visual_objects:
                objects_at_this_timestamp = self.visual_objects[timestamp_millis]
            else:
                self.visual_objects[timestamp_millis] = objects_at_this_timestamp

            object_name: str = label['Label']['Name']
            confidence: float = label['Label']['Confidence']

            # If this is a Person object, then register this into the timestamp list for face detection.
            if object_name == "Person" and confidence >= FaceFinding.face_detection_confidence_threshold:
                self.person_timestamps_millis.append(timestamp_millis)
                person_timestamps_seconds.append(timestamp_second)
            
            if object_name == "Text":
                self.text_timestamps_millis.append(timestamp_millis)

            if (object_name not in objects_at_this_timestamp):
                # Append the object name to the list of objects for timestamp second
                object_finding = ObjectFinding(label=object_name, confidence_score=confidence)
                objects_at_this_timestamp.append(object_finding)
  
    def iterate_object_detection_result(self):
        get_object_detection_result: dict = self.rekognition_client.get_label_detection(
            JobId=self.label_detection_job_id,
            MaxResults=1000,
            SortBy='TIMESTAMP'
        )

        if get_object_detection_result["JobStatus"] == "FAILED": return # In case the job failed, just skip this channel.

        self.video_duration_millis = int(get_object_detection_result["VideoMetadata"]["DurationMillis"])
        self.video_duration_seconds = self.video_duration_millis/1000

        # Extract visual scenes and populate self.visual_objects
        self.extract_visual_objects(get_object_detection_result)

        # In case results is large, iterate the next pages until no more page left.
        while("NextToken" in get_object_detection_result):
            get_object_detection_result: dict = self.rekognition_client.get_label_detection(
                JobId=self.label_detection_job_id,
                MaxResults=1000,
                NextToken=get_object_detection_result["NextToken"]
            )
            self.extract_visual_objects(get_object_detection_result)

    def fetch_transcription(self) -> dict:
        get_transcription = self.transcribe_client.get_transcription_job(TranscriptionJobName=self.transcription_job_name)
        if get_transcription["TranscriptionJob"]["TranscriptionJobStatus"] == "FAILED": return # In case the job failed, just skip this channel.

        video_transcription_file: dict = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.video_transcript_s3_path)
        self.transcript = json.loads(video_transcription_file['Body'].read().decode('utf-8'))

    def download_video_and_load_metadata(self):
        filename: str = os.path.basename(self.video_s3_path)
        self.s3_client.download_file(self.bucket_name, self.video_s3_path, filename)
        self.video_filename = filename
        video: cv2.VideoCapture = cv2.VideoCapture(self.video_filename)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)
        self.video_duration_seconds = float(frame_count/fps)
        self.video_duration_millis = int(self.video_duration_seconds*1000)

    def load_video(self, filename) -> cv2.VideoCapture:
        video: cv2.VideoCapture = cv2.VideoCapture(filename)
        return video
    
    def _detect_faces_and_celebrities_at_timestamp(self, timestamp_data):
        timestamp_millis: int = timestamp_data[0]
        image: bytes = timestamp_data[1]

        # Call Rekognition to detect celebrity
        recognize_celebrity_response: dict = self.rekognition_client.recognize_celebrities(Image={'Bytes': image})
        celebrity_findings: list[dict] = recognize_celebrity_response["CelebrityFaces"]
        unrecognized_faces: list[dict] = recognize_celebrity_response["UnrecognizedFaces"]
        
        # Parse Rekognition celebrity detection data and add to dictionary as appropriate
        if len(celebrity_findings) > 0:
            if timestamp_millis not in self.celebrities: self.celebrities[timestamp_millis] = []
            celebrity_finding_dict: dict
            for celebrity_finding_dict in celebrity_findings:
                if int(celebrity_finding_dict["MatchConfidence"]) < CelebrityFinding.celebrity_match_confidence_threshold: continue
                celebrity_finding = CelebrityFinding(celebrity_finding_dict)
                self.celebrities[timestamp_millis].append(celebrity_finding)

        # Only call the detect face APU if there are other faces beside the recognized celebrity in this frame
        # This also applies when there is 0 celebrity detected, but there are more faces in the frame.
        if len(unrecognized_faces) == 0: return None

        # Call Rekognition to detect faces
        face_findings: dict = self.rekognition_client.detect_faces(Image={'Bytes': image}, Attributes=['ALL'])['FaceDetails']

        if len(face_findings) == 0: return None

        if timestamp_millis not in self.faces: self.faces[timestamp_millis] = []
        face_finding_dict: dict
        for face_finding_dict in face_findings:
            if int(face_finding_dict["Confidence"]) < FaceFinding.face_detection_confidence_threshold : continue
            face_finding = FaceFinding(face_finding_dict)

            # The below code checks if this face is already captured as celebrity by checking the bounding box for the detected celebrities at this frame
            face_found_in_celebrities_list = False
            if timestamp_millis in self.celebrities:
                for celebrity_finding in self.celebrities[timestamp_millis]:
                    if celebrity_finding.is_matching_face(face_finding.top, face_finding.left, face_finding.height, face_finding.width): 
                        face_found_in_celebrities_list = True

            # Only add if the face is not found in the celebrity list
            if not face_found_in_celebrities_list:
                # Only add if there is no other face with similar age range at the same millisecond.
                if not face_finding.is_duplicate(self.faces[timestamp_millis]):
                    self.faces[timestamp_millis].append(face_finding)

    def detect_faces_and_celebrities(self):
        if len(self.person_timestamps_millis) == 0:
            print("Warning: detect_faces_and_celebrities may be called before objects detection, which caused 0 'person' result")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_degree*15) as executor:
            executor.map(self._detect_faces_and_celebrities_at_timestamp, self.person_frame_bytes)


    def _extract_frame(self, timestamp_millis):
        video = self.load_video(self.video_filename)
        video.set(cv2.CAP_PROP_POS_MSEC, int(timestamp_millis))
        success, frame = video.read()
        if success:
            # Resize frame to 512 x 512 px
            dim = self.frame_dim_for_vqa
            # This may fail if the frame is empty. Just skip the frame if so.
            try:
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            except:
                return None
        
            image_pil = Image.fromarray(frame)
            io_stream = io.BytesIO()
            image_pil.save(io_stream, format='JPEG')
            image = io_stream.getvalue()
            return [timestamp_millis, image]
        del video

    def extract_frames(self):
        # Create a list containing milliseconds where frame should be extracted from the video, according to the interval.
        # This may look like [0, 1000, 2000, 3000]
        regular_timestamps_millis = list(range(0, self.video_duration_millis, self.frame_interval))

        # Remove duplicate text_timestamp_millis timestamps (too close to each other) as compared to regular_timestamp_millis
        # For example, if Amazon Rekognition detects text at millisecond 2103, and there is already regular frame interval to be extracted at 2000 with tolerance of 250 millisecond, then this 2103 timestamp will be ignored assuming the text will be captured at 2000.
        text_timestamps_millis = []
        for t in self.text_timestamps_millis:
            include = True
            for r in regular_timestamps_millis:
                if abs(t-r) < self.frame_interval_tolerance:
                    include = False
                    break
            if include:
                text_timestamps_millis.append(t)

        text_timestamps_millis =  list(filter(lambda t: t is not None, text_timestamps_millis))

        regular_and_text_timestamps_millis = regular_timestamps_millis + text_timestamps_millis

        # Remove duplicate person_timestamps_millis timestamps (too close to each other) as compared to regular_and_text_timestamp_millis
        # For example, if Amazon Rekognition detects a person at millisecond 2789, and there is already regular frame interval to be extracted at 3000 with tolerance of 250 millisecond, then this 2789 timestamp will be ignored assuming the person will be captured at 3000.
        person_timestamps_millis = []
        person_timestamp_millis_joined_with_regular = []
        for t in self.person_timestamps_millis:
            include = True
            for r in regular_and_text_timestamps_millis:
                if abs(t-r) < self.frame_interval_tolerance:
                    include = False
                    person_timestamp_millis_joined_with_regular.append(r)
                    break
            if include:
                person_timestamps_millis.append(t)

        person_timestamps_millis =  list(filter(lambda t: t is not None, person_timestamps_millis))

        with Pool(self.parallel_degree) as p:
            self.frame_bytes = list(p.map(self._extract_frame, regular_and_text_timestamps_millis))
        self.frame_bytes = list(filter(lambda f: f is not None, self.frame_bytes))

        with Pool(self.parallel_degree) as p:
            self.person_frame_bytes = list(p.map(self._extract_frame, person_timestamps_millis))
        self.person_frame_bytes = list(filter(lambda f: f is not None, self.person_frame_bytes))
        
        self.person_frame_bytes = self.person_frame_bytes + list(filter(lambda f: f[0] in person_timestamp_millis_joined_with_regular, self.frame_bytes))
 
    def _extract_scene_from_vqa(self, frame_info: list[list[Union[int, bytes]]]):
        timestamp_millis = frame_info[0][0]
        # Extract all image data into image_list
        image_list = [frame[1] for frame in frame_info]
        
        # Read the system prompt from S3
        system_prompt = ""
        
        try:
            response = s3_client.get_object(Bucket=self.bucket_name, Key='source/system_prompt.txt')
            system_prompt = response['Body'].read().decode('utf-8')
            logging.debug("System prompt:\n" + system_prompt)
        except Exception as e:
            logging.debug(f"Error reading system prompt from S3: {str(e)}. Will use default system prompt.")
            system_prompt = """
        
You are an expert in extracting key events from a soccer game. You will be given sequence of video frames from a soccer game. Your task is to identify whether some key event happens in these sequence of video frames. 

The possible key events are: shot, corner kick, free kick, foul, offside, injury.
 
Each event has it's own JSON structure as followed. Try to capture the information from the video frames and fill in the JSON structure accordingly. 

If you cannot detect any event, output nothing. Don't explain anything.

If you detect the event, output the JSON structure with the key_event field filled in with the event type you detected. Only output the JSON structure for the event you detected. Don't explain anything.

shot => 
{
   "key_event" : "shot",
   "player_nbr" : 7,
   "jersey_color" : "red",
   "event_interval" : string,
   "game_clock" : "02:33",
   "team_name": "FC Bayern",
   "key_event_prediction_confident_score" : int
}

corner kick =>
{
   "key_event" : "corner_kick",
   "corner_side" : string(left|right),
   "player_nbr" : int,
   "jersey_color" : "white",
   "event_interval" : string,
   "game_clock" : "04:13",
   "team_name": "RB Leipzig",
   "replay": False,
   "key_event_prediction_confident_score" : int
}

free kick =>
{
   "key_event" : "free_kick",
   "player_nbr" : int,
   "jersey_color" : "white",
   "event_interval" : string,
   "game_clock" : "12:55",
   "replay": True,
   "team_name": "RB Leipzig",
   "key_event_prediction_confident_score" : int
}

foul =>
{
   "key_event" : "foul",
   "player_nbr" : int,
   "offending_player_jersey_color" : "red",
   "is_yellow_card" : boolean,
   "is_red_card" : boolean,
   "is_penalty" : boolean
   "event_interval" : string,
   "game_clock" : "42:23",
   "team_name": "FC Bayern",
   "replay": False,
   "key_event_prediction_confident_score" : int
}

offside =>
{
   "key_event" : "offside",
   "player_nbr" : int,
   "jersey_color" : "red",
   "event_interval" : string,
   "game_clock" : "32:01",
   "replay": False,
   "team_name": "FC Bayern",
   "key_event_prediction_confident_score" : int
}

injury =>
{
   "key_event" : "injury",
   "player_nbr" : int,
   "injured_player_jersey_color" : "white"
   "event_interval" : string,
   "game_clock" : "55:31",
   "replay": False,
   "team_name": "RB Leipzig",
   "key_event_prediction_confident_score" : int
}

Capture the game clock ONLY if it's visible in the video frames. Game clock is located on the upper left corner of a video frame. DO NOT use any other means to capture the Game clock. If you cannot determine the Game clock, set its value as "none".

Here are a comprehensive and strict guidelines for identify each key event. If you detected Goal, Shot on target, Shot off target, mark the event type as Shot.

### Foul
- A foul could either be a NORMAL foul, a YELLOW CARD or a RED CARD. 
- To help you determine the type of foul, you must first identify the referee or linesman in the video frames. You must look closely as they could be small and hard to detect. Identify them by the color of their shirts and pants. The referee and the linesman wear turquoise color shirt and black shorts. If referee or linesman is visible, describe their actions. If the referee runs towards the players, or raises his arm, it signals a foul.
- If in your description you mentioned a flag is raised, or a yellow card, or red card is visible, it is a strong indicator of a foul.
- In situation where the referee approaches players it is highly like a foul being called, even though no cards are presented. 
- Even a minor foul is a considered a key event.
- In sequence of frames where the referee approaches players quickly it is highly a foul is called, even though no cards are presented at the moment yet.
- Do not assume any yellow or red card unless you saw the referee actually raised the card in the image frames. Do not make any assumptions.

### Offside
- To help you determine an offside, you must identify whether the linesman is shown any of the video frames. The referee and the linesman wear turquoise color shirt and black shorts. If the linesman is detected, describe his actions. If a flag is raised by the linesman, and it's not obvious that the ball is out, then it is a strong indicator of an offside call.

### Goal
- Pay close attention to the ball location on the field. Keep track of the ball locations through the sequence image frames. If the ball travels into the back of the net, it's very likely a goal is scored.
- Hints like a player celebration that follows is a strong indicator that a goal is scored.

### Free Kick
- Pay attention whether the ball is idle on the ground. If it's idle, then check to see if there are at least 1 player is lined up to a set piece. If it's the case, then it's a strong indication of a free kick.
- Pay close attention to the formation of other players on the field. A free kick usually have at least 2 players forming a wall as a defensive play. If a defensive play wall is identified, it's a strong indicator of a free kick.
- Free kick does not start from the position next to a corner flag pole. Do not confuse free kick with corner kicks.

### Corner kick
- Identify the location of the ball in the image frame sequence. A corner kick scene must have the ball and the player positioned at the extreme corner edge of the field, next to the corner flag pole.
- The corner flag must be visible in the image frames to consider a corner kick. Because you have misidentified corner flag poles before, you must describe why you think you see the corner flag pole on the image frames to make sure it's in the image. Do not make any assumptions. Do not confuse the lines on the field with the corner flags pole. The corner flag pole has a flag at the top of the flag stick. You must be 100% sure about seeing the corner flag pole before making any decisions.
- Use the combined appearance of the corner flag pole, the ball and the player as a strong indicator for a corner kick. 
- If the corner flag pole is not visible in the image frames, it is not a corner kick. you MUST NOT make any assumptions.


### Injury
- When determining an injury, one or both players should be laying on the ground in pain. 
- If there are multiple players surrounding the injured player, it's a strong indicator of an injury occured.


### Shot On Target
- Identify the goalkeeper first. You should identify the goalie by their outfits. The goalkeeper for Bayern Munich wears light green long sleeves, light green shorts and light green socks. The goalkeeper for RB Leipzig wears bright yellow long sleeves, bright yellow shorts and bright yellow socks.
- If the goalkeeper is visible in the image frames, pay close attention to his action. Describe what the goalkeeper is doing in the image frames. If the image frames suggest he caught the ball, or dived to deflect it from a goal, it is a shot on goal target.
- Be aware that the action of the goalkeeper might not be obvious because the camera did not zoom in on him. You must pay detail attention to the goalkeeper to identify his action.
- In addition to goalkeeper's action, you must also identify the position of the ball. It's important to identify through the continuous sequence of images that the ball is moving towards the goal. Tracking the ball direction is important to determine whether a shot on target.

### Shot off target  
- You must fist identify the position of the ball and the direction where it is traveling. It's important to identify through the continuous sequence of images that the ball is moving towards the goal. 
- Once you have identified that the ball moves in the direction of the goal, you must pay close attention find image frames of the player who kicks the ball towards the goal post. 
- The goalkeeper must be present in the image frames. Describe his action through the sequence of image frames. If he did not attempt to save the ball, it's a strong indicator that it's a shot off target.


Other important tips are described in the following :

- You must analyze the given commentary very carefully to determine whether it is useful in determining the key event, do not make up any assumptions or guesses because the given commentary could have incomplete sentences. For example, do not assume there is a key event just because the commentary mentions a player's name.
- You should use the images to identify key events first, then use commentary if you are unsure.
- If the goalkeeper, or the goal post is not in video frames, you must determine that there are no shots taken.
- Capture if you are seeing a Replay. You must look closely to see graphic overlays. These graphic overlays are a grid of 3d white boxes with the word "SUPERCUP" in the center box. If you find a frame or frames having this, assign True to the "replay" attribute in the JSON structure. If not, assign False.
- Pay attention to the jersey colors of players. If the jersey color of the player_nbr has the color "red", assign the "team_name" as "RB Leipzig".If the jersey color of the player_nbr has the color "white", assign the "team_name" as "FC Bayern".

Tips to achieve high confidence in predicting the correct key event:

- Provide a confident score (0 to 100) to indicate your confident level of the predicted key event after going through your analysis. 
- You should come up with this confident scores after analyzing all the frames. 
- Confident scores equal or higher than 80 is good enough to predict any key event. 
- Do your best to provide the confident score, it will help human to determine whether the predicted key event is relevant.

You need to consider all the video frames as a whole to effectively detect the event. You must not make any assumptions.

Only return the key events in JSON format defined above. There should only be 1 key event for the given video frames. Do not provide any other further explanations.."""
        
        
        task_prompt =   "Analyze the given sequence of video frames"
        
        vqa_response = self.call_vqa(image_data=[base64.b64encode(image).decode("utf-8") for image in image_list], system_prompt = system_prompt, task_prompt=task_prompt) 
        
        # Log the vqa_response
        logging.debug(f"Warning: VQA Response for timestamp {timestamp_millis}: {vqa_response}")

        # use scenes to store the key event
        self.visual_scenes[timestamp_millis] = vqa_response

    def _batch_frames(self, iterable, n):
        "Batch data into lists of length n. The last batch may be shorter."
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, n))
            if not batch:
                return
            yield batch        

    def extract_scenes_from_vqa(self):         
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_degree*15) as executor:
            batches = list(self._batch_frames(self.frame_bytes, int(number_of_frames_to_llm)))
            executor.map(self._extract_scene_from_vqa, batches)
    
    def wait_for_dependencies(self):
        if label_detection_enabled:
            self.wait_for_rekognition_label_detection(sort_by="TIMESTAMP")
        if transcription_enabled:
            self.wait_for_transcription_job()

    def run(self):
        self.download_video_and_load_metadata()
        
        if label_detection_enabled:
            self.iterate_object_detection_result()
        
        print("Extracting frames...")
        self.extract_frames()
        
        if label_detection_enabled:
            self.detect_faces_and_celebrities()
        
        print("Extracting scenes from VQA...")
        self.extract_scenes_from_vqa()
        
        if transcription_enabled:
            self.fetch_transcription()
        return self.visual_objects, self.visual_scenes, self.visual_captions, self.visual_texts, self.transcript, self.celebrities, self.faces

class VideoPreprocessorBedrockVQA(VideoPreprocessor):
    config = Config(read_timeout=1000) # Extends botocore read timeout to 1000 seconds
    bedrock_client = boto3.client(service_name="bedrock-runtime", config=config)
    
    def __init__(self, 
        label_detection_job_id: str,
        transcription_job_name: str,
        bucket_name: str,
        video_s3_path: str,
        video_transcript_s3_path: str,
        frame_interval: str,
        vqa_model_name: str
        ):

        super().__init__(label_detection_job_id=label_detection_job_id,
            transcription_job_name=transcription_job_name, 
            bucket_name=bucket_name,
            video_s3_path=video_s3_path,
            video_transcript_s3_path=video_transcript_s3_path,
            frame_interval=frame_interval
        )

        self.vqa_model_name = vqa_model_name
        self.llm_parameters = {
            "anthropic_version": "bedrock-2023-05-31",    
            "max_tokens": 1000,
            "temperature": 0.1,
            "top_k": 3,
        }
    
    def call_vqa(self, image_data: list[bytes], system_prompt: str, task_prompt: str) -> str:
        self.llm_parameters["system"] = system_prompt
        self.llm_parameters["messages"] = [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": task_prompt }
                ]
            }
        ]
        
        # Add each image to the content list
        for img in image_data:
            self.llm_parameters["messages"][0]["content"].insert(0, 
                { "type": "image", "source": { "type": "base64", "media_type": "image/jpeg", "data": img } }
            )
        
        encoded_input = json.dumps(self.llm_parameters).encode("utf-8")
        call_done = False
        while(not call_done):
            try:
                bedrock_response = self.bedrock_client.invoke_model(body=encoded_input, modelId=self.vqa_model_name)
                call_done = True
            except self.bedrock_client.exceptions.ThrottlingException as e:
                print("Amazon Bedrock throttling exception")
                time.sleep(60)
            except Exception as e:
                raise e

        response: str = json.loads(bedrock_response.get("body").read())["content"][0]["text"]
        return response
    
class VideoAnalyzer(ABC):
    def __init__(self, 
        bucket_name: str,
        video_name: str, 
        video_path: str, 
        visual_objects: dict[int, list[str]],
        visual_scenes: dict[int, str], 
        visual_captions: dict[int, str],
        visual_texts: dict[int, list[str]], 
        transcript: dict,
        celebrities: dict[int, list[str]],
        faces: dict[int, list[str]],
        summary_folder: str,
        entity_sentiment_folder: str,
        video_script_folder: str,
        transcription_job_name: str,
        ):

        self.s3_client = boto3.client("s3")
        self.transcribe_client = boto3.client("transcribe")
        self.bucket_name: str = bucket_name
        self.summary_folder: str = summary_folder
        self.entity_sentiment_folder: str = entity_sentiment_folder
        self.transcription_job_name: str = transcription_job_name
        self.video_script_folder: str = video_script_folder
        self.video_caption_folder: str = video_caption_folder
        self.video_name: str = video_name
        self.video_path: str = video_path
        self.original_visual_objects: dict[int, list[ObjectFinding]] = visual_objects
        self.original_visual_scenes: dict[int, str] = visual_scenes
        self.original_visual_captions: dict[int, str] = visual_captions
        self.original_visual_texts: dict[int, list[str]] = visual_texts
        self.original_transcript: dict = transcript
        self.original_celebrities: dict[int, list[CelebrityFinding]] = celebrities
        self.original_faces: dict[int, list[FaceFinding]] = faces
        self.visual_objects: list[list[Union[int, list[str]]]] = []
        self.visual_scenes: list[list[Union[int, str]]] = []
        self.visual_captions: list[list[Union[int, str]]] = []
        self.visual_texts: list[list[Union[int, list[str]]]] = []
        self.transcript: list[list[Union[int, str]]] = []
        self.celebrities:list[list[Union[int, list[str]]]]  = []
        self.faces: list[list[Union[int, list[str]]]]  = []
        self.video_script_chunk_size_for_summary_generation: int = 100000 # characters
        self.video_script_chunk_overlap_for_summary_generation: int = 500 # characters
        self.video_script_chunk_size_for_entities_extraction: int = 50000 #10000 # characters
        self.video_script_chunk_overlap_for_entities_extraction: int = 500 #200 # characters
        self.embedding_storage_chunk_size: int = 2048 #10000 # Number of characters, which depends on the embedding model
        self.text_similarity_score: float = 0.9
        self.objects_similarity_score: float = 0.8
        self.video_rolling_summary: str = ""
        self.video_rolling_sentiment: str = ""
        self.combined_video_script: str = ""
        self.all_combined_video_script: str = ""
        self.combined_visual_captions: str = ""
        self.llm_parameters: dict = {}
        self.summary: str = ""
        self.entities: str = ""
        self.video_script: str = ""
        self.language_code: str = ""
    
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
    def call_llm(self, system_prompt, prompt, prefilled_response):
        pass
    
    @abstractmethod
    def call_embedding_llm(self, document):
        pass
    
    def preprocess_visual_objects(self):
        visual_objects_across_timestamps = dict(sorted(copy.deepcopy(self.original_visual_objects).items()))

        timestamp_millis: int
        visual_objects_at_particular_timestamp: list
        for timestamp_millis, visual_objects_at_particular_timestamp in list(visual_objects_across_timestamps.items()):
            visual_objects_at_particular_timestamp.sort(key=lambda object_finding: object_finding.confidence_score, reverse=True)
            
            # Find the top N of the identified visual objects who have confidence score >= threshold, and return the object label name only
            top_n_threshold = ObjectFinding.top_n_threshold
            visual_objects_at_particular_timestamp = [object_finding.display() for object_finding in filter(lambda obj: obj.confidence_score >= ObjectFinding.confidence_score_threshold, visual_objects_at_particular_timestamp )][:top_n_threshold]
      
        self.visual_objects = sorted(visual_objects_across_timestamps.items())
        # TODO : Store this in DB and use in analytics
  
    def preprocess_visual_scenes(self):
        self.visual_scenes =  sorted(self.original_visual_scenes.items())

    def preprocess_visual_captions(self):
        self.visual_captions =  sorted(self.original_visual_captions.items())
      
    def preprocess_visual_texts(self):
        visual_texts_across_timestamps = dict(sorted(self.original_visual_texts.items()))
        prev_texts: list = []

        timestamp_millis: int
        visual_texts_at_particular_timestamp: list
        for timestamp_millis, visual_texts_at_particular_timestamp in list(visual_texts_across_timestamps.items()):
            # The loop below is to check how much of this text resembling previous text
            num_of_matches_with_prev_texts: int = 0
            text: str
            for text in visual_texts_at_particular_timestamp:
                if text in prev_texts: num_of_matches_with_prev_texts += 1
            #  Delete text entry if there is not detected text
            if len(visual_texts_at_particular_timestamp) == 0: 
                del visual_texts_across_timestamps[timestamp_millis]
            # Delete text entry if the detected texts are too similar with the previous scene
            elif float(num_of_matches_with_prev_texts) > len(prev_texts)*self.text_similarity_score:
                del visual_texts_across_timestamps[timestamp_millis]
            else:
                prev_texts = visual_texts_at_particular_timestamp
      
        self.visual_texts = sorted(visual_texts_across_timestamps.items())

    def preprocess_transcript(self):
        transcript= dict()
        previous_millis: int = 0
        if "results" in self.original_transcript and "items" in self.original_transcript['results']:
            for item in self.original_transcript["results"]["items"]:
                if item['type'] == 'punctuation':
                    current_millis = previous_millis + 1 # Just add 1 millisecond to avoid this punctuation replaces the previous word.
                    transcript[current_millis] = item['alternatives'][0]['content']
                else:
                    try:
                        time_millis: int = int(float(item['start_time'])*1000) # In millisecond
                        content: str = ""
                        
                        if "speaker_label" in item:
                            match = re.search(r"spk_(\d+)", item['speaker_label'])
                            speaker_number = int(match.group(1)) + 1 # So that it starts from 1, not 0
                            content += f" Speaker {speaker_number} "
                        if "language_code" in item:
                            content += f"in {item['language_code']}"

                        content += f": {item['alternatives'][0]['content']}"

                        transcript[time_millis] = content
                        previous_millis = time_millis
                    except Exception as e:
                        print("Error in transcribing") 
                        raise e

        self.transcript = sorted(transcript.items())

    def preprocess_celebrities(self):
        self.celebrities = sorted(self.original_celebrities.items())

    def preprocess_faces(self):
        self.faces = sorted(self.original_faces.items())
    
    def generate_visual_captions(self):
        def transform_captions(x):
            timestamp = round(x[0]/1000, 1) # Convert millis to second
            caption = x[1]
            return (timestamp, f"Caption:{caption}")
        captions = list(map(transform_captions, self.visual_captions))

        def transform_transcript(x):
            timestamp = round(x[0]/1000, 1) # Convert millis to second
            transcript = x[1]
            return (timestamp, f"Voice:{transcript}")
        transcript  = list(map(transform_transcript, self.transcript))

        # Combine all inputs
        combined_visual_captions = sorted( captions + transcript )
        
        combined_visual_captions = "\n".join(list(map(lambda x: f"{x[0]}:{x[1]}", combined_visual_captions)))
        
        self.combined_visual_captions = combined_visual_captions
      
    def generate_combined_video_script(self):
        def transform_objects(x):
            timestamp = round(x[0]/1000, 1) # Convert millis to second
            objects = ",".join(x[1])
            return (timestamp, f"Objects:{objects}")
        objects  = list(map(transform_objects, self.visual_objects))

        def transform_scenes(x):
            timestamp = round(x[0]/1000, 1) # Convert millis to second
            scene = x[1]
            return (timestamp, f"Scene:{scene}")
        scenes  = list(map(transform_scenes, self.visual_scenes))

        def transform_texts(x):
            timestamp = round(x[0]/1000, 1) # Convert millis to second
            texts = ",".join(x[1])
            return (timestamp, f"Texts:{texts}")
        visual_texts  = list(map(transform_texts, self.visual_texts))

        def transform_transcript(x):
            timestamp = round(x[0]/1000, 1) # Convert millis to second
            transcript = x[1]
            return (timestamp, f"Voice:{transcript}")
        transcript  = list(map(transform_transcript, self.transcript))

        def transform_celebrities(x):
            timestamp = round(x[0]/1000, 1) # Convert millis to second
            celebrities = ",".join([f"Celeb {(idx+1)}:{celeb_finding.display()}" for idx, celeb_finding in enumerate(x[1])])
            return (timestamp, f"Celebrities:{celebrities}")
        visual_celebrities = list(map(transform_celebrities, self.celebrities))

        def transform_faces(x):
            timestamp = round(x[0]/1000, 1) # Convert millis to second
            faces = ",".join([f"Face {(idx+1)}:{face_finding.display()}" for idx, face_finding in enumerate(x[1])])
            return (timestamp, f"Faces:{faces}")
        visual_faces = list(map(transform_faces, self.faces))
 
        # Combine all inputs
        combined_video_script = sorted( objects + scenes + visual_texts + transcript + visual_celebrities + visual_faces )
        
        combined_video_script = "\n".join(list(map(lambda x: f"{x[0]}:{x[1]}", combined_video_script)))
        
        self.combined_video_script = combined_video_script
        self.all_combined_video_script += combined_video_script
  
    def get_language_code(self):
        language_code: str = 'en'

        if transcription_enabled and self.transcription_job_name != "":
            get_transcription = self.transcribe_client.get_transcription_job(TranscriptionJobName=self.transcription_job_name)
        
            language_code_validity_duration_threshold: float = 2.0 # Only consider the language code as valid if the speech is longer than 2 seconds, otherwise it might be invalid data.
            
            if "LanguageCodes" in get_transcription["TranscriptionJob"] and get_transcription["TranscriptionJob"]["TranscriptionJobStatus"] == "COMPLETED":
                if len(get_transcription["TranscriptionJob"]["LanguageCodes"]) > 0:
                    if float(get_transcription["TranscriptionJob"]["LanguageCodes"][0]["DurationInSeconds"]) >= language_code_validity_duration_threshold:
                        language_code = get_transcription["TranscriptionJob"]["LanguageCodes"][0]["LanguageCode"]
                        if "-" in language_code: language_code = language_code.split("-")[0] # Get only the language code and ignore the dialect information
        
        self.language_code = language_code
        
        return language_code

    def prompt_translate(self):
        language_code = self.get_language_code() if self.language_code == "" else self.language_code

        if language_code == "en":
            return f"You are a native speaker of this language code '{language_code}' and your answer MUST be in '{language_code}'"
        else:
            return f"You are a native speaker of this language code '{language_code}' and your answer MUST be in '{language_code}', not 'en'"

    def generate_summary(self):
        system_prompt = "You are an expert video analyst who reads a Video Timeline and creates summary of the video.\n" \
                        "The Video Timeline is a text representation of a video.\n" \
                        "The Video Timeline contains the visual scenes, the visual texts, human voice, celebrities, and human faces in the video.\n" \
                        "Visual objects (objects) represents what objects are visible in the video at that second. This can be the objects seen in camera, or objects from a screen sharing, or any other visual scenarios.\n" \
                        "Visual scenes (scene) represents the description of how the video frame look like at that second.\n" \
                        "Visual texts (text) are the text visible in the video. It can be texts in real world objects as recorded in video camera, or those from screen sharing, or those from presentation recording, or those from news, movies, or others. \n" \
                        "Human voice (voice) is the transcription of the video.\n" \
                        "Celebrities (celebrity) provides information about the celebrity detected in the video at that second. It may also has information on where the face is located relative to the video frame size. The celebrity may not be speaking as he/she may just be portrayed. \n" \
                        "Human faces (face) lists the face seen in the video at that second. This may have information about the facial features and the face location relative to the video frame. \n" \
                        f"{self.prompt_translate()}\n"

        video_script_length = len(self.combined_video_script)
        prefilled_response = "Here is the summary of the video:"

        # When the video is short enough to fit into 1 chunk
        if video_script_length <= self.video_script_chunk_size_for_summary_generation:
            core_prompt = f"The VIDEO TIMELINE has format below.\n" \
                            "timestamp in seconds:scene / text / voice\n" \
                            "<Video Timeline>\n" \
                            f"{self.combined_video_script}\n" \
                            "</Video Timeline>\n"
          
            prompt = f"{core_prompt}\n\n" \
            "<Task>\n" \
            "Describe the summary of the video in paragraph format.\n" \
            "You can make reasonable extrapolation of the actual video given the Video Timeline.\n" \
            "DO NOT mention 'Video Timeline' or 'video timeline'.\n" \
            f"{self.prompt_translate()}\n" \
            "</Task>\n\n"

            self.video_rolling_summary = self.call_llm(system_prompt, prompt, prefilled_response, stop_sequences=["<Task>"])
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
                        "timestamp in milliseconds:scene / text / voice\n" \
                        "<Video Timeline>\n" \
                        f"{chunk_combined_video_script}\n" \
                        "</Video Timeline>\n"
                
                if is_last_chunk:
                    prompt = f"The video has {number_of_chunks} parts.\n\n" \
                    f"The below Video Timeline is only for the part {chunk_number+1} of the video, which is the LAST part.\n\n" \
                    f"{core_prompt}\n\n" \
                    "Below is the summary of all previous part/s of the video:\n\n" \
                    f"{self.video_rolling_summary}\n\n" \
                    "<Task>\n" \
                    "Describe the summary of the whole video in paragraph format.\n" \
                    "You can make reasonable extrapolation of the actual video given the Video Timeline.\n" \
                    "DO NOT mention 'Video Timeline' or 'video timeline'.\n" \
                    "Give the summary directly without any other sentence.\n" \
                    f"{self.prompt_translate()}\n" \
                    "</Task>\n\n"
                    
                    chunk_summary = self.call_llm(system_prompt, prompt, prefilled_response, stop_sequences=["<Task>"])
                    self.video_rolling_summary = chunk_summary
                elif is_first_chunk:
                    prompt = f"The video has {number_of_chunks} parts. The below Video Timeline is only for the first part.\n\n" \
                    f"{core_prompt}\n\n" \
                    "<Task>\n" \
                    "Describe the summary of the first part of the video in paragraph format.\n" \
                    "You can make reasonable extrapolation of the actual video given the Video Timeline.\n" \
                    "DO NOT mention 'Video Timeline' or 'video timeline'.\n" \
                    "Give the summary directly without any other sentence.\n" \
                    f"{self.prompt_translate()}\n" \
                    "</Task>\n\n"
                    
                    chunk_summary = self.call_llm(system_prompt, prompt, prefilled_response, stop_sequences=["<Task>"])
                    self.video_rolling_summary = chunk_summary
                else:
                    prompt = f"The video has {number_of_chunks} parts.\n\n" \
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
                    f"{self.prompt_translate()}\n" \
                    "</Task>\n\n"
                    
                    chunk_summary = self.call_llm(system_prompt, prompt, prefilled_response, stop_sequences=["<Task>"])
                    self.video_rolling_summary = chunk_summary

        return self.video_rolling_summary
    
    def extract_sentiment(self):
        system_prompt = "You are an expert video analyst who reads a Video Timeline and extract entities and their associated sentiment.\n" \
                        "The Video Timeline is a text representation of a video.\n" \
                        "The Video Timeline contains the visual scenes, the visual objects, visual texts, human voice, celebrities, and human faces in the video.\n" \
                        "Visual objects (objects) represents what objects are visible in the video at that second. This can be the objects seen in camera, or objects from a screen sharing, or any other visual scenarios.\n" \
                        "Visual scenes (scene) represents the description of how the video frame look like at that second.\n" \
                        "Visual texts (text) are the text visible in the video. It can be texts in real world objects as recorded in video camera, or those from screen sharing, or those from presentation recording, or those from news, movies, or others. \n" \
                        "Human voice (voice) is the transcription of the video.\n" \
                        "Celebrities (celebrity) provides information about the celebrity detected in the video at that second. It may also has information on where the face is located relative to the video frame size. The celebrity may not be speaking as he/she may just be portrayed. \n" \
                        "Human faces (face) lists the face seen in the video at that second. This may have information about the facial features and the face location relative to the video frame. \n"
                        
        video_script_length = len(self.combined_video_script)

        prefilled_response = "Here are the entities I extracted:"

        # When the video is short enough to fit into 1 chunk
        if video_script_length <= self.video_script_chunk_size_for_entities_extraction:
            core_prompt = f"The Video Timeline has a format below.\n" \
                            "timestamp in seconds:scene / text / voice\n" \
                            "<Video Timeline>\n" \
                            f"{self.combined_video_script}\n" \
                            "</Video Timeline>\n"
          
            prompt = f"{core_prompt}\n\n" \
            f"To help you, here is the summary of the whole video.\n\n{self.video_rolling_summary}\n\n" \
            "<Task>\n" \
            "Now your job is to infer and list the entities, their sentiment [positive, negative, mixed, neutral], and the sentiment's reason.\n" \
            "You will never see the video. This video timeline is the best you get. You can make reasonable extrapolation of the actual video given the Video Timeline.\n" \
            "Entities can be a person, company, country, concept, brand, or anything where audience may be interested in knowing the trend.\n" \
            "You MUST ONLY list important entities of interest, not every entity you found.\n" \
            "For person or celebrity or individual, DO NOT give sentiment rating. Put N/A for the sentiment and reason fields.\n" \
            "Sentiment's reason MUST justify the sentiment. For no meaningful reason, just put N/A for the sentiment's reason.\n" \
            "Each row of your answer MUST be of this format entity|sentiment|sentiment's reason. Follow the below example.\n\n" \
            "Entities:\n" \
            "mathematic|negative|The kid interviewed in the video seems to be really afraid of mathematics as his grade is always struggling.\n" \
            "Rudy Donna|N/A|Rudy Donna is the interviewee's friend. No sentiment score is provided for person/individual.\n" \
            "teacher|positive|Despite the kid having not so good grades, he seems to like all his teachers in school as they are all patient.\n" \
            "extracurricular activities|neutral|N/A.\n" \
            "examination|mixed|While the interviewee dreads examination, he likes the challenge of it.\n\n" \
            "STRICTLY FOLLOW the format above. DO NOT mention Video Timeline or video timeline.\n" \
            "</Task>\n\n"
          
            self.video_rolling_sentiment = self.call_llm(system_prompt, prompt, prefilled_response, temperature=0.1, stop_sequences=["<Task>"])
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
                    prompt = f"The video has {number_of_chunks} parts. The below Video Timeline is only for part {chunk_number+1} of the video, which is the LAST part.\n\n" \
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
                    "</Task>\n\n"
                    
                    chunk_sentiment = self.call_llm(system_prompt, prompt, prefilled_response, temperature=0.01, stop_sequences=["<Task>"])
                    self.video_rolling_sentiment = chunk_sentiment
                elif is_first_chunk:
                    prompt = f"The video has {number_of_chunks} parts. The below Video Timeline is only for the first part.\n\n" \
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
                    "</Task>\n\n"
                    
                    chunk_sentiment = self.call_llm(system_prompt, prompt, prefilled_response, temperature=0.01, stop_sequences=["<Task>"])
                    self.video_rolling_sentiment = chunk_sentiment
                else:
                    prompt = f"The video has {number_of_chunks} parts. The below Video Timeline is only for part {chunk_number+1} of the video.\n\n" \
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
                    "</Task>\n\n"
                    
                    chunk_sentiment = self.call_llm(system_prompt, prompt, prefilled_response, temperature=0.01, stop_sequences=["<Task>"])
                    self.video_rolling_sentiment = chunk_sentiment
                
        return self.video_rolling_sentiment
    
    def store_summary_result(self):
        # Store summary in S3
        self.s3_client.put_object(
            Body=self.summary, 
            Bucket=self.bucket_name, 
            Key=f"{self.summary_folder}/{self.video_path}.txt"
        )

        summary_embedding = self.call_embedding_llm(self.summary[:self.embedding_storage_chunk_size])

        # Store summary in database
        update_stmt = (
            update(self.Videos).
            where(self.Videos.name == self.video_path).
            values(summary = self.summary, summary_embedding = summary_embedding)
        )
        
        session.execute(update_stmt)
        session.commit()

    def store_sentiment_result(self):
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
        number_of_chunks = math.ceil( (video_script_length + 1) / self.embedding_storage_chunk_size)

        chunks: list[self.Contents] = []
        for chunk_number in range(0, number_of_chunks):
            is_last_chunk = (chunk_number == (number_of_chunks - 1))
            is_first_chunk = (chunk_number == 0)

            start = 0 if is_first_chunk else int(chunk_number*self.embedding_storage_chunk_size)
            stop = video_script_length if is_last_chunk else (chunk_number+1)*self.embedding_storage_chunk_size
            chunk_string = self.video_script[start:stop]
        
            # So long as this is not the first chunk, remove whatever before first \n since likely the chunk cutting is not done exactly at the \n
            if not is_first_chunk:
                try:
                    chunk_string = chunk_string[chunk_string.index("\n"):] 
                except:
                    pass
            
            # Get the embedding for the chunk
            chunk_embedding = self.call_embedding_llm(chunk_string)
            
            # Create database object
            chunks.append(self.Contents(
                chunk=chunk_string,
                chunk_embedding=chunk_embedding,
                video_name=self.video_path
            ))

        # Store in database
        session.add_all(chunks)
        session.commit()
    
    def store_video_visual_captions(self):
        self.s3_client.put_object(
            Body=self.combined_visual_captions, 
            Bucket=self.bucket_name, 
            Key=f"{self.video_caption_folder}/{self.video_path}.txt"
        )

    def run(self):
        self.preprocess_visual_scenes()
        self.preprocess_visual_captions()
        self.preprocess_visual_texts()
        self.preprocess_transcript()
        self.preprocess_celebrities()
        self.preprocess_faces()

        self.generate_combined_video_script()
        self.generate_visual_captions()

        self.summary = self.generate_summary()
        self.entities = self.extract_sentiment()
        self.video_script = self.all_combined_video_script
    
    def store(self):
        self.store_video_visual_captions()
        self.store_video_script_result()
        self.store_summary_result()
        self.store_sentiment_result()
        

class VideoAnalyzerBedrock(VideoAnalyzer):    
    def __init__(self, 
        model_name: str,
        embedding_model_name: str,
        bucket_name: str,
        video_name: str, 
        video_path: str, 
        visual_objects: dict[int, list[str]],
        visual_scenes: dict[int, str], 
        visual_captions: dict[int, str],
        visual_texts: dict[int, list[str]], 
        transcript: dict,
        celebrities: dict[int, list[str]],
        faces: dict[int, list[str]],
        summary_folder: str,
        entity_sentiment_folder: str,
        video_script_folder: str,
        transcription_job_name: str
        ):
        super().__init__(bucket_name, video_name, video_path, visual_objects, visual_scenes, visual_captions, visual_texts, transcript, celebrities, faces,summary_folder, entity_sentiment_folder, video_script_folder, transcription_job_name)
        
        config = Config(read_timeout=1000) # Extends botocore read timeout to 1000 seconds
        self.bedrock_client = boto3.client(service_name="bedrock-runtime", config=config)

        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.llm_parameters = {
            "anthropic_version": "bedrock-2023-05-31",    
            "max_tokens": 2000,
            "temperature": 0.7,
            "top_k": 10,
            "stop_sequences": []
        }
    
    def call_llm(self, system_prompt: str, prompt: str, prefilled_response: str, temperature=None, top_k=None, stop_sequences=[]) -> str:
        self.llm_parameters["system"] = system_prompt
        self.llm_parameters["messages"] = [
            {
                "role": "user",
                "content":  prompt
            },
            { "role": "assistant", "content": prefilled_response},
        ]
        if temperature is not None:
            self.llm_parameters['temperature'] = temperature
        if top_k is not None:
            self.llm_parameters['top_k'] = top_k
        if stop_sequences is not None:
            self.llm_parameters['stop_sequences'] += stop_sequences
        
        encoded_input = json.dumps(self.llm_parameters).encode("utf-8")
        call_done = False
        while(not call_done):
            try:
                bedrock_response = self.bedrock_client.invoke_model(body=encoded_input, modelId=self.model_name)
                call_done = True
            except self.bedrock_client.exceptions.ThrottlingException as e:
                print("Amazon Bedrock throttling exception")
                time.sleep(60)
            except Exception as e:
                raise e

        response: str = json.loads(bedrock_response.get("body").read())["content"][0]["text"]
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
            except self.bedrock_client.exceptions.ThrottlingException:
                print("Amazon Bedrock throttling exception")
                time.sleep(60)
            except Exception as e:
                raise e

        # Disabling semgrep rule for checking data size to be loaded to JSON as the source is from Amazon Bedrock
        # nosemgrep: python.aws-lambda.deserialization.tainted-json-aws-lambda.tainted-json-aws-lambda
        embedding = json.loads(response.get("body").read().decode())["embeddings"][0] #["embedding"]
        return embedding

def handler():
    video_name = os.path.basename(video_s3_path)
    video_path= '/'.join(video_s3_path.split('/')[1:])
    video_transcript_s3_path = f"{transcription_folder}/{video_path}.txt"

    try:
        # Initiate class for video preprocessing
        video_preprocessor = VideoPreprocessorBedrockVQA( 
            label_detection_job_id=label_detection_job_id,
            transcription_job_name=transcription_job_name,
            bucket_name=bucket_name,
            video_s3_path=video_s3_path,
            video_transcript_s3_path=video_transcript_s3_path,
            frame_interval=frame_interval,
            vqa_model_name=vqa_model_id
        )
        # Wait for extraction jobs to finish
        video_preprocessor.wait_for_dependencies()
        
        # Preprocess and extract information
        visual_objects, visual_scenes, visual_captions, visual_texts, transcript, celebrities, faces = video_preprocessor.run()
        
        # Initiate class for video analysis
        video_analyzer = VideoAnalyzerBedrock(
            model_name=model_id, 
            embedding_model_name=embedding_model_id,
            bucket_name=bucket_name,
            video_name=video_name,
            video_path=video_path,
            visual_objects=visual_objects,
            visual_scenes=visual_scenes,
            visual_captions=visual_captions,
            visual_texts=visual_texts, 
            transcript=transcript,
            celebrities=celebrities,
            faces=faces,
            summary_folder=summary_folder,
            entity_sentiment_folder=entity_sentiment_folder,
            video_script_folder=video_script_folder,
            transcription_job_name=transcription_job_name
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

if __name__ == "__main__":
    handler()
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

secrets_manager = boto3.client('secretsmanager')

model_id = os.environ["MODEL_ID"]
vqa_model_id = os.environ["VQA_MODEL_ID"]
frame_interval = os.environ['FRAME_INTERVAL']
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
        self.smile: bool = bool(celebrity_dict['Face']['Smile']['Value'])
        self.smile_confidence: int = int(celebrity_dict['Face']['Smile']['Confidence'])
        self.emotions: list[str] = list(filter(lambda em: (len(em) > 0),[emo['Type'].lower() if int(emo['Confidence']) >= self.celebrity_emotion_confidence_threshold else '' for emo in celebrity_dict['Face']['Emotions']]))
        self.top: float = float(celebrity_dict['Face']['BoundingBox']['Top'])
        self.top_display: str = str(round(self.top, 2))
        self.left: float = float(celebrity_dict['Face']['BoundingBox']['Left'])
        self.left_display: str = str(round(self.left, 2))
        self.height: float = float(celebrity_dict['Face']['BoundingBox']['Height'])
        self.height_display: str = str(round(self.height, 2))
        self.width: float = float(celebrity_dict['Face']['BoundingBox']['Width'])
        self.width_display: str = str(round(self.width, 2))

    def is_matching_face(self, bb_top:float , bb_left: float, bb_height: float, bb_width: float) -> bool:
        return (
            (abs(self.top - bb_top) <= self.face_bounding_box_overlap_threshold) and
            (abs(self.left - bb_left) <= self.face_bounding_box_overlap_threshold) and
            (abs(self.height - bb_height) <= self.face_bounding_box_overlap_threshold) and
            (abs(self.width - bb_width) <= self.face_bounding_box_overlap_threshold)
        )

    def display(self) -> str:
        display_string = f"Name is {self.name}. "
        if len(self.emotions) > 0:
            display_string += "Emotions appear to be " + ", ".join(self.emotions) + ". "
        if self.smile_confidence >= self.celebrity_feature_confidence_threshold: 
            display_string += "The celebrity is smiling. " if self.smile else "The celebrity is not smiling at this moment. "
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
        self.smile: bool = bool(face_dict['Smile']['Value'])
        self.smile_confidence: int = int(face_dict['Smile']['Confidence'])
        self.mouthopen: bool = bool(face_dict['MouthOpen']['Value'])
        self.mouthopen_confidence: int = int(face_dict['MouthOpen']['Confidence'])
        self.mustache: bool = bool(face_dict['Mustache']['Value'])
        self.mustache_confidence: int = int(face_dict['Mustache']['Confidence'])
        self.gender: str = str(face_dict['Gender']['Value']).lower()
        self.gender_confidence: int = int(face_dict['Gender']['Confidence'])
        self.emotions: list[str] = list(filter(lambda em: (len(em) > 0),[emo['Type'].lower() if int(emo['Confidence']) >= self.face_emotion_confidence_threshold else '' for emo in face_dict['Emotions']]))

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
        if len(self.emotions) > 0:
            display_string += "Emotion appears to be " + ", ".join(self.emotions) + ". "
        if self.smile_confidence >= self.face_feature_confidence_threshold:
            display_string += "Seems to be smiling. " if self.smile else "Seems to be not smiling. "
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
        self.transcript: dict
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
        while(get_transcription["TranscriptionJob"]["TranscriptionJobStatus"] == 'IN_PROGRESS'):
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
 
    def _extract_scene_from_vqa(self, frame_info: list[Union[int, bytes]]):
        timestamp_millis = frame_info[0]
        image = frame_info[1]

        system_prompt = "You are an expert in extracting information from video frames. Each video frame is an image. You will extract the scene, text, and caption."
        task_prompt =   "Extract information from this image and output a JSON with this format:\n" \
                    "{\n" \
                    "\"scene\" : \"String\",\n" \
                    "\"caption\" : \"String\",\n" \
                    "\"text\" : [\"String\", \"String\" , . . .],\n" \
                    "}\n" \
                    "For \"scene\", look carefully, think hard, and describe what you see in the image in detail, yet succinct. \n" \
                    "For \"caption\", look carefully, think hard, and give a SHORT caption (3-8 words) that best describes what is happening in the image. This is intended for visually impaired ones. \n" \
                    "For \"text\", list the text you see in that image confidently. If nothing return empty list.\n"
        vqa_response = self.call_vqa(image_data=base64.b64encode(image).decode("utf-8"), system_prompt = system_prompt, task_prompt=task_prompt) 

        # Sometimes the response might be censored due to false positive of inappropriate content. When that happens, just skip this frame.
        pattern = r'"scene"\s*:\s*"(.+?)".*?"caption"\s*:\s*"(.+?)".*?"text"\s*:\s*\[(.*?)\]'
        match = re.search(pattern, vqa_response, re.DOTALL)

        if match:
            scene = match.group(1)
            caption = match.group(2)
            text = match.group(3)
            self.visual_scenes[timestamp_millis] = scene
            if len(text) > 0:
                self.visual_texts[timestamp_millis] = [t.strip().strip("\"") for t in text.replace("\n","").split(",")]
            self.visual_captions[timestamp_millis] = caption

            

    
    def extract_scenes_from_vqa(self):
        timestamp_millis: int
        image: bytes
          
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_degree*15) as executor:
            executor.map(self._extract_scene_from_vqa, self.frame_bytes)
    
    def wait_for_dependencies(self):
        self.wait_for_rekognition_label_detection(sort_by="TIMESTAMP")
        self.wait_for_transcription_job()

    def run(self):
        self.download_video_and_load_metadata()
        self.iterate_object_detection_result()
        self.extract_frames()
        self.detect_faces_and_celebrities()
        self.extract_scenes_from_vqa()
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
    
    def call_vqa(self, image_data: bytes, system_prompt: str, task_prompt: str) -> str:
        self.llm_parameters["system"] = system_prompt
        self.llm_parameters["messages"] = [
            {
                "role": "user",
                "content": [
                    { "type": "image", "source": { "type": "base64", "media_type": "image/jpeg", "data": image_data } },
                    { "type": "text", "text": task_prompt }
                ]
            }
        ]
        
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
        get_transcription = self.transcribe_client.get_transcription_job(TranscriptionJobName=self.transcription_job_name)
        language_code: str = 'en-US'
        language_code_validity_duration_threshold: float = 2.0
        
        if "LanguageCodes" in get_transcription["TranscriptionJob"]:
            if len(get_transcription["TranscriptionJob"]["LanguageCodes"]) > 0:
                if float(get_transcription["TranscriptionJob"]["LanguageCodes"][0]["DurationInSeconds"]) >= language_code_validity_duration_threshold:
                    language_code = get_transcription["TranscriptionJob"]["LanguageCodes"][0]["LanguageCode"]
        return language_code

    def prompt_translate(self):
        language_code = self.get_language_code()

        if language_code == "en-US":
            return f"You are a native speaker of {language_code} and your answer must be {language_code}"
        else:
            return f"You are a native speaker of {language_code} and your answer must be {language_code}, not en-US"

    def generate_summary(self):
        system_prompt = "You are an expert video analyst who reads a Video Timeline and creates summary of the video.\n" \
                        "The Video Timeline is a text representation of a video.\n" \
                        "The Video Timeline contains the visual scenes, the visual texts, human voice, celebrities, and human faces in the video.\n" \
                        "Visual objects (objects) represents what objects are visible in the video at that second. This can be the objects seen in camera, or objects from a screen sharing, or any other visual scenarios.\n" \
                        "Visual scenes (scene) represents the description of how the video frame look like at that second.\n" \
                        "Visual texts (text) are the text visible in the video. It can be texts in real world objects as recorded in video camera, or those from screen sharing, or those from presentation recording, or those from news, movies, or others. \n" \
                        "Human voice (voice) is the transcription of the video.\n" \
                        "Celebrities (celebrity) provides information about the celebrity detected in the video at that second. It may have the information on whether the celebrity is smiling and the captured emotions. It may also has information on where the face is located relative to the video frame size. The celebrity may not be speaking as he/she may just be portrayed. \n" \
                        "Human faces (face) lists the face seen in the video at that second. This may have information on the emotions, face location relative to the video frame, and more facial features. \n" \
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
                        "Celebrities (celebrity) provides information about the celebrity detected in the video at that second. It may have the information on whether the celebrity is smiling and the captured emotions. It may also has information on where the face is located relative to the video frame size. The celebrity may not be speaking as he/she may just be portrayed. \n" \
                        "Human faces (face) lists the face seen in the video at that second. This may have information on the emotions, face location relative to the video frame, and more facial features. \n"
                        
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
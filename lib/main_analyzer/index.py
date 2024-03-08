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

config = Config(read_timeout=1000) # Extends botocore read timeout to 1000 seconds

rekognition = boto3.client('rekognition')
transcribe = boto3.client("transcribe")
secrets_manager = boto3.client('secretsmanager')
bedrock = boto3.client(service_name="bedrock-runtime", config=config)
s3 = boto3.client("s3")

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
        display_string = self.name
        if len(self.emotions) > 0:
            display_string += "|" + "-".join(self.emotions)
        if self.smile_confidence >= self.celebrity_feature_confidence_threshold: 
            display_string += "|smiling" if self.smile else "|not smiling"
        display_string += f"|face is located {self.left_display} from left - {self.top_display} from top - with height {self.height_display} and width {self.width_display} of the video frame"

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
        display_string = f"{self.age_low}-{self.age_high} years old"
        if self.gender_confidence >= self.face_feature_confidence_threshold:
            display_string += f"|{self.gender}"
        if len(self.emotions) > 0:
            display_string += "|" + "-".join(self.emotions)
        if self.smile_confidence >= self.face_feature_confidence_threshold:
            display_string += "|smiling" if self.smile else "|not smiling"
        if self.beard_confidence >= self.face_feature_confidence_threshold:
            display_string += "|has beard" if self.beard else "|no beard"
        if self.mustache_confidence >= self.face_feature_confidence_threshold:
            display_string += "|has mustache" if self.mustache else "|no mustache"
        if self.sunglasses_confidence >= self.face_feature_confidence_threshold:
            display_string += "|wears sunglasses" if self.sunglasses else "|no sunglasses"
        if self.eyeglasses_confidence >= self.face_feature_confidence_threshold:
            display_string += "|wears eyeglasses" if self.eyeglasses else "|no eyeglasses"
        if self.mouthopen_confidence >= self.face_feature_confidence_threshold:
            display_string += "|mouth is open" if self.mouthopen else "|mouth is closed"
        if self.eyesopen_confidence >= self.face_feature_confidence_threshold:
            display_string += "|eyes are open" if self.eyesopen else "|eyes is closed"
        display_string += f"|face is located {self.left_display} from left - {self.top_display} from top - with height {self.height_display} and width {self.width_display} of the video frame"
        
        return display_string

class VideoPreprocessor(ABC):
    def __init__(self, 
        transcription_job_name: str,
        bucket_name: str,
        video_s3_path: str,
        video_transcript_s3_path: str,
        frame_interval: str):

        self.s3_client = boto3.client("s3")
        self.transcribe_client = boto3.client("transcribe")
        self.rekognition_client = boto3.client("rekognition")
        self.transcription_job_name: str = transcription_job_name
        self.bucket_name: str = bucket_name
        self.video_s3_path: str = video_s3_path
        self.video_transcript_s3_path: str = video_transcript_s3_path
        self.video_duration_seconds: float  = 0.0
        self.video_duration_millis: int = 0
        self.visual_scenes: dict[int, str] = {}
        self.visual_texts: dict[int, list[str]] = {}
        self.transcript: dict
        self.celebrities: dict[int, list[str]] = {}
        self.faces: dict[int, list[str]] = {}
        self.person_timestamps_millis: list[int] = []
        self.frame_interval: int = int(frame_interval) # Millisecond
        self.frame_dim_for_vqa: tuple(int) = (512, 512)
    
    @abstractmethod
    def call_vqa(self, image_data: str) ->str:
        pass

    def wait_for_transcription_job(self):
        get_transcription = self.transcribe_client.get_transcription_job(TranscriptionJobName=self.transcription_job_name)
        while(get_transcription["TranscriptionJob"]["TranscriptionJobStatus"] == 'IN_PROGRESS'):
            time.sleep(5)
            get_transcription = self.transcribe_client.get_transcription_job(TranscriptionJobName=self.transcription_job_name)
    
    def fetch_transcription(self) -> dict:
        video_transcription_file: dict = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.video_transcript_s3_path)
        self.transcript = json.loads(video_transcription_file['Body'].read().decode('utf-8'))

    def download_video(self) -> str:
        filename: str = os.path.basename(self.video_s3_path)
        self.s3_client.download_file(self.bucket_name, self.video_s3_path, filename)
        return filename

    def load_video(self, video_filename: str) -> cv2.VideoCapture:
        return cv2.VideoCapture(video_filename)

    def extract_information_from_vqa(self):
        video: cv2.VideoCapture = self.load_video( self.download_video() )

        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)
        self.video_duration_seconds = float(frame_count/fps)
        self.video_duration_millis = int(self.video_duration_seconds*1000)
        
        timestamp_millis: int
        for timestamp_millis in range(0, self.video_duration_millis, self.frame_interval):
            video.set(cv2.CAP_PROP_POS_MSEC, int(timestamp_millis))
            success, frame = video.read()
            # Resize frame to 512 x 512 px
            dim = self.frame_dim_for_vqa
            # This may fail if the frame is empty. Just skip the frame if so.
            try:
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            except:
                continue
            
            if not success: continue

            image_pil = Image.fromarray(frame)
            io_stream = io.BytesIO()
            image_pil.save(io_stream, format='JPEG')
            image = io_stream.getvalue()

            vqa_response = self.call_vqa(base64.b64encode(image).decode("utf-8")) 

            # Sometimes the response might be censored due to false positive of inappropriate content. When that happens, just skip this frame.
            try:
                parsed_vqa_results = json.loads(vqa_response)
            except:
                continue

            self.visual_scenes[timestamp_millis] = parsed_vqa_results["scene"]
            self.visual_texts[timestamp_millis] = parsed_vqa_results["text"]

            if int(parsed_vqa_results["has_face"]) == 1:

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
                if len(unrecognized_faces) == 0: continue

                # Call Rekognition to detect faces
                face_findings: dict = self.rekognition_client.detect_faces(Image={'Bytes': image}, Attributes=['ALL'])['FaceDetails']

                if len(face_findings) == 0: continue

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
    
    def wait_for_dependencies(self):
        self.wait_for_transcription_job()

    def run(self):
        self.extract_information_from_vqa()
        self.fetch_transcription()
        return self.visual_scenes, self.visual_texts, self.transcript, self.celebrities, self.faces

class VideoPreprocessorBedrockVQA(VideoPreprocessor):
    def __init__(self, 
        transcription_job_name: str,
        bucket_name: str,
        video_s3_path: str,
        video_transcript_s3_path: str,
        frame_interval: str,
        vqa_model_name: str
        ):

        super().__init__(transcription_job_name=transcription_job_name, 
            bucket_name=bucket_name,
            video_s3_path=video_s3_path,
            video_transcript_s3_path=video_transcript_s3_path,
            frame_interval=frame_interval
        )

        self.bedrock_client = boto3.client("bedrock-runtime")
        self.vqa_model_name = vqa_model_name
        self.llm_parameters = {
            "anthropic_version": "bedrock-2023-05-31",    
            "max_tokens": 1000,
            "temperature": 0.1,
            "top_k": 3,
        }
    
    def call_vqa(self, image_data) -> str:
        system_prompt = "You are an expert in extracting information from video frames. Each video frame is an image. You will extract the scene, text, and more information."
        task_prompt =   "Extract information from this image and output a JSON with this format:\n" \
                        "{\n" \
                        "\"scene\" : \"String\",\n" \
                        "\"text\" : [\"String\", \"String\" , . . .],\n" \
                        "\"has_face\": \"Integer\"\n" \
                        "}\n" \
                        "For \"scene\", describe what you see in the picture in detail.\n" \
                        "For \"text\", list the text you see in that picture.\n" \
                        "For \"has_face\": \"1\" for True or \"0\" for False on whether you see face in the picture."
        
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
            except bedrock.exceptions.ThrottlingException as e:
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
        visual_scenes: dict[int, list[str]], 
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
        self.video_script_folder: str = video_script_folder
        self.video_name: str = video_name
        self.video_path: str = video_path
        self.original_visual_scenes: dict[int, str] = visual_scenes
        self.original_visual_texts: dict[int, list[str]] = visual_texts
        self.original_transcript: dict = transcript
        self.original_celebrities: dict[int, list[CelebrityFinding]] = celebrities
        self.original_faces: dict[int, list[FaceFinding]] = faces
        self.visual_scenes: list[list[Union[int, str]]] = []
        self.visual_texts: list[list[Union[int, list[str]]]] = []
        self.transcript: list[list[Union[int, str]]] = []
        self.celebrities:list[list[Union[int, list[str]]]]  = []
        self.faces: list[list[Union[int, list[str]]]]  = []
        self.video_script_chunk_size_for_summary_generation: int = 100000 # characters
        self.video_script_chunk_overlap_for_summary_generation: int = 500 # characters
        self.video_script_chunk_size_for_entities_extraction: int = 50000 #10000 # characters
        self.video_script_chunk_overlap_for_entities_extraction: int = 500 #200 # characters
        self.video_script_storage_chunk_size: int = 768 #10000 # Number of characters, which depends on the embedding model
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
  
    def preprocess_visual_scenes(self):
        self.visual_scenes =  sorted(self.original_visual_scenes.items())
      
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
        transcript= {}
        word_start_time = None
        sentence_start_time = -1
        sentence = ""
        for item in self.original_transcript["results"]["items"]:
            # If this is a punctuation, which could be end of sentence
            if item['type'] != 'punctuation':
                word_start_time = int(float(item['start_time'])*1000) # In millisecond
                # If this is start of sentence
                if sentence_start_time  == -1:
                    # If there is a speaker label, then start the sentence by identifying the speaker id.
                    if "speaker_label" in item:
                        label = item['speaker_label'].replace('spk','speaker')
                        sentence += f" <{label}>"
                    # Add word to sentence with heading space
                    sentence += f" { item['alternatives'][0]['content'] }"
                    # Set the start time of the sentence to be the start time of this first word in the sentence
                    sentence_start_time  = word_start_time
                # If this is mid of sentence
                else:
                    # Add word to sentence with heading space
                    sentence += f" { item['alternatives'][0]['content'] }"

                self.transcript = sorted(transcript.items())
            else:
                # Add punctuation to sentence without heading space
                sentence += f"{ item['alternatives'][0]['content'] }"
                # Add sentence to transcription
                transcript[word_start_time] = sentence
                # Reset the sentence and sentence start time
                sentence = ""
                sentence_start_time  = -1
    
    def preprocess_celebrities(self):
        self.celebrities = sorted(self.original_celebrities.items())

    def preprocess_faces(self):
        self.faces = sorted(self.original_faces.items())
      
    def generate_combined_video_script(self):
        def transform_scenes(x):
            timestamp = x[0]
            objects = ",".join(x[1])
            return (timestamp, f"Scene:{objects}")
        scenes  = list(map(transform_scenes, self.visual_scenes))

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
            celebrities = ",".join([celeb_finding.display() for celeb_finding in x[1]])
            return (timestamp, f"Celebrity:{celebrities}")
        visual_celebrities = list(map(transform_celebrities, self.celebrities))

        def transform_faces(x):
            timestamp = x[0]
            faces = ",".join([face_finding.display() for face_finding in x[1]])
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
                        "Visual scenes (scene) represents what objects are visible in the video at that millisecond. This can be the objects seen in camera, or objects from a screen sharing, or any other visual scenarios. You can infer how the visualization look like, so long as you are confident.\n" \
                        "Visual texts (text) are the text visible in the video. It can be texts in real world objects as recorded in video camera, or those from screen sharing, or those from presentation recording, or those from news, movies, or others. \n" \
                        "Human voice (voice) is the transcription of the video.\n" \
                        "Celebrities (celebrity) provides information about the celebrity detected in the video at that millisecond. It may have the information on whether the celebrity is smiling and the captured emotions. It may also has information on where the face is located relative to the video frame size. The celebrity may not be speaking as he/she may just be portrayed. \n" \
                        "Human faces (face) lists the face seen in the video at that millisecond. This may have information on the emotions, face location relative to the video frame, and more facial features. \n"

        video_script_length = len(self.combined_video_script)

        # When the video is short enough to fit into 1 chunk
        if video_script_length <= self.video_script_chunk_size_for_summary_generation:
            core_prompt = f"The VIDEO TIMELINE has format below.\n" \
                            "timestamp in milliseconds:scene / text / voice\n" \
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
                        "timestamp in milliseconds:scene / text / voice\n" \
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
                        "Visual scenes (scene) represents what objects are visible in the video at that millisecond. This can be the objects seen in camera, or objects from a screen sharing, or any other visual scenarios. You can infer how the visualization look like, so long as you are confident.\n" \
                        "Visual texts (text) are the text visible in the video. It can be texts in real world objects as recorded in video camera, or those from screen sharing, or those from presentation recording, or those from news, movies, or others. \n" \
                        "Human voice (voice) is the transcription of the video.\n" \
                        "Celebrities (celebrity) provides information about the celebrity detected in the video at that millisecond. It may have the information on whether the celebrity is smiling and the captured emotions. It may also has information on where the face is located relative to the video frame size. The celebrity may not be speaking as he/she may just be portrayed. \n" \
                        "Human faces (face) lists the face seen in the video at that millisecond. This may have information on the emotions, face location relative to the video frame, and more facial features. \n"
                        
        
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
        number_of_chunks = math.ceil( (video_script_length + 1) / self.video_script_storage_chunk_size )

        chunks: list[self.Contents] = []
        for chunk_number in range(0, number_of_chunks):
            is_last_chunk = (chunk_number == (number_of_chunks - 1))
            is_first_chunk = (chunk_number == 0)

            start = 0 if is_first_chunk else int(chunk_number*self.video_script_storage_chunk_size)
            stop = video_script_length if is_last_chunk else (chunk_number+1)*self.video_script_storage_chunk_size
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

      
    def run(self):
        self.preprocess_visual_scenes()
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
        visual_scenes: dict[int, list[str]], 
        visual_texts: dict[int, list[str]], 
        transcript: dict,
        celebrities: dict[int, list[str]],
        faces: dict[int, list[str]],
        summary_folder: str,
        entity_sentiment_folder: str,
        video_script_folder: str
        ):
        super().__init__(bucket_name, video_name, video_path, visual_scenes, visual_texts, transcript, celebrities, faces,summary_folder, entity_sentiment_folder, video_script_folder)
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

def handler():
    video_name = os.path.basename(video_s3_path)
    video_path= '/'.join(video_s3_path.split('/')[1:])
    video_transcript_s3_path = f"{transcription_folder}/{video_path}.txt"

    try:
        # Initiate class for video preprocessing
        video_preprocessor = VideoPreprocessorBedrockVQA( 
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
        visual_scenes, visual_texts, transcript, celebrities, faces = video_preprocessor.run()
        
        # Initiate class for video analysis
        video_analyzer = VideoAnalyzerBedrock(
            model_name=model_id, 
            embedding_model_name=embedding_model_id,
            bucket_name=bucket_name,
            video_name=video_name,
            video_path=video_path,
            visual_scenes=visual_scenes,
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

if __name__ == "__main__":
    handler()
import os, time, json, copy, math
from abc import ABC, abstractmethod
import boto3

rekognition = boto3.client('rekognition')
transcribe = boto3.client("transcribe")
sagemaker = boto3.client("sagemaker-runtime")
bedrock = boto3.client("bedrock-runtime")  
s3 = boto3.client("s3")

model_id = os.environ["MODEL_ID"]
bucket_name = os.environ["BUCKET_NAME"]
raw_folder = os.environ["RAW_FOLDER"]
video_script_folder = os.environ["VIDEO_SCRIPT_FOLDER"]
transcription_folder = os.environ["TRANSCRIPTION_FOLDER"]
entity_sentiment_folder = os.environ["ENTITY_SENTIMENT_FOLDER"]
summary_folder = os.environ["SUMMARY_FOLDER"]

def handler(event, context):
    print("received event:")
    print(event)

    video_s3_path = ""
    labels_job_id = ""
    texts_job_id=""
    transcription_job_name=""

    for payload in event:
        video_s3_path = payload["videoS3Path"]
        if 'labelDetectionResult' in payload: labels_job_id = payload['labelDetectionResult']['JobId'] 
        if 'textDetectionResult' in payload: texts_job_id = payload['textDetectionResult']['JobId']
        if 'transcriptionResult' in payload: transcription_job_name = payload['transcriptionResult']["TranscriptionJob"]["TranscriptionJobName"]

    video_name = os.path.basename(video_s3_path)
    video_transcript_s3_path = f"{transcription_folder}/{video_name}.txt"

    try:
        wait_for_rekognition_label_detection(labels_job_id=labels_job_id, sort_by='TIMESTAMP')
        wait_for_rekognition_text_detection(texts_job_id=texts_job_id)
        wait_for_transcription_job(transcription_job_name=transcription_job_name)

        response = analyze_video(labels_job_id=labels_job_id, texts_job_id=texts_job_id, video_name=video_name, video_transcript_s3_path=video_transcript_s3_path)

        store_summary_result(summary=response['summary'], video_name=video_name)
        store_video_script_result(video_script=response['video_script'], video_name=video_name)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps(response)
    }

def store_summary_result(summary, video_name):
    s3.put_object(
        Body=summary, 
        Bucket=bucket_name, 
        Key=f"{summary_folder}/{video_name}.txt"
    )
    
def store_video_script_result(video_script, video_name):
    s3.put_object(
        Body=video_script, 
        Bucket=bucket_name, 
        Key=f"{video_script_folder}/{video_name}.txt"
    )

def analyze_video(labels_job_id, texts_job_id, video_name, video_transcript_s3_path):
    video_scenes, video_length = visual_scenes_iterate_pages({}, labels_job_id)
    video_texts = visual_texts_iterate_pages({}, texts_job_id)
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

def visual_scenes_iterate_pages(scenes, labelsJobId, next_token=None):
    if next_token:
        getObjectDetection = rekognition.get_label_detection(
            JobId=labelsJobId,
            SortBy='TIMESTAMP',
            NextToken=next_token
        )
    else:
        getObjectDetection = rekognition.get_label_detection(
            JobId=labelsJobId,
            SortBy='TIMESTAMP'
        )

    video_duration = float(getObjectDetection["VideoMetadata"]["DurationMillis"])/1000
    
    for l in getObjectDetection['Labels']:
        scene = []
        timestamp = int(l['Timestamp']/1000)
        if timestamp in scenes:
            scene = scenes[timestamp]
        else:
            scenes[timestamp] =scene
        detected_labels = l['Label']['Name']
        if detected_labels not in scene:
            scene.append(detected_labels)
    
    if "NextToken" in getObjectDetection:
        visual_scenes_iterate_pages(scenes, labelsJobId, getObjectDetection["NextToken"])
    
    return scenes, video_duration
    
def visual_texts_iterate_pages(texts, texts_job_id, next_token=None):
    if next_token:
        getTextDetection = rekognition.get_text_detection(
            JobId=texts_job_id,
            NextToken=next_token
        )
    else:
        getTextDetection = rekognition.get_text_detection(
            JobId=texts_job_id
        )

    for l in getTextDetection['TextDetections']:
        if l['TextDetection']["Type"] == "WORD": continue
        
        text = []
        timestamp = int(l['Timestamp']/1000)
        if timestamp in texts:
            text = texts[timestamp]
        else:
            texts[timestamp] =text

        detected_texts = l['TextDetection']['DetectedText']
        text.append(detected_texts)
    
    if "NextToken" in getTextDetection:
        visual_texts_iterate_pages(texts, texts_job_id, getTextDetection["NextToken"])
    
    return texts
    
class VideoAnalyzer(ABC):
  def __init__(self, video_length, video_scenes, video_texts, transcript):
      self.original_scenes = video_scenes
      self.original_visual_texts = video_texts
      self.original_transcript = transcript
      self.scenes = []
      self.visual_texts = []
      self.transcript = []
      #self.visual_objects_mapping = {}
      #self.visual_objects_mapping_string = ""
      self.video_length = int(video_length)
      self.video_chunk_length = 10*100 # seconds
      self.video_chunk_overlap = 60 # seconds
      self.current_time_slice = (0, self.video_chunk_length if int(video_length) > self.video_chunk_length else int(video_length))
      self.scene_similarity_score = 0.5
      self.video_rolling_summary = ""
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
          # Delete scene entry if it is outside of the requested time window
          elif float(k) < float(self.current_time_slice[0]) or float(k) >= float(self.current_time_slice[1]): 
              del scenes[k]
          # Delete scene entry if the detected objects are the same as the previous scene
          elif v == prev_objects:
              del scenes[k]
          # Delete scene entry if the detected objects are too similar with the previous scene
          elif float(num_of_matches_with_prev_objects) > len(prev_objects)*self.scene_similarity_score:
              del scenes[k]
          else:
              prev_objects = v
      
      # The code below maps the object names into symbols and create a mapping from the symbols to the actual words    
      """
      objects_mapping = {}
      for t, detected_objects in scenes.items():
          for i, detected_object in enumerate(detected_objects):
              if detected_object not in objects_mapping:
                  numeric_code = len(objects_mapping.keys())
                  objects_mapping[detected_object] = chr(numeric_code+1200)
                  detected_objects[i] = chr(numeric_code+1200)
              else:
                  detected_objects[i]  = objects_mapping[detected_object]
      objects_mapping = {v: k for k, v in objects_mapping.items()}
      
      self.scenes = sorted(scenes.items())
      self.visual_objects_mapping = objects_mapping
      """
      self.scenes = sorted(scenes.items())
      
  def preprocess_visual_texts(self):
      visual_texts = dict(sorted(self.original_visual_texts.items()))
      prev_texts = []
      for k,v in list(visual_texts.items()):
          #  Delete text entry if there is not detected text
          if v == []: 
              del visual_texts[k]
          # Delete text entry if it is outside of the requested time window
          elif float(k) < float(self.current_time_slice[0]) or float(k) >= float(self.current_time_slice[1]): 
              del visual_texts[k]
          # Delete text entry if the detected texts are the same as the previous text
          elif v == prev_texts:
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
              # If this punctuation is placed after a word whose timestamp is outside of the requested range, then ignore.
              if word_start_time < self.current_time_slice[0] or word_start_time > self.current_time_slice[1]:
                  continue
              # Add punctuation to sentence without heading space
              sentence += f"{ item['alternatives'][0]['content'] }"
              # Add sentence to transcription
              transcript[word_start_time] = sentence
              # Reset the sentence and sentence start time
              sentence = ""
              sentence_start_time  = -1
          else:
              word_start_time = int(float(item['start_time']))
              # If this word comes after the requested time window
              if word_start_time >= self.current_time_slice[1]:
                  # If this is not the first word of a sentence
                  if  sentence_start_time  >= 0:
                      # Then insert the partial sentence in o the transcription
                      transcript[word_start_time] = sentence
                  # And stop
                  break
              # If this word comes before the requested time window
              elif word_start_time < self.current_time_slice[0]:
                  # Then ignore the word and ensure sentence start time is reset
                  sentence_start_time  = -1
                  continue
              elif sentence_start_time  == -1:
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

      #self.visual_objects_mapping_string = ",".join(list(map(lambda x: f"{x[0]}:{x[1]}",list(self.visual_objects_mapping.items()))))
      
      # Combine all inputs
      combined_video_script = sorted( scenes + visual_texts + transcript)
      
      combined_video_script = "\n".join(list(map(lambda x: f"{x[0]}:{x[1]}", combined_video_script)))
      
      self.combined_video_script = combined_video_script
      self.all_combined_video_script += combined_video_script
  
  def analyze(self):
      prompt_prefix = "You are an expert video analyst who reads a VIDEO SCRIPT and creates summary of the video and why it is interesting.\n" \
                       "The VIDEO SCRIPT contains the visual scenes, the visual texts, and human voice in the video."
      
      # When the video is short enough to fit into 1 chunk
      if self.video_length <= self.video_chunk_length:
          self.preprocess_scenes()
          self.preprocess_visual_texts()
          self.preprocess_transcript()
          self.generate_combined_video_script()
          
          """
          core_prompt = f"The VIDEO SCRIPT has format below.\n" \
                      "timestamp in seconds:scene / text / voice\n" \
                      "Use LEGEND to interpret each of the scene objects represented by symbols.\n\n" \
                      "=VIDEO SCRIPT BEGINS=\n" \
                      f"{self.combined_video_script}\n" \
                      "=VIDEO SCRIPT ENDS=\n" \
                      "=LEGEND BEGINS=\n" \
                      f"{self.visual_objects_mapping_string}\n" \
                      "=LEGEND ENDS="
          """
          
          core_prompt = f"The VIDEO SCRIPT has format below.\n" \
                      "timestamp in seconds:scene / text / voice\n" \
                      "=VIDEO SCRIPT BEGINS=\n" \
                      f"{self.combined_video_script}\n" \
                      "=VIDEO SCRIPT ENDS=\n" \
          
          prompt = f"{prompt_prefix}\n\n" \
          f"{core_prompt}\n\n" \
          "Given the VIDEO SCRIPT above, decribe the summary of the video and why it is interesting. DO NOT make up anything you do not know.\n" \
          "Summary: "
          
          self.video_rolling_summary = self.call_llm(prompt)
      # When the video is long enough to be divided into multiple chunks to fit within LLM's context length
      else:
          number_of_chunks = math.ceil( (self.video_length + 1) / (self.video_chunk_length - self.video_chunk_overlap) )
          
          for chunk_number in range(0, number_of_chunks):
              is_first_chunk = False
              if chunk_number == 0:
                  is_first_chunk = True
                  start = 0
              else:
                  start = int(0 + chunk_number*(self.video_chunk_length - self.video_chunk_overlap))
              stop = int(self.video_length) if self.video_length < start + self.video_chunk_length else int(start + self.video_chunk_length)
              
              is_last_chunk = True if chunk_number == (number_of_chunks - 1) else False
              
              self.current_time_slice = (start, stop)
              
              self.preprocess_scenes()
              self.preprocess_visual_texts()
              self.preprocess_transcript()
              self.generate_combined_video_script()
              
              """
              core_prompt = f"The VIDEO SCRIPT has format below.\n" \
                      "timestamp in seconds:scene / text / voice\n" \
                      "Use LEGEND to interpret each of the scene objects represented by symbols.\n\n" \
                      "=VIDEO SCRIPT BEGINS=\n" \
                      f"{self.combined_video_script}\n" \
                      "=VIDEO SCRIPT ENDS=\n" \
                      "=LEGEND BEGINS=\n" \
                      f"{self.visual_objects_mapping_string}\n" \
                      "=LEGEND ENDS="
              """
              
              core_prompt = f"The VIDEO SCRIPT has format below.\n" \
                      "timestamp in seconds:scene / text / voice\n" \
                      "=VIDEO SCRIPT BEGINS=\n" \
                      f"{self.combined_video_script}\n" \
                      "=VIDEO SCRIPT ENDS=\n" \
              
              if is_last_chunk:
                  prompt = f"{prompt_prefix}\n\n" \
                  f"The video has duration of {self.video_length} seconds.\n\n" \
                  f"Below is the summary of the video {start} seconds into the video:\n\n" \
                  f"{self.video_rolling_summary}\n\n" \
                  f"The below VIDEO SCRIPT is only for the LAST video chunk between {start} to {stop} seconds.\n\n" \
                  f"{core_prompt}\n\n" \
                  "Given the previous summary and the VIDEO SCRIPT above, decribe the summary of the whole video and why it is interesting. DO NOT make up anything you do not know.\n" \
                  "Summary: "
                  
                  chunk_summary = self.call_llm(prompt)
                  self.video_rolling_summary = chunk_summary
                  
              elif is_first_chunk:
                  prompt = f"{prompt_prefix}\n\n" \
                  f"The video has duration of {self.video_length} seconds. The below VIDEO SCRIPT is only for a chunk of the video, from {start} to {stop} seconds.\n\n" \
                  f"{core_prompt}\n\n" \
                  f"Given VIDEO SCRIPT above, decribe the summary of the video so far. DO NOT make up anything you do not know.\n" \
                  "Summary: "
                  
                  chunk_summary = self.call_llm(prompt)
                  self.video_rolling_summary = chunk_summary
              else:
                  prompt = f"{prompt_prefix}\n\n" \
                  f"The video has duration of {self.video_length} seconds.\n\n" \
                  f"Below is the summary of the video {start} seconds into the video:\n\n" \
                  f"{self.video_rolling_summary}\n\n" \
                  f"The below VIDEO SCRIPT is only for video chunk between {start} to {stop} seconds.\n\n" \
                  f"{core_prompt}\n\n" \
                  "Given the previous summary and the VIDEO SCRIPT above, decribe the summary of the whole video so far. DO NOT make up anything you do not know.\n" \
                  "Summary: "
                  
                  chunk_summary = self.call_llm(prompt)
                  self.video_rolling_summary = chunk_summary

      return self.video_rolling_summary
      
  def run(self):
      summary = self.analyze()
      video_script = self.all_combined_video_script
      
      return {
          'summary': summary,
          'video_script': video_script
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
  
  def call_llm(self, prompt):
      print("Prompt:")
      print(f"\033[92m{prompt} \033[00m")
      
      self.llm_parameters['prompt'] = f"\n\nHuman:{prompt}\n\nAssistant:"
      input_str = json.dumps(self.llm_parameters)
      encoded_input = input_str.encode("utf-8")

      bedrock_response = bedrock.invoke_model(body=encoded_input, modelId=self.endpoint_name)
      response = json.loads(bedrock_response.get("body").read())["completion"]
      print("LLM response:")
      print(response)
      return response
      
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
      print("Prompt:")
      print(f"\033[92m{prompt} \033[00m")
      input_str = json.dumps({"inputs": prompt, "parameters": self.llm_parameters})
      encoded_input = input_str.encode("utf-8")

      response = self.query_endpoint_with_json_payload(encoded_input, endpoint_name, content_type="application/json")
      response = self.parse_response_model(response)[0]
      print("LLM response:")
      print(response)
      return response
import json
from datetime import timedelta
import re

def read_and_process_scenes(file_path):
    segments = []
    current_segment = None
    current_scene = None
    current_timestamp = None

    with open(file_path, 'r') as file:
        for line in file:
            if re.match(r'^\d+\.\d+:Scene:', line.strip()):
                # New scene starts
                if current_scene:
                    process_scene(current_timestamp, current_scene, segments, current_segment)
                    current_scene = None
                    current_segment = segments[-1] if segments else None

                parts = line.strip().split(':', 2)
                if len(parts) == 3 and parts[1] == 'Scene':
                    current_timestamp, _, content = parts
                    current_scene = content
                else:
                    current_timestamp = None
                    current_scene = None
            elif current_scene is not None:
                # Continuation of the current scene
                current_scene += line

    # Process the last scene
    if current_scene:
        process_scene(current_timestamp, current_scene, segments, current_segment)

    return segments

def process_scene(timestamp, scene, segments, current_segment):
    # Find the JSON part in the scene text
    json_start = scene.find('{')
    json_end = scene.rfind('}') + 1
    data = None
    
    # Scan the scene and replace "False" with "false", and "True" with "true"
    scene = scene.replace('False', 'false')
    scene = scene.replace('True', 'true')
    scene = scene.replace('None', 'null')
    scene = scene.replace('none', 'null')
    
    if json_start != -1 and json_end != -1:
        json_part = scene[json_start:json_end]
    else:
        # Create a non-event JSON body if no valid JSON is found
        json_part = '{"key_event": "no_event", "description": "No significant event detected"}'

    try:
        data = json.loads(json_part)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Problematic JSON string: {json_part}")
        data = None
        
    key_event = data.get('key_event')
    
    if key_event and timestamp:
        if current_segment and current_segment['event'] == key_event:
            current_segment['end_time'] = float(timestamp) + 5.0
        else:           
            new_segment = {
                'start_time': float(timestamp),
                'end_time': float(timestamp) + 5.0,
                'event': key_event,
                'score': data.get('highlight_score'),
            }
            segments.append(new_segment)

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))[0:7]

def save_merged_segments(segments, output_file):
    with open(output_file, 'w') as file:
        for segment in segments:
            if segment['event'] == 'no_event' or segment['end_time'] - segment['start_time'] <= 10.0:
                continue
            start = format_time(segment['start_time'])
            end = format_time(segment['end_time'])
            file.write(f"{start} - {end}: {segment['event']} {segment['score']}\n")

# Usage
input_file = r'C:\Users\fpengzha\Downloads\Full_Spain_vs_Italy___Semi_Final_UEFA_Nations_League_22_23__1_.mp4 (5).txt'
output_file = r'C:\Users\fpengzha\Downloads\segments.txt'

segments = read_and_process_scenes(input_file)
save_merged_segments(segments, output_file)

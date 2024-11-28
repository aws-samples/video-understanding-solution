import json
from datetime import timedelta
import re

def read_and_process_scenes(file_path):
    segments = []
    current_segment = None
    current_scene = None
    current_timestamp = None

    with open(file_path, 'r', encoding='utf-8') as file:
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
    # Scan the scene and replace "False" with "false", and "True" with "true"
    scene = scene.replace('False', 'false')
    scene = scene.replace('True', 'true')
    scene = scene.replace('None', 'null')
    scene = scene.replace(': none', ': null')
    scene = scene.replace(': int', ': null')
    scene = scene.replace(': string', ': null')
    scene = scene.replace(': boolean', ': null')
    scene = scene.replace('\\"', '"')
    scene = scene.replace('\\n', '')
    
    # Find the JSON part in the scene text
    json_start = scene.find('{')
    json_end = scene.find('}', json_start) + 1
    data = None
    
    if json_start != -1 and json_end != -1:
        json_part = scene[json_start:json_end]
    else:
        # Create a non-event JSON body if no valid JSON is found
        json_part = '{"key_event": "none", "description": "No significant event detected"}'

    try:
        data = json.loads(json_part)
    except json.JSONDecodeError as e:
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
        previous_end_time = 0
        new_segments = []
        for segment in segments:
            if segment['event'] == 'none' or segment['event'] == 'injury' or segment['event'] == 'foul' or segment['event'] == 'offside' or segment['event'] == 'free_kick' or segment['event'] == 'shot_off_goal_target' or segment['event'] == 'corner_kick':
                continue
            start = format_time(max(previous_end_time, segment['start_time']))
            end = format_time(segment['end_time'])
            
            if start >= end:
                continue
            
            # Extend start_time and end_time by 5 seconds for goal events
            if segment['event'] == 'goal' or segment['event'] == 'shot_on_goal_target':
                segment['start_time'] = max(previous_end_time, segment['start_time'] - 5)
                segment['end_time'] += 5
                
            previous_end_time = segment['end_time']
            
            start = format_time(segment['start_time'])
            end = format_time(segment['end_time'])
            
            new_segment = {
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'event': segment['event']
            }
            new_segments.append(new_segment)
            
        # Merge consecutive segments
        merged_segments = []
        for segment in new_segments:
            if not merged_segments or segment['start_time'] > merged_segments[-1]['end_time']:
                merged_segments.append(segment)
            else:
                # Extend the previous segment
                merged_segments[-1]['end_time'] = max(merged_segments[-1]['end_time'], segment['end_time'])
        
        # Update new_segments with the merged segments
        new_segments = merged_segments
        
        # Write out new_segments to the file
        for segment in new_segments:
            start = format_time(segment['start_time'])
            end = format_time(segment['end_time'])
            file.write(f"{start} - {end}: {segment['event']}\n")
            

# Usage
input_file = r'C:\Users\fpengzha\Downloads\Full_Spain_vs_Italy___Semi_Final_UEFA_Nations_League_22_23__1_.mp4 (8).txt'
output_file = r'C:\Users\fpengzha\Downloads\segments.txt'

segments = read_and_process_scenes(input_file)
save_merged_segments(segments, output_file)

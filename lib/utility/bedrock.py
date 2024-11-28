import json
import boto3
import os
import re
from botocore.exceptions import ClientError

session = boto3.Session()
bedrock = session.client(service_name='bedrock-runtime') #creates a Bedrock client

bedrock_model_id = "anthropic.claude-3-haiku-20240307-v1:0" 
# bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0" 

system_prompt = """
You are an expert in extracting key events from soccer game video frames. You will be given sequences of video frames from sports broadcasts. Your task is to identify specific key events using strict criteria.

Set the key_event field to one of the following values: "goal", "corner kick", "free kick", "foul", "offside", "injury", "shot on target", "shot off target", "none".

Each event has it's own JSON structure as followed. Try to capture the information from the video frames and fill in the JSON structure accordingly. Game event interval captured for the given frames should be put in the "event_interval" field.

goal => 
{
   "key_event" : "goal",
   "player_nbr" : 7,
   "jersey_color" : "red",
   "is_penalty_kick" : False,
   "event_interval" : string,
   "game_clock" : "02:33",
   "team_name": "FC Bayern",
   "replay": False,
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

shot on target =>
{
   "key_event" : "shot_on_goal_target",
   "player_nbr" : int,
   "player_jersey_color" : "red",
   "event_interval" : string,
   "game_clock" : "62:00",
   "replay": False,
   "team_name": "FC Bayern",
   "key_event_prediction_confident_score" : int
}

shot off target =>
{
   "key_event" : "shot_off_goal_target",
   "player_nbr" : int,
   "player_jersey_color" : "red",
   "event_interval" : string,
   "game_clock" : "70:33",
   "replay": False,
   "team_name": "FC Bayern",
   "key_event_prediction_confident_score" : int
}

no key event =>
{
   "key_event" : "none"
   "event_interval" : string
}

Capture the game clock ONLY if it's visible in the video frames. Game clock is located on the upper left corner of a video frame. DO NOT use any other means to capture the Game clock. If you cannot determine the Game clock, set its value as "none".

Here are a comprehensive and strict guidelines for identify each key event:

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

Think step by step. Put all your thoughts and analysis into <thinking> tags including the confident scores. Use the analysis and thoughts to help determine the key event. You have to be very confident about the event to suggest it. Think very hard. If your confident score is lower than 90, analyze the video frames again until you have good confidence about the event.
You should analyze 3 image frames at once.  Describe where the ball is, what the goalkeeper is doing. If the referee is visible in any images, describe what he is doing.

Only return the key events in JSON format defined above. 
There should only be 1 key event for the given video frames. Do not provide any other further explanations.
        
"""

prompt = "Analyze the given sequence of video frames. Tell me what you found!" 

def analyze_frames(folder):
    frame_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    frame_files.sort()  # Ensure files are in order

    image_bytes_list = []
    for frame_file in frame_files:
        frame_path = os.path.join(folder, frame_file)
        with open(frame_path, "rb") as image_file:
            image_bytes = image_file.read()
            image_bytes_list.append(image_bytes)

    message_list = []

    image_message = {
        "role": "user",
        "content": [
            { "text": prompt },
            *[
                {
                    "image": {
                        "format": "jpeg",
                        "source": {
                            "bytes": image_bytes
                        }
                    }
                }
                for image_bytes in image_bytes_list
            ]
        ],
    }

    message_list.append(image_message)

    response = bedrock.converse(
        modelId=bedrock_model_id,
        messages=message_list,
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0
        },
        system=[
            { "text": system_prompt }
        ],
    )

    response_message = json.dumps(response['output']['message']['content'][0]['text'], indent=4)
    # Extract the key_event value from the response without using regex

    content = response_message
    key_event_start = content.find(r'\"key_event\"')
    if key_event_start != -1:
        colon_pos = content.find(':', key_event_start)
        if colon_pos != -1:
            start_quote = content.find('"', colon_pos + 1)
            if start_quote != -1:
                end_quote = content.find('"', start_quote + 1)
                if end_quote != -1:
                    key_event = content[start_quote + 1:end_quote-1]
                    print(f"\n*** {key_event} ***\n")
                
    # Extract the last digits in the folder name and calculate start timestamp
    last_digits = re.search(r'\d+$', folder)
    formatted_message = None
    if last_digits:
        start_timestamp = float(last_digits.group())
        start_timestamp = f"{start_timestamp:.1f}"
        
        # Format the message
        formatted_message = f"{start_timestamp}:Scene:{response_message}\n"
        print(formatted_message)
    
    return formatted_message

ACTUAL_GAME_START_FRAME = 200          
# Create a list of folders from batch5 to batch40, incrementing by 5
folders = [f"batch{i}" for i in range(ACTUAL_GAME_START_FRAME, 236, 5)]

# Update the folder path to use the list
base_path = r"C:\Users\fpengzha\Downloads\clipped_segments"
scenes = []
for folder_name in folders:
    folder_path = os.path.join(base_path, folder_name)
    print(f"Analyzing folder: {folder_path}")
    scenes.append(analyze_frames(folder_path))
    
# Write the message to Scenes.txt
with open(r"C:\Users\fpengzha\Downloads\Scenes.txt", "w") as f:
    for scene in scenes:
        if scene is not None:
            f.write(scene)
    
    



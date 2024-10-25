import cv2
from PIL import Image
import io
import os

def load_video(filename) -> cv2.VideoCapture:
        video: cv2.VideoCapture = cv2.VideoCapture(filename)
        return video

def _extract_frame(video_filename, timestamp_millis):
    video = load_video(video_filename)
    video.set(cv2.CAP_PROP_POS_MSEC, int(timestamp_millis))
    success, frame = video.read()
    if success:
        # Resize frame to 512 x 512 px
        dim = (512, 512)
        # This may fail if the frame is empty. Just skip the frame if so.
        try:
            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        except:
            return None

        image_pil = Image.fromarray(frame)
        # Save image to local file
        timestamp_str = str(timestamp_millis).zfill(10)  # Pad with zeros for consistent naming
        folder_path = r"C:\Users\fpengzha\Downloads\clipped_segments"  # Add a folder path
        os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
        local_filename = os.path.join(folder_path, f"frame_{timestamp_str}.jpg")
        image_pil.save(local_filename, format='JPEG')
        print(f"Saved frame to {local_filename}")
        
        io_stream = io.BytesIO()
        image_pil.save(io_stream, format='JPEG')
        image = io_stream.getvalue()
        return [timestamp_millis, image]
    

filename: str = r"C:\Users\fpengzha\Downloads\Full Spain vs Italy _ Semi Final UEFA Nations League 22_23 (1).mp4"
video: cv2.VideoCapture = cv2.VideoCapture(filename)
frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
fps = video.get(cv2.CAP_PROP_FPS)
video_duration_seconds = float(frame_count/fps)
video_duration_millis = int(video_duration_seconds*1000)
regular_timestamps_millis = range(0, video_duration_millis, 500)

for timestamp in regular_timestamps_millis:
    _ = _extract_frame(video_filename=filename, timestamp_millis=timestamp)
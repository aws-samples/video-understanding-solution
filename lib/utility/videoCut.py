import ffmpeg
import os
import tempfile
import subprocess

def clip_video(input_file, output_file, start_time, duration):
    """
    Clip a short video from a longer MP4 video.

    Args:
    input_file (str): Path to the input video file.
    output_file (str): Path to save the output video file.
    start_time (str): Start time of the clip in format "HH:MM:SS" or "MM:SS" or seconds.
    duration (str): Duration of the clip in format "HH:MM:SS" or "MM:SS" or seconds.

    Returns:
    bool: True if successful, False otherwise.
    """
    try:
        # Open the input file
        stream = ffmpeg.input(input_file, ss=start_time, t=duration)

        # Output the video without re-encoding
        stream = ffmpeg.output(stream, output_file, c='copy')

        # Run the FFmpeg command
        ffmpeg.run(stream, overwrite_output=True)

        print(f"Video clipped successfully: {output_file}")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def read_segments(segments_file):
    """
    Read segments from a file and return a list of tuples (start_time, end_time, event).
    """
    segments = []
    with open(segments_file, 'r') as file:
        for line in file:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                times, event = parts
                start, end = times.split(' - ')
                
                event_parts = event.split(' ', 1)
                event_type = event_parts[0]
                event_score = event_parts[1] if len(event_parts) > 1 else ""
                
                segments.append((start, end, event_type, event_score))
    return segments

def process_segments(input_file, segments_file, output_folder):
    """
    Process segments from a file and clip the video for each segment.
    """
    segments = read_segments(segments_file)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    clipped_segments = []
    
    for i, (start, end, event, score) in enumerate(segments):
        output_file = os.path.join(output_folder, f"clip_{i+1}_{event.replace(' ', '_')}.mp4")
        
        # Calculate duration
        start_parts = start.split(':')
        end_parts = end.split(':')
        start_seconds = int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + int(start_parts[2])
        end_seconds = int(end_parts[0]) * 3600 + int(end_parts[1]) * 60 + int(end_parts[2])
        duration = end_seconds - start_seconds
        
        clip_success = False;
        # Clip the video segment
        if (int(score) >= 0 and duration <= 15):
            clip_success = clip_video(input_file, output_file, start, str(duration))
        
        if clip_success:
            # Add the clipped segment to a list for later stitching
            clipped_segments.append(output_file)

    # After all segments are clipped, stitch them together
    if clipped_segments:
        final_output = os.path.join(output_folder, "final_stitched_video.mp4")
        stitch_videos(clipped_segments, final_output)
        print(f"Final stitched video created: {final_output}")
    else:
        print("No segments were successfully clipped. Cannot create final video.")
        
def stitch_videos(video_files, output_file):
    """
    Stitch multiple video files into a single video file.
    
    Args:
    video_files (list): List of paths to input video files.
    output_file (str): Path to the output stitched video file.
    """
    if not video_files:
        print("No video files to stitch.")
        return

    # Create a temporary file to store the list of input videos
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        for video in video_files:
            temp_file.write(f"file '{video}'\n")
        temp_file_path = temp_file.name

    try:
        # Use FFmpeg to concatenate the videos
        ffmpeg_command = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_file_path,
            '-c', 'copy',
            output_file
        ]
        
        subprocess.run(ffmpeg_command, check=True)
        print(f"Successfully stitched videos into: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while stitching videos: {e}")
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


# Example usage:
input_file = r"C:\Users\fpengzha\Downloads\Full Spain vs Italy _ Semi Final UEFA Nations League 22_23 (1).mp4"
segments_file = r"C:\Users\fpengzha\Downloads\segments.txt"
output_folder = r"C:\Users\fpengzha\Downloads\segments"

process_segments(input_file, segments_file, output_folder)


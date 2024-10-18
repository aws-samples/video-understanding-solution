import ffmpeg

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

# Example usage:
input_file = r"C:\Users\fpengzha\Downloads\Full Spain vs Italy _ Semi Final UEFA Nations League 22_23 (1).mp4"
output_file = r"C:\Users\fpengzha\\Downloads\Spain_vs_Italy_short.mp4"

clip_video(input_file, output_file, "00:01:30", "00:00:30")


from moviepy.editor import VideoFileClip
import os 
import subprocess
def reduce_video_quality(input_path, output_path):
    cmd = f'D:\HandBrakeCLI\HandBrakeCLI.exe --input {input_path} --output {output_path} --encoder x264 --quality 28'
    subprocess.run(cmd, shell=True)
# from moviepy.editor import VideoFileClip
# def reduce_video_quality(input_path, output_path, target_bitrate="1000k"):
#     video_clip = VideoFileClip(input_path)
    
#     # Set a lower bitrate codec
#     video_clip.write_videofile(output_path, codec="libx264", bitrate=target_bitrate)
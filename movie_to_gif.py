# Databricks notebook source
# MAGIC %pip install moviepy
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

from moviepy.editor import VideoFileClip

def convert_mov_to_gif(input_path, output_path, start_time=None, end_time=None, resize_factor=1.0):
    # Load the video file
    clip = VideoFileClip(input_path)
    
    # Optionally, trim the video if start_time and end_time are provided
    if start_time is not None and end_time is not None:
        clip = clip.subclip(start_time, end_time)
    
    # Optionally, resize the video
    if resize_factor != 1.0:
        clip = clip.resize(resize_factor)
    
    # Write the video clip to a GIF file
    clip.write_gif(output_path, fps=10)  # You can adjust fps for smoother or shorter GIFs

    clip = clip.speedx(factor=3)

    print(f"GIF created and saved to {output_path}")

# Example usage
convert_mov_to_gif('Multi-turn-Conversational-Agent.mov', 
                   'output_part2.gif',
                    start_time=100,
                     end_time=230, 
                     resize_factor=0.5)

# COMMAND ----------

# dbutils.fs.cp("file:/Workspace/Repos/ananya.roy@databricks.com/Ananya_Playground/blog_agent/output_part1.gif", "dbfs:/Volumes/mosaic_agent/agent/output/output_part1.gif")
dbutils.fs.cp("file:/Workspace/Repos/ananya.roy@databricks.com/Ananya_Playground/blog_agent/output_part2.gif", "dbfs:/Volumes/mosaic_agent/agent/output/output_part2.gif")

# COMMAND ----------



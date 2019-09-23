# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:27:54 2019

@author: rohin.selva
"""
from moviepy.editor import VideoFileClip

my_clip = VideoFileClip(r"C:\Users\rohin.selva\.spyder-py3\videoplayback.mp4","r")
print("Duration of video : ", my_clip.duration)
print("FPS : ", my_clip.fps)

my_clip = my_clip.subclip(2,7)
print('new clip duration : ',my_clip.duration)
my_clip.write_videofile("new_clip.mov", codec = "libx264", fps=25)
my_clip.close()


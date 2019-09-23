# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 05:09:08 2019

@author: rohin.selva
"""
import numpy as np # for numerical operations
from moviepy.editor import VideoFileClip, concatenate

clip = VideoFileClip(r"C:\Users\rohin.selva\Downloads\videoplayback.mp4","r")
cut = lambda i: clip.audio.subclip(i,i+1).to_soundarray(fps=22000)
volume = lambda array: np.sqrt(((1.0*array)**2).mean())
volumes = [volume(cut(i)) for i in range(0,int(clip.audio.duration-2))] 
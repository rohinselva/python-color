# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:24:06 2019

@author: rohin.selva
"""
import json

# read file
with open('changeneeded.json', 'r') as myfile:
    data=myfile.read()

# parse file
obj = json.loads(data)

# show values
print("Name: " + str(obj['Name']))
print("Yaw: " + str(obj['Yaw']))
print("gbp: " + str(obj['gbp']))
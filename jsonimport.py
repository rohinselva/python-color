# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:57:19 2019

@author: rohin.selva
"""

import json
#File I/O Open function for read data from JSON File
with open('sample.json') as file_object:
        # store file data in object
        data = json.load(file_object)
        
        for (k, v) in data.items():
            print("Key: " + k)
            print("Value: " + str(v))

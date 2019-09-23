# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:27:10 2019

@author: rohin.selva
"""
import random

start = 11
stop = 99
i = 0

def buildNumberTable(j):
        i = 0;
        list = []
        while i <= j:
                a = random.randrange(start, stop)
                list.append(a)
                i += 1;
        for b in list: print b
        print "\r"

while (i <= stop):
    buildNumberTable(66); #the parameter 66 fits to 15.4" screen if you have 13" put dif. number
    i += 1
    if i == stop:
        i=0

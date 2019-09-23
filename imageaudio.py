# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:24:59 2019

@author: rohin.selva
"""

import IPython.display as ipd
from IPython.display import HTML
import librosa
import librosa.display
import subprocess
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import io
import pickle
import scipy
import base64
import imageio
import itertools
import cv2
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

 
import keras
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
 
folder = f"{os.path.join(os.getcwd(), 'data', 'train')}"
 
class DataSet():
    
    def __init__(self, folder):
        self.folder = folder
        self.dataset = self.get_dataset()
    
    def get_dataset(self):
        dataset = pd.read_csv(f"{os.path.join(folder, 'labels.txt')}", sep='\t')
        dataset.columns = ['path', 'label']    
        dataset['name'] = dataset.path
        dataset.path = dataset.path.apply(lambda x: os.path.join(folder, x+'.mp4'))
        return dataset    
    
    def preprocess_input_resnet50(self, x):
        from keras.applications.resnet50 import preprocess_input
        X = np.expand_dims(x, axis=0)
        X = preprocess_input(X)
        return X[0]
    
    def preprocess_input_vgg16(self, x):
        from keras.applications.vgg16 import preprocess_input
        X = np.expand_dims(x, axis=0)
        X = preprocess_input(X)
        return X[0]
    
    def augment(self, src, choice):
        if choice == 0:
            # Rotate 90
            src = np.rot90(src, 1)
        if choice == 1:
            # flip vertically
            src = np.flipud(src)
        if choice == 2:
            # Rotate 180
            src = np.rot90(src, 2)
        if choice == 3:
            # flip horizontally
            src = np.fliplr(src)
        if choice == 4:
            # Rotate 90 counter-clockwise
            src = np.rot90(src, 3)
        if choice == 5:
            # Rotate 180 and flip horizontally
            src = np.rot90(src, 2)
            src = np.fliplr(src)
        if choice == 6:
            # leave it as is
            src = src
        if choice == 7:
            # shift
            src = scipy.ndimage.shift(src, 0.2)
        return src
        
    def prepare_image(self, img, size, preprocessing_function, aug=False):
        img = scipy.misc.imresize(img, size)
        img = np.array(img).astype(np.float64)
        if aug: img = self.augment(img, np.random.randint(8))
        img = preprocessing_function(img)
        return img
    
    def extract_audio(self, filepath):
        command = f"ffmpeg -i {filepath} -ab 160k -ac 2 -ar 44100 -vn {filepath.replace('mp4', 'wav')}"
        subprocess.call(command);
        
    def extract_audio_features(self, filepath, n_mfcc, return_ts=False):
        if not os.path.isfile(f"{filepath.replace('mp4', 'wav')}"): self.extract_audio(filepath)
        n_mfcc = 40 if n_mfcc is None else n_mfcc
        X, sample_rate = librosa.load(f"{filepath.replace('mp4', 'wav')}", res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc).T,axis=0)
        if not return_ts: return mfccs
        else: return mfccs, X, sample_rate
    
    def get_mfccs(self, n_mfcc):
        data = self.dataset
        data['mfcc'] = data.path.apply(lambda x: self.extract_audio_features(x, n_mfcc))
        return data
            
    def process_video(self, filepath, size, preprocessing_function, aug=False):
        vid = imageio.get_reader(filepath)
        nframes = vid.get_meta_data()['nframes']
        l = []
        for frame in range(0, nframes, 3): 
            try:
                l.append(self.prepare_image(vid.get_data(frame), size, preprocessing_function, aug=aug))
            except RuntimeError:
                pass
        return l
    
    def extract_features(self, size, which_net, what, audio=False, n_mfcc=None, return_ts=False, aug=False, iterations=1):
        if which_net == 'resnet50': 
            preprocessing_function=self.preprocess_input_resnet50
            base_model = ResNet50(weights='imagenet', include_top=False)
        elif which_net == 'vgg16': 
            preprocessing_function=self.preprocess_input_vgg16
            base_model = VGG16(weights='imagenet', include_top=False)
            
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        model = Model(input=base_model.input, output=x)
        
        train = []
        valid = []
        train_i = 0
        valid_i = 0
        train_dict = {'idx': defaultdict(list), 'data': None}
        valid_dict = {'idx': defaultdict(list), 'data': None}
                
        for path, label, name in self.dataset.values:
            for _ in range(iterations):
                print(f'Processing: {path}; Label: {label}')
                v = np.array(self.process_video(path, size, preprocessing_function, aug=aug))
                frames = v.shape[0]
                p = np.squeeze(model.predict_on_batch(v))
                p = np.hstack((np.tile(label, frames).reshape(-1, 1), p))
                if audio: 
                    if not return_ts:
                        mfccs = self.extract_audio_features(path, n_mfcc=n_mfcc)
                        mfccs = np.tile(mfccs, (frames, 1))
                        p = np.hstack((p, mfccs))
                    else:
                        mfccs, X, sample_rate = self.extract_audio_features(path, n_mfcc=n_mfcc, return_ts=True)
                        X = np.array_split(X, frames)
                        X = [np.random.choice(i, 100, replace=False) for i in X]
                        X = np.vstack(X)
                        p = np.hstack((p, X))                    
                if 'validation' in path:
                    valid.append(p)
                    valid_dict['idx'][name].append((valid_i, valid_i+frames))
                    valid_i+=frames
                else:
                    train.append(p)
                    train_dict['idx'][name].append((train_i, train_i+frames))
                    train_i+=frames
                if not aug: break
                
        
        valid_dict['data'] = np.vstack(valid)
        train_dict['data'] = np.vstack(train)
        
        with open(what+'_train.pickle', 'wb') as handle:
            pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(what+'_valid.pickle', 'wb') as handle:
            pickle.dump(valid_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)            
        
        return valid_dict, train_dict
    
    def map_frames_to_video(self, X, r, d):
        t_p = r.predict(X)
        l = []
        for k, v in d['idx'].items():
            l += [k]*(v[0][1]-v[0][0])
 
        d = pd.DataFrame({'name': l, 'frame_pred': t_p})
        d = d.groupby('name').frame_pred.mean().reset_index()
        d['video_pred'] = np.where(d.frame_pred > .5, 1, 0)
        return d
 
# instantiating the class
folder = cv2.VideoCapture("'C:\\Users\rohin.selva\Downloads\videoplayback.mp4'")

data = DataSet(folder)

valid_d, train_d = data.extract_features((224, 224), 'vgg16', 'frames_audio_trace', audio=True, n_mfcc=100, return_ts=True)
 
X_train, y_train = train_d['data'][:, 1:], train_d['data'][:, 0]
X_valid, y_valid = valid_d['data'][:, 1:], valid_d['data'][:, 0]
 
r = RandomForestClassifier(n_estimators=150, min_samples_leaf=19, oob_score=True, random_state=40)
r.fit(X_train, y_train)


valid_d, train_d = data.extract_features((224, 224), 'vgg16', 'frames_mfcc', audio=True, n_mfcc=100)
 
X_train, y_train = train_d['data'][:, 1:], train_d['data'][:, 0]
X_valid, y_valid = valid_d['data'][:, 1:], valid_d['data'][:, 0]
 
r = RandomForestClassifier(n_estimators=250, min_samples_leaf=25, oob_score=True, random_state=40)
r.fit(X_train, y_train)
########################################################
########################################################
# ML EXPERIMENTS GO HERE. AS SHOWN ABOVE
########################################################
########################################################
 
# from frame-level predictions to video-level picking the best model
Xv = data.map_frames_to_video(X_valid, r, valid_d)
Xt = data.map_frames_to_video(X_train, r, train_d)
Xv['split'] = 'valid'
Xt['split'] = 'train'
X = Xt.append(Xv).merge(data.dataset)
X['accuracy'] = X.label == X.video_pred
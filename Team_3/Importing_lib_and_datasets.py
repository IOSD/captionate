# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 11:47:29 2018

@author: Vijay Gupta
"""

#importing libraries
import numpy as np
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pickle import dump
from keras.preprocessing import image, sequence
from keras.applications import inception_v3
from keras.layers import Dense, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from os import listdir
#importing dataset
path1='Flicker8k_text/Flickr8k.token.txt'
path2='Flicker8k_text/Flickr_8k.trainImages.txt'
path3='Flicker8k_text/Flickr_8k.devImages.txt'
caption=open(path1,'r').read().split("\n")
train_img=open(path2,'r').read().split("\n")
val_img=open(path3,'r').read().split("\n")
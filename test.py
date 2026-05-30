# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:34:43 2019
Edited on Wed Jun 12 21:00:00 2024
@author: vikhy
@author: nswider
"""
import sys
# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image

from tensorflow.keras.models import load_model
 
classifier = load_model("bike_classifier.keras")
import tensorflow as tf
writer = tf.summary.create_file_writer("./logs_test")
#Appending the test set directory and file names to load later
from os import listdir
from os.path import isfile, join
onlyfiles = [r'./images/mountain'+'/'+f for f in listdir(r'./images/mountain') if isfile(join(r'./images/mountain', f))]
onlyfiles += [r'./images/road'+'/'+f for f in listdir(r'./images/road') if isfile(join(r'./images/road', f))]

#Shuffling the test set
from random import shuffle
shuffle(onlyfiles)

def predict_display(filename):
    test_image = image.load_img(filename, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    if result[0][0] == 0:
        prediction = 'mountain_bike'
    else:
        prediction ='road_bike'
   
    return prediction 

#Getting the results
result=[]
for filename in onlyfiles:
    result.append(predict_display(filename))
    

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.figure()
#%matplotlib inline
import io
for i in range(0, len(onlyfiles)):
    img=mpimg.imread(onlyfiles[i])
    plt.imshow(img)
    #plt.title(result[i])
    #plt.waitforbuttonpress()
    #plt.show() 
    # 1. Save the figure to a PNG buffer

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # 2. Decode the PNG buffer into a TensorFlow image tensor
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0) # Add batch dimension

    # 3. Log using TensorFlow's native API
    with writer.as_default():
        tf.summary.image(result[i]+onlyfiles[i], image, step=0) # ✅ Works for TensorFlow
 
sys.exit(0)
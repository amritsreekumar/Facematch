_Companyname_ = "Foundingminds"
_Author_ = "Amrit Sreekumar"

import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
import argparse
from configparser import SafeConfigParser
from scipy import misc
import json
import os
import time
from datetime import datetime


CONFIG_FILE = "settings.conf"
CONFIG_SECTION = "settings"
threshold = [0.6, 0.8, 0.9]

class facematch():
    def __init__(self):
        parser = SafeConfigParser()
        parser.read(CONFIG_FILE)
        self.classifier_pickle = parser.get(CONFIG_SECTION,"classifier_pickle")
        self.modelCNN = parser.get(CONFIG_SECTION,"model")
        self.gpu_memory_fraction = parser.getfloat(CONFIG_SECTION,"gpu_memory_fraction")
        self.margin = parser.getint(CONFIG_SECTION,"margin")
        self.image_size = parser.getint(CONFIG_SECTION,"image_size")
        self.minsize = parser.getint(CONFIG_SECTION,"minsize")
        self.factor = parser.getfloat(CONFIG_SECTION,"factor")
        self.json_file = parser.get(CONFIG_SECTION,"json_file")
        self.distance_threshold = parser.getfloat(CONFIG_SECTION,"distance_threshold")

matchobject = facematch()

#create mtcnn
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=matchobject.gpu_memory_fraction)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
pnet, rnet, onet = detect_face.create_mtcnn(sess, None)


def match_face(img1,img2):


    #load the model
    facenet.load_model(matchobject.modelCNN)


    #timestamp
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d:%m:%Y;%H:%M:%S.%f")

    #find the distance between images
    distance = compare2face(img1, img2)
    threshold = matchobject.distance_threshold    
    match_percentage = ((threshold - distance)/threshold)*100
    if(match_percentage<0):
        match_percentage = 0

    match = "True"
    if (match_percentage == 0):
        match = "False"

    to_json(match, match_percentage, timestampStr)

def getFace(img):
    margin = matchobject.margin
    try:
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = RGB_img
    except (IOError, ValueError, IndexError) as e:
        print(errorMessage)
    else:
        if img.ndim<2:
             print('Unable to align' )
        if img.ndim == 2:
            img = to_rgb(img)
        img = img[:,:,0:3]
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, matchobject.minsize, pnet, rnet, onet, threshold, matchobject.factor)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces>0:
            faces = []
            det = bounding_boxes[:,0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                det_arr.append(np.squeeze(det))

            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-margin/2, 0)
                bb[1] = np.maximum(det[1]-margin/2, 0)
                bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                resized = misc.imresize(cropped, (matchobject.image_size, matchobject.image_size), interp='bilinear')
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                #cv2.imwrite('xyz.jpg',resized)
                prewhitened = facenet.prewhiten(resized)
                faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]],'embedding':getEmbedding(prewhitened)})
    return faces

def getEmbedding(resized):
    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")    
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    #get embeddings
    reshaped = resized.reshape(-1,matchobject.image_size,matchobject.image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding

def compare2face(img1,img2):
    face1 = getFace(img1)
    face2 = getFace(img2)
    if face1 and face2:
        # calculate Euclidean distance
        dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
        return dist
    return -1

def to_json(match, match_percentage, timestamp):

    fname = matchobject.json_file
    # Data to be written 
    dictionary ={ 
        "timestamp" : timestamp, 
        "match" : match, 
        "matchpercentage" : match_percentage
    } 
      
    # Writing to sample.json 
    
    if not os.path.isfile(fname):
        data = [dictionary]

    else:
        data = json.load(open(fname))
        data.append(dictionary)
    
    with open(fname, 'w') as outfile:
            json.dump(data, outfile, indent=3)
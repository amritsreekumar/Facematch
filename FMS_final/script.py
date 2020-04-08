
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_Companyname_ = "Foundingminds"
_Author_ = "Amrit Sreekumar"


import cv2
import time
from datetime import datetime
import json
import traceback
import os
import base64
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS


import random
from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import detect_face
import random
from time import sleep


import facenet
import sys
import math
import pickle
from sklearn.svm import SVC
from skimage import transform
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

gpu_memory_fraction = 0.75
margin = 44
batch_size = 1


url = 'rtsp://192.168.1.10:554/user=admin&password=&channel=1&stream=0.sdp?' 

vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

#input_dir = '/Users/amritsreekumar/Desktop/FMS_final/input_dir/images'
output_dir = '/Users/amritsreekumar/Desktop/FMS_final/output_dir'
image_size = 160
sleep(random.random())
output_dir = os.path.expanduser(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
print('Creating networks and loading parameters')
    
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None) 
    
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

print('Loading feature extraction model')
with tf.Graph().as_default():
    with tf.Session() as sess:
        np.random.seed(seed=666)
        # Get input and output tensors
        facenet.load_model('model.pb')
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        while(True):
            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            dateTimeObj = datetime.now()
            timestampStr = dateTimeObj.strftime("(%H:%M:%S.%f)")
            #cv2.imshow('frame',frame)
            #ret1,buffer = cv2.imencode('.jpg', frame)
            #image_path = os.path.join(input_dir ,timestampStr + '.jpg')
            #cv2.imwrite(os.path.join(input_dir ,timestampStr + '.jpg'), frame)
    




            random_key = np.random.randint(0, high=99999)
            ''' bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    
            with open(bounding_boxes_filename, "w") as text_file:
                nrof_images_total = 0
                nrof_successfully_aligned = 0''' 
            output_class_dir = os.path.join(output_dir)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
            #nrof_images_total += 1
            filename = timestampStr
            output_filename = os.path.join(output_class_dir, filename+'.png')
            #print(image_path)
            if not os.path.exists(output_filename):
                try:
                    RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = RGB_img
                except (IOError, ValueError, IndexError) as e:
                    #errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim<2:
                        print('Unable to align' )
                        #text_file.write('%s\n' % (output_filename))
                    if img.ndim == 2:
                        img = to_rgb(img)
                    img = img[:,:,0:3]
    
                    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces>0:
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
                            try:
                                scaled = transform.resize(cropped, (image_size, image_size))
                                #nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)
                                output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                misc.imsave(output_filename_n, scaled)
                                #text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))           
                            except(ValueError) as e:
                                continue
                            
                            paths = [output_filename_n]                        
                            # Run forward pass to calculate embeddings
                            #print('Calculating features for images')
                            emb_array = np.zeros((1, embedding_size))
                            start_index = 0
                            end_index = min((1)*batch_size, 1)
                            paths_batch = paths
                            images = np.zeros((1, 160, 160, 3))
                            img = misc.imread(paths_batch[0])
                            if img.ndim == 2:
                                img = to_rgb(img)
                            img = facenet.prewhiten(img)
                            img = facenet.crop(img, False, 160)
                            img = facenet.flip(img, False)
                            images[0,:,:,:] = img
                            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                            emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
                            classifier_filename_exp = os.path.expanduser('my_classifier.pkl')
                
                            #print('Testing classifier')
                            with open(classifier_filename_exp, 'rb') as infile:
                                (model, class_names) = pickle.load(infile)
                            #print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            try:
                                if best_class_probabilities[i] > 0.45:
                                    print('%s: %.3f' % (class_names[best_class_indices[i]], best_class_probabilities[i]))
                                else:
                                    print("Unknown")
                            except:
                                continue
                    else:
                        continue
                        #text_file.write('%s\n' % (output_filename))
                            
            #print("Total number of images: %d" % nrof_images_total)
            #print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
            #align_dataset_mtcnn.main(input_dir1,output_dir,img_size, image_path)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


fps.stop()
cv2.destroyAllWindows()
vs.stop()
            
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
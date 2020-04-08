
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import time
from datetime import datetime
import json
import traceback
import align_dataset_mtcnn
import os
import base64

import random
from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep
#import cv2
#from mtcnn import MTCNN

gpu_memory_fraction = 0.75
margin = 44

#detector = MTCNN()

url = 'rtsp://192.168.1.10:554/user=admin&password=&channel=1&stream=0.sdp?' 

cap = cv2.VideoCapture(0)

#input_dir = '/Users/amritsreekumar/Desktop/FMS_final/input_dir'
input_dir = '/Users/amritsreekumar/Desktop/FMS_final/input_dir/images'
output_dir = '/Users/amritsreekumar/Desktop/FMS_final/output_dir'
image_size = 160
sleep(random.random())
output_dir = os.path.expanduser(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    #src_path,_ = os.path.split(os.path.realpath(__file__))
    #facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
dataset = facenet.get_dataset(input_dir)
    
print('Creating networks and loading parameters')
    
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None) 
    
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor



while(True):
    ret, frame = cap.read()
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("(%H:%M:%S.%f)")
    if cap.isOpened() == 0:
        cap.open(0)

    #cv2.imshow('frame',frame)
    #ret1,buffer = cv2.imencode('.jpg', frame)
    image_path = os.path.join(input_dir ,timestampStr + '.jpg')
    cv2.imwrite(os.path.join(input_dir ,timestampStr + '.jpg'), frame)




    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    
    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
    output_class_dir = os.path.join(output_dir)
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)
    nrof_images_total += 1
    filename = os.path.splitext(os.path.split(image_path)[1])[0]
    output_filename = os.path.join(output_class_dir, filename+'.png')
    print(image_path)
    if not os.path.exists(output_filename):
        try:
            img = misc.imread(image_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(image_path, e)
            print(errorMessage)
        else:
            if img.ndim<2:
                print('Unable to align "%s"' % image_path)
                text_file.write('%s\n' % (output_filename))
            if img.ndim == 2:
                img = facenet.to_rgb(img)
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
                    scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    nrof_successfully_aligned += 1
                    filename_base, file_extension = os.path.splitext(output_filename)
                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                    misc.imsave(output_filename_n, scaled)
                    text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
            else:
                print('Unable to align "%s"' % image_path)
                text_file.write('%s\n' % (output_filename))
                            
    print("Total number of images: %d" % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    #align_dataset_mtcnn.main(input_dir1,output_dir,img_size, image_path)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
            

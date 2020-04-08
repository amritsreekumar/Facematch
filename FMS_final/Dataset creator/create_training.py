
_Companyname_ = "Foundingminds"
_Author_ = "Amrit Sreekumar"


import cv2
import time
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


input_dir = 'training_videos/'
output_dir = 'train/'

print("Enter the name of the person to be trained")
name = input()
name2 = os.path.join(input_dir,name)
output_dir = os.path.expanduser(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(name2 + '.mp4')

output_class_dir = os.path.join(output_dir,name)
if not os.path.exists(output_class_dir):
    os.makedirs(output_class_dir)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("(%H:%M:%S.%f)")
    #cv2.imshow('frame',frame)
    #ret1,buffer = cv2.imencode('.jpg', frame)
    #image_path = os.path.join(input_dir ,timestampStr + '.jpg')
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(output_class_dir ,timestampStr + '.jpg'), frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
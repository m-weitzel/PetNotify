
# Based on the work of Evan Juras (https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi)
# Using the TensorFlow Object Detection API on a Raspberry Pi

# The notification feature requires a setup of notify-run. For this, go to http://notify.run and follow the instructions.


import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
from notify_run import Notify
import pickle
from datetime import datetime as dt
import time


IM_WIDTH = 1280
IM_HEIGHT = 720

camera_type = 'picamera'
parser = argparse.ArgumentParser()

# Initialize Web notification interface
notify = Notify()


# Initialize TensorFlow model ####

from utils import label_map_util
from utils import visualization_utils as vis_util

sys.path.append('..')
sys.path.append('/home/pi/tensorflow1/models/research/Object_detection/')

# Change model location here
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Boot up the model functionality
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

TL_outside = (int(IM_WIDTH*0.46),int(IM_HEIGHT*0.25))
BR_outside = (int(IM_WIDTH*0.8),int(IM_HEIGHT*.85))

detected_outside = False

outside_counter = 0

pause = 0
pause_counter = 0


def pet_detector(frame):

    global detected_outside, outside_counter
    global pause, pause_counter

    frame_expanded = np.expand_dims(frame, axis=0)

    # Run model on inserted frame
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Dump classes and scores to file for analysis / debugging
    classes = classes[0]
    scores = scores[0]
    pickle.dump([boxes, scores, classes, num], open('classes.dat', 'wb'))
    
    # Visualization of detections
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.40)

    # cv2.rectangle(frame,TL_outside,BR_outside,(255,20,20),3)
    # cv2.putText(frame,"Outside box",(TL_outside[0]+10,TL_outside[1]-10),font,1,(255,20,255),3,cv2.LINE_AA)

    wanted_classes = [17, 18, 88]  # 17: Cat, 18: Dog
    min_confidence = 0.3

    found_combinations = list()

    for wanted in wanted_classes:
        acc_class = [(int(cls) == wanted) for cls in classes]
        if any(acc_class):
            here = [i for i, x in enumerate(acc_class) if x]
            score_for_wanted = scores[acc_class]
            this_combis = [(classes[cls], sc) for cls, sc in zip(here, score_for_wanted) if sc > min_confidence]
            found_combinations.append(this_combis)

    # accept = [(int(cls)==17)&(int(conf)>0.3) for cls, conf in zip(classes, scores)]

    # if (((int(classes[0]) == 17) or (int(classes[0] == 18) or (int(classes[0]) == 44))) and (pause == 0)):
    if (len(found_combinations) > 0) & ~pause:
        x = int(((boxes[0][0][1]+boxes[0][0][3])/2)*IM_WIDTH)
        y = int(((boxes[0][0][0]+boxes[0][0][2])/2)*IM_HEIGHT)

        # Draw a circle at center of object
        cv2.circle(frame,(x,y), 5, (75, 13, 180), -1)
        # cv2.text(frame, (x, y), found_combinations)

        # Increase counter by one
        outside_counter = outside_counter + 1

    # If more than 10 detections are accumulated, send a notification
    if outside_counter > 10:
        detected_outside = True

        notify.send('Katze detected!')
        
        # Save detected image to local storage
        det_time = dt.strftime(dt.utcnow(), '%H:%M:%S')
        cv2.imwrite('/home/pi/DetectionImages/{}.png'.format(det_time), frame)

        outside_counter = 0
        # Pause detection
        pause = 1

    # If pause flag is set, draw message on screen.
    if pause == 1:
        # Pause for a certain number of frames - to be adjusted for frame rate!
        pause_counter = pause_counter + 1
        if pause_counter > 30:
            pause = 0
            pause_counter = 0
            detected_outside = False

    # Draw counter info
    cv2.putText(frame, 'Detection counter: ' + str(outside_counter), (10, 100), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Pause counter: ' + str(pause_counter),(10,150),font,0.5, (255, 255, 0), 1, cv2.LINE_AA)

    return frame


if camera_type == 'picamera':
    camera = PiCamera()
    camera.rotation = 180  # adjust depending on camera orientation
    camera.resolution = (IM_WIDTH, IM_HEIGHT)
    # camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # reshape to 1D array of color pixels
        frame = frame1.array
        frame.setflags(write=1)

        # Call actual object detection function
        frame = pet_detector(frame)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        cv2.imshow('Object detector', frame)

        # Delaying image acquisition by a set time - change to adjust Raspi Workload
        time.sleep(5)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Option to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)
        

    camera.close()

cv2.destroyAllWindows()

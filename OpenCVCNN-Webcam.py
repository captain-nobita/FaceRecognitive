from __future__ import division
import cv2
import time
import sys
import numpy as np
import glob
import os
from keras.models import load_model
import FaceNet
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img

font = cv2.FONT_HERSHEY_SIMPLEX

def detectFaceOpenCVDnn(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


def image_to_embedding(image, model):
    # image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (96, 96))
    img = image[..., ::-1]
    img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


# ## About <font color=blue>recognize_face</font> function
# This function calculate similarity between the captured image and the images that are already been stored. It passes the image to the trained neural network to generate its embedding vector. Which is then compared with all the embedding vectors of the images stored by calculating L2 Euclidean distance.
#
# If the minimum L2 distance between two embeddings is less than a threshpld (here I have taken the threashhold as .68 (which can be adjusted) then we have a match.

# In[5]:


def recognize_face(face_image, input_embeddings, model):
    embedding = image_to_embedding(face_image, model)

    minimum_distance = 200
    name = None

    # Loop over  names and encodings.
    for (input_name, input_embedding) in input_embeddings.items():

        euclidean_distance = np.linalg.norm(embedding - input_embedding)

        print('Euclidean distance from %s is %s' % (input_name, euclidean_distance))

        if euclidean_distance < minimum_distance:
            minimum_distance = euclidean_distance
            name = input_name

    if minimum_distance < 0.68:
        return str(name)
    else:
        return None

def create_input_image_embeddings(model):
    input_embeddings = {}

    for file in glob.glob("images/*"):
        person_name = os.path.splitext(os.path.basename(file))[0]
        image_file = cv2.imread(file, 1)
        input_embeddings[person_name] = image_to_embedding(image_file, model)

    return input_embeddings

if __name__ == "__main__" :

    # OpenCV DNN supports 2 networks.
    # 1. FP16 version of the original caffe implementation ( 5.4 MB )
    # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
    DNN = "TF"
    modelFile = "models/opencv_face_detector_uint8.pb"
    configFile = "models/opencv_face_detector.pbtxt"
    net_detection = cv2.dnn.readNetFromTensorflow(modelFile, configFile)


    net_recognition = FaceNet.create_model()
    net_recognition.load_weights("models/facenet_weights.h5")
    input_embeddings = create_input_image_embeddings(net_recognition)

    conf_threshold = 0.7

    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture(source)
    hasFrame, frame = cap.read()

    frame_count = 0
    tt_opencvDnn = 0
    while(True):
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        frame_count += 1

        height, width, channels = frame.shape

        t = time.time()
        outOpencvDnn, bboxes = detectFaceOpenCVDnn(net_detection,frame)
        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn

        for (x1, y1, x2, y2) in bboxes:
            face_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            identity = recognize_face(face_image, input_embeddings, net_recognition)

            if identity is not None:
                # img = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(outOpencvDnn, str(identity), (x1 + 5, y1 - 5), font, 1, (255, 255, 255), 2)

        cv2.imshow("Face Detection Comparison", outOpencvDnn)

        if frame_count == 1:
            tt_opencvDnn = 0

        k = cv2.waitKey(10)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

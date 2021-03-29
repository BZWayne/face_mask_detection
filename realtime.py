from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import keras
import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt
# %matplotlib inline

# from google.colab import drive
# drive.mount('/content/drive/', force_remount=True)

model = tf.keras.models.load_model("./model_trained.h5")
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"

# from google.colab.patches import cv2_imshow

# cap = cv2.VideoCapture("/Users/bzwayne/Desktop/client/input_video.mp4")
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('/Users/bzwayne/Desktop/client/output_video.mp4', fourcc, 20.0, (640,480))

while (cap.isOpened()):
    
    ret, frame = cap.read()
    
    if ret == True:
        
        # frame = cv2.flip(frame,0)
        # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # faces = face_cascade.detectMultiScale(gray, 1.3, 8)
        # for (x, y, w, h) in faces:
        #     face = frame[y:y+h, x:x+w]
        #     face = cv2.resize(face, (224, 224))
        #     face = img_to_array(face)
        #     face = preprocess_input(face)
        #     face = np.expand_dims(face, axis=0)

        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces = net.forward()
        #to draw faces on image
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                # (x, y, x1, y1) = box.astype("int")
                # cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

                (mask, no_mask) = model.predict(box)[0]
                mask, no_mask = mask*100, no_mask*100
                    
                if mask > no_mask:
                    cv2.putText(frame,
                                "Mask: " + str("%.2f" % round(mask)),
                                (x-10,y-8), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 5)
                else:
                    cv2.putText(frame,
                                "No Mask: " + str("%.2f" % round(no_mask)),
                                (x-5,y-15), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)

                cv2.imshow("frame", frame)
                # out.write(frame)

                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break
    else:
        break



cap.release()
# out.release()
cv2.destroyAllWindows()
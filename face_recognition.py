import cv2 as cv
import numpy as np
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

# Khởi tạo model
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Khởi tạo camera
cap = cv.VideoCapture(0)

# Biến theo dõi
current_name = None
start_time = None

while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    
    for x, y, w, h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160, 160))  # 1x160x160x3
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)
        final_name = encoder.inverse_transform(face_name)[0]
        
        # Vẽ khung nhận diện
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
        cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 3, cv.LINE_AA)
        
        # Kiểm tra thời gian nhận diện liên tục
        if current_name == final_name:
            if time.time() - start_time >= 5:  # Nhận diện đủ 5 giây
                print(f"Đã nhận diện: {final_name}")
                cap.release()
                cv.destroyAllWindows()
                exit(0)
        else:
            current_name = final_name
            start_time = time.time()  # Bắt đầu tính thời gian nhận diện khuôn mặt mới

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

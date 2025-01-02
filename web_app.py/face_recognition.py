# import cv2 as cv
# import numpy as np
# import os
# import time
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder
# import pickle
# from keras_facenet import FaceNet
# import mysql.connector
# from datetime import datetime
# import serial

# # Initialize model
# facenet = FaceNet()
# faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
# Y = faces_embeddings['arr_1']
# encoder = LabelEncoder()
# encoder.fit(Y)
# haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
# model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# # Initialize camera
# cap = cv.VideoCapture(0)

# # Database connection
# conn = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="",
#     database="smartlocksystem"
# )
# cursor = conn.cursor()

# # Save login history
# def them_lich_su_dang_nhap(ten_user, anh=None):
#     try:
#         if anh is not None:
#             _, anh_encoded = cv.imencode('.jpg', anh)
#             anh_bytes = anh_encoded.tobytes()

#             query = "INSERT INTO LichSuDangNhap (Ten, ThoiDiemDangNhap, Anh) VALUES (%s, %s, %s)"
#             values = (ten_user, datetime.now(), anh_bytes)
#             cursor.execute(query, values)
#             conn.commit()
#             print(f"Lịch sử đăng nhập cho '{ten_user}' đã được lưu thành công!")
#         else:
#             print("Ảnh không hợp lệ, không thể lưu.")
#     except mysql.connector.Error as err:
#         print(f"Đã xảy ra lỗi: {err}")
#         conn.rollback()

# # Tracking variables
# current_name = None
# start_time = None
# unknown_start_time = None  # New variable to track unknown face start time
# unknown_count = 0

# try:
#     while cap.isOpened():
#         _, frame = cap.read()
#         rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#         gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

#         recognized = False

#         for x, y, w, h in faces:
#             img = rgb_img[y:y+h, x:x+w]
#             img = cv.resize(img, (160, 160))
#             img = np.expand_dims(img, axis=0)
#             ypred = facenet.embeddings(img)
#             probabilities = model.predict_proba(ypred)
#             max_prob = np.max(probabilities)

#             if max_prob < 0.4:
#                 final_name = "unknown"
#             else:
#                 face_name = model.predict(ypred)
#                 final_name = encoder.inverse_transform(face_name)[0]
#                 recognized = True

#             cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
#             cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 
#                        1, (0, 0, 255), 3, cv.LINE_AA)

#             # Handle known face recognition
#             if current_name == final_name and final_name != "unknown":
#                 if time.time() - start_time >= 3:
#                     print(f"Đã nhận diện: {final_name}")
#                     them_lich_su_dang_nhap(final_name, frame)
#                     with serial.Serial('COM3', 9600, timeout=1) as arduino:
#                         time.sleep(2)  # Wait for the serial connection to initialize
#                         arduino.write(b'1')  # Send command 1 to Arduino
#                     start_time = time.time()
#             else:
#                 current_name = final_name
#                 start_time = time.time()

#         # Handle unknown face
#         if current_name == "unknown":
#             if unknown_start_time is None:
#                 unknown_start_time = time.time()
            
#             if time.time() - unknown_start_time >= 5:
#                 print("Không nhận diện được khuôn mặt. Đang lưu thông tin...")
#                 them_lich_su_dang_nhap("Unknown", frame)
#                 unknown_count += 1
#                 with serial.Serial('COM3', 9600, timeout=1) as arduino:
#                     time.sleep(2)  # Wait for the serial connection to initialize
#                     if unknown_count % 3 == :
#                         arduino.write(b'2')  # Send command 2 to Arduino
#                         unknown_count = 0
#                     else:
#                         arduino.write(b'3')  # Send command 3 to Arduino
#                 unknown_start_time = None  # Reset the timer after handling unknown face

#         else:
#             unknown_start_time = None  # Reset the timer if a known face is detected

#         cv.imshow("Face Recognition:", frame)
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break

# finally:
#     cap.release()
#     cv.destroyAllWindows()

import cv2 as cv
import numpy as np
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
import mysql.connector
from datetime import datetime
import serial

# Initialize model
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Initialize camera
cap = cv.VideoCapture(0)

# Database connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="smartlocksystem"
)
cursor = conn.cursor()

# Save login history
def them_lich_su_dang_nhap(ten_user, anh=None):
    try:
        if anh is not None:
            _, anh_encoded = cv.imencode('.jpg', anh)
            anh_bytes = anh_encoded.tobytes()

            query = "INSERT INTO LichSuDangNhap (Ten, ThoiDiemDangNhap, Anh) VALUES (%s, %s, %s)"
            values = (ten_user, datetime.now(), anh_bytes)
            cursor.execute(query, values)
            conn.commit()
            print(f"Lịch sử đăng nhập cho '{ten_user}' đã được lưu thành công!")
        else:
            print("Ảnh không hợp lệ, không thể lưu.")
    except mysql.connector.Error as err:
        print(f"Đã xảy ra lỗi: {err}")
        conn.rollback()

# Tracking variables
current_name = None
start_time = None
unknown_start_time = None  # New variable to track unknown face start time
unknown_count = 0

try:
    while cap.isOpened():
        _, frame = cap.read()
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

        if len(faces) == 0:
            current_name = None  # Reset current name when no face is detected
            start_time = None  # Reset start time when no face is detected
            unknown_start_time = None  # Reset unknown start time when no face is detected

        recognized = False

        for x, y, w, h in faces:
            img = rgb_img[y:y+h, x:x+w]
            img = cv.resize(img, (160, 160))
            img = np.expand_dims(img, axis=0)
            ypred = facenet.embeddings(img)
            probabilities = model.predict_proba(ypred)
            max_prob = np.max(probabilities)

            if max_prob < 0.4:
                final_name = "unknown"
            else:
                face_name = model.predict(ypred)
                final_name = encoder.inverse_transform(face_name)[0]
                recognized = True

            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
            cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 3, cv.LINE_AA)

            # Handle known face recognition
            if current_name == final_name and final_name != "unknown":
                if time.time() - start_time >= 3:
                    print(f"Đã nhận diện: {final_name}")
                    them_lich_su_dang_nhap(final_name, frame)
                    with serial.Serial('COM3', 9600, timeout=1) as arduino:
                        time.sleep(2)  # Wait for the serial connection to initialize
                        arduino.write(b'1')  # Send command 1 to Arduino
                    start_time = time.time()
            else:
                current_name = final_name
                start_time = time.time()

        # Handle unknown face
        if current_name == "unknown":
            if unknown_start_time is None:
                unknown_start_time = time.time()
            
            if time.time() - unknown_start_time >= 5:
                print("Không nhận diện được khuôn mặt. Đang lưu thông tin...")
                them_lich_su_dang_nhap("Unknown", frame)
                unknown_count += 1
                with serial.Serial('COM3', 9600, timeout=1) as arduino:
                    time.sleep(2)  # Wait for the serial connection to initialize
                    if unknown_count % 3 == 0:
                        arduino.write(b'2')  # Send command 2 to Arduino
                        unknown_count = 0
                    else:
                        arduino.write(b'3')  # Send command 3 to Arduino
                unknown_start_time = None  # Reset the timer after handling unknown face

        else:
            unknown_start_time = None  # Reset the timer if a known face is detected

        cv.imshow("Face Recognition:", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv.destroyAllWindows()
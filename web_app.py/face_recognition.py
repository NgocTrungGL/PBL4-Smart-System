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

# # Khởi tạo model
# facenet = FaceNet()
# faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
# Y = faces_embeddings['arr_1']
# encoder = LabelEncoder()
# encoder.fit(Y)
# haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
# model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# # Khởi tạo camera
# cap = cv.VideoCapture(0)
# anh = None

# # CSDL
# conn = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="",
#     database="smartlocksystem"
# )
# cursor = conn.cursor()

# def them_lich_su_dang_nhap(ten_user, anh=None):
#     try:
#         if anh is not None:
#             # Chuyển ảnh từ mảng NumPy (frame) sang định dạng nhị phân (bytes)
#             _, anh_encoded = cv.imencode('.jpg', anh)
#             anh_bytes = anh_encoded.tobytes()

#             # Cập nhật thời gian đăng nhập và tên người dùng
#             query = "INSERT INTO LichSuDangNhap (Ten, ThoiDiemDangNhap, Anh) VALUES (%s, %s, %s)"
#             values = (ten_user, datetime.now(), anh_bytes)

#             # Thực hiện câu lệnh SQL
#             cursor.execute(query, values)

#             # Lưu thay đổi vào cơ sở dữ liệu
#             conn.commit()
#             print("Lịch sử đăng nhập đã được lưu thành công!")
#         else:
#             print("Ảnh không hợp lệ, không thể lưu.")
            
#     except mysql.connector.Error as err:
#         print(f"Đã xảy ra lỗi: {err}")
#         conn.rollback()

# # Biến theo dõi
# current_name = None
# start_time = None
# unknown_start_time = time.time()

# while cap.isOpened():
#     _, frame = cap.read()
#     rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#     gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    
#     recognized = False
    
#     for x, y, w, h in faces:
#         img = rgb_img[y:y+h, x:x+w]
#         img = cv.resize(img, (160, 160))
#         img = np.expand_dims(img, axis=0)
#         ypred = facenet.embeddings(img)
#         probabilities = model.predict_proba(ypred)
#         max_prob = np.max(probabilities)
#         if max_prob < 0.5:
#             final_name = "unknown"
#         else:
#             face_name = model.predict(ypred)
#             final_name = encoder.inverse_transform(face_name)[0]
#             recognized = True
        
#         cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
#         cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
#                    1, (0, 0, 255), 3, cv.LINE_AA)
        
#         if current_name == final_name and final_name != "unknown":
#             if time.time() - start_time >= 3:
#                 print(f"Đã nhận diện: {final_name}")
#                 anh = frame
#                 them_lich_su_dang_nhap(str(final_name), anh)
#                 time.sleep()  # Dừng 3 giây trước khi tiếp tục
#                 start_time = time.time()
#                 # cap.release()
#                 # cv.destroyAllWindows()
#                 # exit(0)
#         else:
#             current_name = final_name
#             start_time = time.time()

#     if not recognized:
#         if current_name == "unknown" and time.time() - start_time >= 5:
#             print("Không nhận diện được khuôn mặt. Đang lưu thông tin...")
#             anh = frame
#             them_lich_su_dang_nhap("Unknown", anh)
#             time.sleep(5)  # Dừng 3 giây trước khi tiếp tục
#             start_time = time.time()
#     else:
#         unknown_start_time = time.time()

#     cv.imshow("Face Recognition:", frame)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()

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

try:
    while cap.isOpened():
        _, frame = cap.read()
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

        recognized = False

        for x, y, w, h in faces:
            img = rgb_img[y:y+h, x:x+w]
            img = cv.resize(img, (160, 160))
            img = np.expand_dims(img, axis=0)
            ypred = facenet.embeddings(img)
            probabilities = model.predict_proba(ypred)
            max_prob = np.max(probabilities)

            if max_prob < 0.5:
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
                    time.sleep(5)
                    start_time = time.time()
            else:
                current_name = final_name
                start_time = time.time()

        # Handle unknown face
        if not recognized and current_name == "unknown" and time.time() - start_time >= 5:
            print("Không nhận diện được khuôn mặt. Đang lưu thông tin...")
            them_lich_su_dang_nhap("Unknown", frame)
            time.sleep(5)
            start_time = time.time()

        cv.imshow("Face Recognition:", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv.destroyAllWindows()

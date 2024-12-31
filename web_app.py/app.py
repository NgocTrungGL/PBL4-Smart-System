from flask import Flask, render_template, request, redirect, url_for
import mysql.connector

app = Flask(__name__)

# Kết nối tới MySQL
def get_database_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",  # Thay bằng username MySQL của bạn
        password="",  # Thay bằng mật khẩu MySQL của bạn
        database="smartlocksystem"  # Thay bằng tên database
    )
    return connection

# Route trang chủ
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

# Route đăng nhập
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['Username']
        password = request.form['PasswordHash']
        # Xử lý đăng nhập (kiểm tra cơ sở dữ liệu)
        conn = get_database_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM admin WHERE Username = %s AND PasswordHash = %s", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Đăng nhập thất bại. Vui lòng thử lại.")

    return render_template('login.html')

# Route hỗ trợ
@app.route('/help')
def help_page():
    return render_template('help.html')

# Route lịch sử hoạt động
import base64

@app.route('/history')
def history():
    conn = get_database_connection()
    cursor = conn.cursor(dictionary=True)
    # Lấy dữ liệu bao gồm ảnh dưới dạng binary
    cursor.execute("SELECT LanDangNhap, Ten, ThoiDiemDangNhap, Anh FROM lichsudangnhap")  
    history_data = cursor.fetchall()
    
    # Chuyển ảnh sang dạng Base64
    for record in history_data:
        if record['Anh']:
            record['Anh'] = base64.b64encode(record['Anh']).decode('utf-8')
        else:
            record['Anh'] = None  # Nếu không có ảnh, gán None
    
    conn.close()
    return render_template('history.html', history=history_data)



# Route quản lý người dùng (đã có)
@app.route('/usermanagement')
def user_management():
    conn = get_database_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT AdminID, Username, PasswordHash, Email FROM admin")
    users = cursor.fetchall()
    conn.close()

    return render_template('usermanagement.html', users=users)

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, render_template, Response
# import cv2
# import numpy as np
# import mysql.connector
# from datetime import datetime
# from keras_facenet import FaceNet
# from sklearn.preprocessing import LabelEncoder
# import pickle
# import time

# app = Flask(__name__)

# # Khởi tạo model và các phần mềm hỗ trợ nhận diện khuôn mặt
# facenet = FaceNet()
# faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
# Y = faces_embeddings['arr_1']
# encoder = LabelEncoder()
# encoder.fit(Y)
# haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# # Khởi tạo camera
# cap = cv2.VideoCapture(0)

# # Kết nối đến cơ sở dữ liệu MySQL
# conn = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="",
#     database="smartlocksystem"
# )
# cursor = conn.cursor()

# # Hàm lưu lịch sử đăng nhập vào CSDL
# def them_lich_su_dang_nhap(ten_user, anh=None):
#     try:
#         if anh is not None:
#             _, anh_encoded = cv2.imencode('.jpg', anh)
#             anh_bytes = anh_encoded.tobytes()

#             query = "INSERT INTO LichSuDangNhap (Ten, ThoiDiemDangNhap, Anh) VALUES (%s, %s, %s)"
#             values = (ten_user, datetime.now(), anh_bytes)
#             cursor.execute(query, values)
#             conn.commit()
#             print("Lịch sử đăng nhập đã được lưu thành công!")
#         else:
#             print("Ảnh không hợp lệ, không thể lưu.")
#     except mysql.connector.Error as err:
#         print(f"Đã xảy ra lỗi: {err}")
#         conn.rollback()

# # Hàm xử lý video stream và nhận diện khuôn mặt
# def generate():
#     current_name = None
#     start_time = None
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Chuyển đổi ảnh sang RGB và nhận diện khuôn mặt
#         rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

#         recognized = False
        
#         for (x, y, w, h) in faces:
#             img = rgb_img[y:y+h, x:x+w]
#             img = cv2.resize(img, (160, 160))
#             img = np.expand_dims(img, axis=0)
#             ypred = facenet.embeddings(img)
#             probabilities = model.predict_proba(ypred)
#             max_prob = np.max(probabilities)
            
#             if max_prob < 0.5:
#                 final_name = "unknown"
#             else:
#                 face_name = model.predict(ypred)
#                 final_name = encoder.inverse_transform(face_name)[0]
#                 recognized = True

#             # Vẽ khung nhận diện
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
#             cv2.putText(frame, str(final_name), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

#             # Lưu ảnh nếu nhận diện thành công
#             if recognized and current_name != final_name:
#                 current_name = final_name
#                 start_time = time.time()
#                 if time.time() - start_time >= 5:  # 5 giây nhận diện
#                     print(f"Đã nhận diện: {final_name}")
#                     them_lich_su_dang_nhap(str(final_name), frame)

#         # Chuyển frame video thành byte để gửi qua HTTP
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         if not ret:
#             break

#         # Trả lại video frame dưới dạng byte stream
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# # Route chính để hiển thị video trên web
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route stream video (truyền video trực tiếp tới frontend)
# @app.route('/video_feed')
# def video_feed():
#     return Response(generate(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)

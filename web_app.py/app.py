import os
import cv2
from flask import Flask, render_template, request, redirect, url_for
import mysql.connector
import numpy as np

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


################TEST#####################
def save_image(folder_name, image_base64, count):
    # Tạo đường dẫn thư mục
    dataset_path = r"D:\Code\PBL4-Smart-System\dataset"

    folder_path = os.path.join(dataset_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Chuyển base64 thành ảnh
    try:
        image_data = base64.b64decode(image_base64.split(",")[1])
    except (IndexError, ValueError):
        return "Dữ liệu base64 không hợp lệ."

    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Lưu ảnh
    img_path = os.path.join(folder_path, f"{count}.jpg")
    if cv2.imwrite(img_path, image):
        return f"Ảnh {count} đã được lưu tại {img_path}"
    else:
        return "Không thể lưu ảnh."

# Giao diện thêm khuôn mặt
@app.route('/add_face')
def add_face():
    return render_template('add_face.html')

# Xử lý lưu ảnh từ client
@app.route('/save_snapshot', methods=['POST'])
def save_snapshot():
    data = request.json
    folder_name = data['folder_name']
    image_base64 = data['image']

    # Đếm số lượng ảnh đã lưu
    dataset_path = r"D:\Code\PBL4-Smart-System\dataset"

    folder_path = os.path.join(dataset_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    count = len(os.listdir(folder_path)) + 1 if os.path.exists(folder_path) else 1

    # Lưu ảnh
    if count <= 10:  # Giới hạn lưu 10 ảnh
        result = save_image(folder_name, image_base64, count)
        return result
    else:
        return "Đã đủ 10 ảnh, không lưu thêm."

if __name__ == '__main__':
    app.run(debug=True)

#########
if __name__ == '__main__':
    app.run(debug=True)
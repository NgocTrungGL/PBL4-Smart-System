import os
import shutil
import cv2
from flask import Flask, flash, render_template, request, redirect, url_for
import mysql.connector
import numpy as np

from flask import Flask, render_template
import os
import numpy as np
import cv2 as cv
import pickle
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
# from sklearn.base import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = os.urandom(24)

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
    cursor.execute("SELECT id, name, role FROM user")
    users = cursor.fetchall()
    conn.close()

    return render_template('usermanagement.html', users=users)

def save_image(folder_name, image_base64, count):
    # Tạo đường dẫn thư mục
    dataset_path = r"D:\Code\PBL4-Smart-System\dataset"

    folder_path = os.path.join(dataset_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

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
    
# Luu thay doi
@app.route('/save_changes')
def save_changes():
    try:
        dataset_dir = "../dataset"
        target_size = (160, 160)
        detector = MTCNN()
        embedder = FaceNet()

        def extract_face(img_path):
            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            x, y, w, h = detector.detect_faces(img)[0]['box']
            x, y = abs(x), abs(y)
            face = img[y:y+h, x:x+w]
            return cv.resize(face, target_size)

        X, Y = [], []
        for sub_dir in os.listdir(dataset_dir):
            sub_path = os.path.join(dataset_dir, sub_dir)
            if os.path.isdir(sub_path):
                for img_name in os.listdir(sub_path):
                    img_path = os.path.join(sub_path, img_name)
                    try:
                        face = extract_face(img_path)
                        X.append(face)
                        Y.append(sub_dir)
                    except Exception:
                        pass

        X = np.asarray(X)
        Y = np.asarray(Y)

        def get_embedding(face_img):
            face_img = face_img.astype('float32')
            face_img = np.expand_dims(face_img, axis=0)
            return embedder.embeddings(face_img)[0]

        EMBEDDED_X = np.array([get_embedding(face) for face in X])
        np.savez_compressed('faces_embeddings_done_4classes.npz', EMBEDDED_X, Y)

        encoder = LabelEncoder()
        Y = encoder.fit_transform(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, test_size=0.2, random_state=17)
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, Y_train)
        
        with open('svm_model_160x160.pkl', 'wb') as f:
            pickle.dump(model, f)

        return redirect(url_for('user_management'))

    except Exception as e:
        print(f"Error occurred: {e}")
        return redirect(url_for('user_management'))


@app.route('/delete_user', methods=['POST'])
def delete_user():
    user_id = request.form.get('user_id')
    if not user_id:
        flash("Không thể xóa: Thiếu thông tin ID người dùng.", "error")
        return redirect(url_for('user_management'))

    try:
        conn = get_database_connection()
        cursor = conn.cursor(dictionary=True)  # Trả kết quả dưới dạng dictionary

        # Lấy thông tin tên người dùng từ cơ sở dữ liệu
        cursor.execute("SELECT name FROM user WHERE id = %s", (user_id,))
        user = cursor.fetchone()

        if not user:
            flash("Người dùng không tồn tại.", "error")
            return redirect(url_for('user_management'))

        user_name = user['name']

        # Xóa người dùng trong cơ sở dữ liệu
        cursor.execute("DELETE FROM user WHERE id = %s", (user_id,))
        conn.commit()

        # Đóng kết nối cơ sở dữ liệu
        cursor.close()
        conn.close()

        # Xóa thư mục tên người dùng
        dataset_path = r"D:\Code\PBL4-Smart-System\dataset"
        folder_path = os.path.join(dataset_path, user_name)
        print(f"Đường dẫn cần xóa: {folder_path}")

        if os.path.exists(folder_path):
            def remove_readonly(func, path, _):
                """Xóa quyền chỉ đọc nếu cần."""
                os.chmod(path, 0o777)
                func(path)

            shutil.rmtree(folder_path, onerror=remove_readonly)
            flash(f"Xóa người dùng và thư mục '{user_name}' thành công.", "success")
        else:
            flash(f"Người dùng đã được xóa. Thư mục '{user_name}' không tồn tại.", "warning")

    except Exception as e:
        flash(f"Lỗi khi xóa người dùng: {e}", "error")
        if 'conn' in locals():
            conn.close()  # Đảm bảo kết nối được đóng trong trường hợp lỗi

    return redirect(url_for('user_management'))

# Giao diện thêm khuôn mặt
@app.route('/add_face')
def add_face():
    return render_template('add_face.html')

@app.route('/save_snapshot', methods=['POST'])
def save_snapshot():
    data = request.json
    folder_name = data.get('folder_name')  # Sử dụng .get() để xử lý trường hợp thiếu khóa
    image_base64 = data.get('image')

    if not folder_name or not image_base64:
        return "Dữ liệu không hợp lệ.", 400

    # Đường dẫn tới thư mục lưu dataset
    dataset_path = r"D:\Code\PBL4-Smart-System\dataset"
    folder_path = os.path.join(dataset_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

    # Đếm số lượng ảnh đã lưu
    count = len(os.listdir(folder_path)) + 1 if os.path.exists(folder_path) else 1

    # Lưu ảnh nếu chưa đạt giới hạn
    if count <= 10:  # Giới hạn lưu 10 ảnh
        # Lưu ảnh với tên tương ứng số thứ tự
        try:
            image_data = base64.b64decode(image_base64.split(',')[1])
            image_filename = f"{count}.jpg"
            image_path = os.path.join(folder_path, image_filename)

            with open(image_path, 'wb') as image_file:
                image_file.write(image_data)

            # Ghi thông tin người dùng vào cơ sở dữ liệu nếu là ảnh đầu tiên
            if count == 1:
                try:
                    conn = get_database_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO user (name, role) VALUES (%s, %s)",
                        (folder_name, "user")
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    return f"Lỗi ghi vào cơ sở dữ liệu: {str(e)}", 500

            return f"Đã lưu ảnh {image_filename} thành công."
        except Exception as e:
            return f"Lỗi khi lưu ảnh: {str(e)}", 500
    else:
        return "Đã đủ 10 ảnh, không lưu thêm."
    
    

if __name__ == '__main__':
    app.run(debug=True)

#########
if __name__ == '__main__':
    app.run(debug=True)
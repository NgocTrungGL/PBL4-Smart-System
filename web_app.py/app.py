from flask import Flask, render_template
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

@app.route('/usermanagement')
def user_management():
    # Lấy dữ liệu từ database
    conn = get_database_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT user_id, user_name, email, sdt, role, created_at FROM Users")
    users = cursor.fetchall()
    conn.close()

    # Trả dữ liệu vào template
    return render_template('usermanagement.html', users=users)

if __name__ == '__main__':
    app.run(debug=True)

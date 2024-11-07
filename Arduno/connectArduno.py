import serial
import time
import serial.tools.list_ports

# Liệt kê các cổng COM hiện có
ports = serial.tools.list_ports.comports()
for port in ports:
    print(port)

# Kết nối với Arduino qua cổng Serial (thay 'COM3' bằng cổng Serial của bạn)
try:
    ser = serial.Serial('COM3', 9600)  # Chọn tốc độ baud tương ứng với Arduino (9600)
    time.sleep(2)  # Chờ một chút để Arduino khởi động
except serial.SerialException as e:
    print(f"Lỗi mở cổng: {e}")
    exit(1)

def send_signal_to_microcontroller(detected):
    if detected:
        ser.write(b'1')  # Gửi giá trị '1' khi nhận diện khuôn mặt
        print("Đã gửi tín hiệu: 1")
    else:
        ser.write(b'0')  # Gửi giá trị '0' nếu không nhận diện
        print("Đã gửi tín hiệu: 0")
    time.sleep(1)  # Dừng 1 giây để đảm bảo tín hiệu được truyền đi

# Ví dụ sử dụng hàm khi phát hiện khuôn mặt
def input_bool(prompt="Nhập True hoặc False: "):
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in ["true", "false"]:
            return user_input == "true"
        else:
            print("Vui lòng nhập 'True' hoặc 'False'.")

# Sử dụng hàm
gia_tri = input_bool()

face_detected = gia_tri  # Đặt thành True nếu nhận diện được khuôn mặt
send_signal_to_microcontroller(face_detected)

# Đóng kết nối khi không cần thiết
ser.close()

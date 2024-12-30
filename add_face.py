import cv2
import os
import time

def capture_images(folder_name):
    # Tạo đường dẫn thư mục lưu ảnh
    dataset_path = "./dataset/"
    folder_path = os.path.join(dataset_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Khởi tạo camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera. Vui lòng kiểm tra kết nối.")
        return

    print(f"Bắt đầu chụp ảnh. Ảnh sẽ được lưu trong thư mục: {folder_path}")

    captured_count = 0
    start_time = time.time()
    while captured_count < 10:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc dữ liệu từ camera.")
            break

        # Hiển thị ảnh từ camera
        cv2.imshow("Chụp Ảnh", frame)

        # Tính thời gian chụp ảnh
        elapsed_time = time.time() - start_time
        if elapsed_time >= (captured_count + 1) * 0.4:  # Chụp mỗi 0.4 giây
            img_path = os.path.join(folder_path, f"{captured_count+1}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Đã lưu ảnh: {img_path}")
            captured_count += 1

        # Thoát nếu người dùng nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Đã hủy chụp ảnh.")
            break

    # Giải phóng camera và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()

    print("Quá trình chụp ảnh hoàn tất.")

if __name__ == "__main__":
    folder_name = input("Nhập tên thư mục để lưu ảnh: ")
    capture_images(folder_name)

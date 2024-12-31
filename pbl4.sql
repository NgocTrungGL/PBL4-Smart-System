-- Tạo cơ sở dữ liệu 

CREATE DATABASE SmartLockSystem; 

-- Sử dụng cơ sở dữ liệu vừa tạo 

USE SmartLockSystem; 

-- Tạo bảng Users 

-- CREATE TABLE Users ( 
--     user_id INT AUTO_INCREMENT PRIMARY KEY, 
--     user_name VARCHAR (100) NOT NULL, 
--    `password` VARCHAR(255) NOT NULL, 
--     email VARCHAR (100) NOT NULL UNIQUE, 
--     sdt VARCHAR(10) NOT NULL UNIQUE, 
--     role VARCHAR (50) NOT NULL, 
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
-- );

CREATE TABLE Admin (
    AdminID INT AUTO_INCREMENT PRIMARY KEY, -- ID duy nhất cho mỗi admin
    Username VARCHAR(50) NOT NULL UNIQUE,   -- Tên đăng nhập của admin
    PasswordHash VARCHAR(255) NOT NULL,     -- Mật khẩu được lưu dưới dạng mã băm
    Email VARCHAR(100) NOT NULL UNIQUE,     -- Email liên hệ của admin
);

CREATE TABLE LichSuDangNhap (
    LanDangNhap INT AUTO_INCREMENT PRIMARY KEY,    -- Lần đăng nhập thứ n, là khóa chính
    Ten NVARCHAR(100) NOT NULL,                    -- Tên người dùng
    ThoiDiemDangNhap TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Thời điểm đăng nhập
    Anh MEDIUMBLOB NULL                                  -- Cột ảnh (Lưu trữ hình ảnh ở dạng nhị phân)
);
-- Tạo cơ sở dữ liệu 

CREATE DATABASE SmartLockSystem; 

-- Sử dụng cơ sở dữ liệu vừa tạo 

USE SmartLockSystem; 

-- Tạo bảng Users 

CREATE TABLE Users ( 
    user_id INT AUTO_INCREMENT PRIMARY KEY, 
    user_name VARCHAR (100) NOT NULL, 
   `password` VARCHAR(255) NOT NULL, 
    email VARCHAR (100) NOT NULL UNIQUE, 
    sdt VARCHAR(10) NOT NULL UNIQUE, 
    role VARCHAR (50) NOT NULL, 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
);  
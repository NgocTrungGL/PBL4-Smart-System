const int ledPin = 13; // Chân kết nối LED (trên board Arduino là chân 13)

void setup() {
  Serial.begin(9600); // Thiết lập tốc độ baud cho Serial
  pinMode(ledPin, OUTPUT); // Đặt chân LED làm OUTPUT
}

void loop() {
  if (Serial.available() > 0) { // Kiểm tra có tín hiệu từ Serial
    char received = Serial.read(); // Đọc tín hiệu từ Python
    Serial.print("Nhận được: "); // In tín hiệu nhận được
    Serial.println(received); // In tín hiệu nhận được

    if (received == '1') {
      digitalWrite(ledPin, HIGH); // Bật LED khi nhận '1'
      Serial.println("Đèn đã bật!");
      
      delay(5000); // Giữ LED sáng trong 5 giây

      digitalWrite(ledPin, LOW); // Tắt LED sau 5 giây
      Serial.println("Đèn đã tắt!");
    } else if (received == '0') {
      digitalWrite(ledPin, LOW); // Tắt LED khi nhận '0'
      Serial.println("Đèn đã tắt!");
    }
  }
}

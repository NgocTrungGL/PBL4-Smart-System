const int relayPin = 8; 
const int ledPin = 13;
const int ledPin2 = 12;
void setup() {
  Serial.begin(9600); 
  pinMode(relayPin, OUTPUT);
  pinMode(ledPin, OUTPUT);
  pinMode(ledPin2, OUTPUT);
  digitalWrite(relayPin, LOW); 
}

void loop() {
  if (Serial.available() > 0) { // Kiểm tra có tín hiệu từ Serial
    char received = Serial.read(); // Đọc tín hiệu từ Python
    Serial.print("Nhận được: "); // In tín hiệu nhận được
    Serial.println(received); // In tín hiệu nhận được

    if (received == '1') {
      digitalWrite(relayPin, HIGH); 
      Serial.println("Khóa bật!");
      digitalWrite(ledPin, HIGH);
      delay(3000); // Giữ LED sáng trong 5 giây

      digitalWrite(relayPin, LOW);  // Tắt LED sau 5 giây
      digitalWrite(ledPin, LOW);
      Serial.println("Khóa đóng!");
    } else if (received == '0') {
      digitalWrite(relayPin, LOW); // Tắt LED khi nhận '0'
      Serial.println("Nhận diện không được!");
      digitalWrite(ledPin2, HIGH);
      delay(3000);
      digitalWrite(ledPin2, LOW);
      
    }
  }
}

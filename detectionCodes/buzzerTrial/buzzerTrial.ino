#define BUZZER_PIN 26

void setup() {
  Serial.begin(115200);

  pinMode(BUZZER_PIN, OUTPUT);

  // OFF for low-level trigger
  digitalWrite(BUZZER_PIN, HIGH);

  Serial.println("Buzzer test");
}

void loop() {
  digitalWrite(BUZZER_PIN, LOW);   // ON
  Serial.println("BUZZER ON");
  delay(2000);

  digitalWrite(BUZZER_PIN, HIGH);  // OFF
  Serial.println("BUZZER OFF");
  delay(2000);
}
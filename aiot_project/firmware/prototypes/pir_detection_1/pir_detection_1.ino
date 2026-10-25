// ESP32 - Dual PIR Sensor Test

#define PIR1_PIN 27

#define PIR2_PIN 25

void setup() {

  Serial.begin(115200);

  pinMode(PIR1_PIN, INPUT);

  pinMode(PIR2_PIN, INPUT);

  Serial.println("Dual PIR system starting...");

  delay(3000);  // Give PIR sensors time to stabilize

  Serial.println("Ready!");

}

void loop() {

  int pir1State = digitalRead(PIR1_PIN);

  int pir2State = digitalRead(PIR2_PIN);

  // PIR 1

  if (pir1State == HIGH) {

    Serial.println("PIR 1: MOTION DETECTED");

  } else {

    Serial.println("PIR 1: No motion");

  }

  // PIR 2

  if (pir2State == HIGH) {

    Serial.println("PIR 2: MOTION DETECTED");

  } else {

    Serial.println("PIR 2: No motion");

  }

  Serial.println("-------------------------");

  delay(500);

}
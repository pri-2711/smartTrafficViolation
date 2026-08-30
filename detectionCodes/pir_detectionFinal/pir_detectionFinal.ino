#define PIR1_PIN 27
#define PIR2_PIN 25

void setup() {
  Serial.begin(115200);

  pinMode(PIR1_PIN, INPUT);
  pinMode(PIR2_PIN, INPUT);

  Serial.println("PIR TEST STARTING...");
  Serial.println("Wait 30 seconds for PIR warm-up.");

  delay(30000);

  Serial.println("READY!");
}

void loop() {

  int pir1State = digitalRead(PIR1_PIN);
  int pir2State = digitalRead(PIR2_PIN);

  Serial.print("PIR 1: ");
  Serial.print(pir1State == HIGH ? "MOTION" : "NO MOTION");

  Serial.print("    |    ");

  Serial.print("PIR 2: ");
  Serial.println(pir2State == HIGH ? "MOTION" : "NO MOTION");

  delay(1000);
}
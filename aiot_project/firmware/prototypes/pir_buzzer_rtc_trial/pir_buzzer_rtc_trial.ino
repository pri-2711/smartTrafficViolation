#include <Wire.h>
#include <RTClib.h>

#define PIR1_PIN 27
#define PIR2_PIN 25
#define BUZZER_PIN 26

RTC_DS3231 rtc;

bool alarmActive = false;
unsigned long alarmStart = 0;

const unsigned long BUZZER_TIME = 3000;

void printTime(DateTime now) {
  Serial.print(now.day());
  Serial.print("/");
  Serial.print(now.month());
  Serial.print("/");
  Serial.print(now.year());
  Serial.print(" ");

  if (now.hour() < 10) Serial.print("0");
  Serial.print(now.hour());
  Serial.print(":");

  if (now.minute() < 10) Serial.print("0");
  Serial.print(now.minute());
  Serial.print(":");

  if (now.second() < 10) Serial.print("0");
  Serial.println(now.second());
}

void setup() {

  Serial.begin(115200);

  pinMode(PIR1_PIN, INPUT);
  pinMode(PIR2_PIN, INPUT);

  // LOW-level trigger buzzer
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, HIGH);   // OFF

  // DS3231
  Wire.begin(21, 22);

  if (!rtc.begin()) {
    Serial.println("ERROR: DS3231 NOT FOUND!");
    while (1);
  }

  // Only reset RTC if backup battery was lost
  if (rtc.lostPower()) {
    Serial.println("RTC lost power.");
    Serial.println("Setting RTC...");
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  }

  Serial.println();
  Serial.println("==============================");
  Serial.println("PIR + BUZZER + RTC TEST");
  Serial.println("==============================");

  Serial.println("Wait 30 seconds for PIR warm-up...");
  delay(30000);

  Serial.println("SYSTEM READY");
}

void loop() {

  int pir1State = digitalRead(PIR1_PIN);
  int pir2State = digitalRead(PIR2_PIN);

  // --------------------------------
  // CONTINUOUS PIR STATUS
  // --------------------------------

  Serial.print("PIR 1 (GPIO 27): ");
  Serial.print(pir1State == HIGH ? "MOTION" : "NO MOTION");

  Serial.print("    |    ");

  Serial.print("PIR 2 (GPIO 25): ");
  Serial.println(pir2State == HIGH ? "MOTION" : "NO MOTION");


  // --------------------------------
  // START BUZZER
  // --------------------------------

  if ((pir1State == HIGH || pir2State == HIGH) && !alarmActive) {

    alarmActive = true;
    alarmStart = millis();

    // Buzzer ON
    digitalWrite(BUZZER_PIN, LOW);

    DateTime now = rtc.now();

    Serial.print(">>> MOTION DETECTED | ");
    printTime(now);

    Serial.println(">>> BUZZER ON");
  }


  // --------------------------------
  // TURN BUZZER OFF AFTER 3 SEC
  // --------------------------------

  if (alarmActive && millis() - alarmStart >= BUZZER_TIME) {

    digitalWrite(BUZZER_PIN, HIGH);

    alarmActive = false;

    Serial.println(">>> BUZZER OFF");
  }

  delay(1000);
}
#include <Wire.h>
#include <RTClib.h>

RTC_DS3231 rtc;

void setup() {
  Serial.begin(115200);

  Wire.begin(21, 22);

  if (!rtc.begin()) {
    Serial.println("ERROR: DS3231 not found!");
    while (1);
  }

  rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));

  Serial.println("DS3231 detected!");
}

void loop() {
  DateTime now = rtc.now();

  Serial.print(now.day());
  Serial.print("/");
  Serial.print(now.month());
  Serial.print("/");
  Serial.print(now.year());

  Serial.print("  ");

  if (now.hour() < 10) Serial.print("0");
  Serial.print(now.hour());
  Serial.print(":");

  if (now.minute() < 10) Serial.print("0");
  Serial.print(now.minute());
  Serial.print(":");

  if (now.second() < 10) Serial.print("0");
  Serial.println(now.second());

  delay(1000);
}
#include <Wire.h>
#include <RTClib.h>
#include <WiFi.h>

// =====================================================
// PINS
// =====================================================

#define PIR1_PIN 27
#define PIR2_PIN 25
#define BUZZER_PIN 26
#define TRIG_PIN 33
#define ECHO_PIN 35

// =====================================================
// WIFI SETTINGS
// =====================================================

const char* ssid = "27";
const char* password = "987654321";

// IMPORTANT:
// Put the IP address of YOUR PC here.
// Find it using: ipconfig
const char* PYTHON_SERVER_IP = "10.136.145.230";

const int PYTHON_SERVER_PORT = 5000;

// =====================================================
// CAMERA SETTINGS
// =====================================================

// This is only printed for information.
// Python itself knows the camera IP.
const char* CAMERA_IP = "10.136.145.144";

// =====================================================
// CAMERA TRIGGER SETTINGS
// =====================================================

// Delay between overspeed detection and camera trigger
const unsigned long CAMERA_DELAY_MS = 1000;

// =====================================================
// RTC
// =====================================================

RTC_DS3231 rtc;

// =====================================================
// SPEED SETTINGS
// =====================================================

const float OVERSPEED_THRESHOLD_KMPH = 20.0;

// =====================================================
// TIMING
// =====================================================

const unsigned long SAMPLE_INTERVAL_MS = 150;
const unsigned long BUZZER_TIME = 3000;
const unsigned long COOLDOWN_TIME = 5000;

// =====================================================
// SYSTEM STATE
// =====================================================

enum SystemState {
  IDLE,
  MEASURING,
  ALARM
};

SystemState state = IDLE;

// =====================================================
// VARIABLES
// =====================================================

unsigned long lastSampleTime = 0;

float lastDistance = 0;
bool haveFirstSample = false;

unsigned long alarmStart = 0;

bool inCooldown = false;
unsigned long cooldownStart = 0;

// =====================================================
// PRINT RTC TIME
// =====================================================

void printTime(DateTime now) {

  Serial.print(now.day());
  Serial.print("/");

  Serial.print(now.month());
  Serial.print("/");

  Serial.print(now.year());
  Serial.print(" ");

  if (now.hour() < 10)
    Serial.print("0");

  Serial.print(now.hour());
  Serial.print(":");

  if (now.minute() < 10)
    Serial.print("0");

  Serial.print(now.minute());
  Serial.print(":");

  if (now.second() < 10)
    Serial.print("0");

  Serial.println(now.second());
}

// =====================================================
// READ ULTRASONIC DISTANCE
// =====================================================

float readDistanceCM() {

  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);

  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);

  if (duration == 0)
    return -1;

  return duration * 0.0343 / 2;
}

// =====================================================
// SEND CAMERA TRIGGER TO PYTHON
// =====================================================

void triggerCamera() {

  Serial.println();
  Serial.println(">>> Sending camera trigger to Python...");

  WiFiClient client;

  if (!client.connect(PYTHON_SERVER_IP, PYTHON_SERVER_PORT)) {

    Serial.println("ERROR: Could not connect to Python server!");

    return;
  }

  DateTime now = rtc.now();

  char message[100];

  sprintf(
    message,
    "CAPTURE|%02d_%02d_%04d|%02d_%02d_%02d",
    now.day(),
    now.month(),
    now.year(),
    now.hour(),
    now.minute(),
    now.second()
  );

  client.print(message);

  Serial.print("Sent: ");
  Serial.println(message);

  client.stop();

  Serial.println(">>> Camera trigger sent.");
}

// =====================================================
// WIFI CONNECTION
// =====================================================

void connectWiFi() {

  Serial.println();
  Serial.println("Connecting to Wi-Fi...");

  WiFi.mode(WIFI_STA);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {

    delay(500);

    Serial.print(".");
  }

  Serial.println();
  Serial.println("Wi-Fi connected!");

  Serial.print("DevKit IP address: ");
  Serial.println(WiFi.localIP());

  Serial.print("Python server IP: ");
  Serial.println(PYTHON_SERVER_IP);

  Serial.print("Python server port: ");
  Serial.println(PYTHON_SERVER_PORT);

  Serial.print("Camera IP: ");
  Serial.println(CAMERA_IP);

  Serial.println();
}

// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial.begin(115200);

  delay(1000);

  Serial.println();
  Serial.println("==============================");
  Serial.println("TRAFFIC VIOLATION DETECTION");
  Serial.println("PIR / ULTRASONIC / BUZZER / RTC");
  Serial.println("==============================");

  // ---------------------------------------------------
  // PIR
  // ---------------------------------------------------

  pinMode(PIR1_PIN, INPUT);
  pinMode(PIR2_PIN, INPUT);

  // ---------------------------------------------------
  // BUZZER
  // ---------------------------------------------------

  pinMode(BUZZER_PIN, OUTPUT);

  // LOW-trigger buzzer
  // HIGH = OFF
  digitalWrite(BUZZER_PIN, HIGH);

  // ---------------------------------------------------
  // ULTRASONIC
  // ---------------------------------------------------

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  digitalWrite(TRIG_PIN, LOW);

  // ---------------------------------------------------
  // RTC
  // ---------------------------------------------------

  Wire.begin(21, 22);

  if (!rtc.begin()) {

    Serial.println("ERROR: DS3231 NOT FOUND!");

    while (1);
  }

  if (rtc.lostPower()) {

    Serial.println("RTC lost power. Setting RTC...");

    rtc.adjust(
      DateTime(
        F(__DATE__),
        F(__TIME__)
      )
    );
  }

  // ---------------------------------------------------
  // WIFI
  // ---------------------------------------------------

  connectWiFi();

  // ---------------------------------------------------
  // PIR WARM-UP
  // ---------------------------------------------------

  Serial.println("Wait 30 seconds for PIR warm-up...");

  delay(30000);

  Serial.println();
  Serial.println("==============================");
  Serial.println("SYSTEM READY");
  Serial.println("==============================");
}

// =====================================================
// LOOP
// =====================================================

void loop() {

  unsigned long nowMs = millis();

  // ===================================================
  // COOLDOWN
  // ===================================================

  if (inCooldown) {

    if (nowMs - cooldownStart >= COOLDOWN_TIME) {

      inCooldown = false;

      Serial.println(">>> Cooldown over, re-armed");

    } else {

      delay(50);

      return;
    }
  }

  // ===================================================
  // BUZZER AUTO-OFF
  // ===================================================

  if (
    state == ALARM &&
    nowMs - alarmStart >= BUZZER_TIME
  ) {

    digitalWrite(BUZZER_PIN, HIGH);

    state = IDLE;

    inCooldown = true;

    cooldownStart = nowMs;

    haveFirstSample = false;

    Serial.println(">>> BUZZER OFF, entering cooldown");

    return;
  }

  // ===================================================
  // READ PIR
  // ===================================================

  int pir1State = digitalRead(PIR1_PIN);
  int pir2State = digitalRead(PIR2_PIN);

  bool motionDetected =
    (pir1State == HIGH || pir2State == HIGH);

  // ===================================================
  // IDLE
  // ===================================================

  if (state == IDLE) {

    if (motionDetected) {

      state = MEASURING;

      haveFirstSample = false;

      Serial.println(
        ">>> Motion detected - starting speed measurement"
      );
    }

    delay(100);

    return;
  }

  // ===================================================
  // MEASURING
  // ===================================================

  if (state == MEASURING) {

    if (!motionDetected) {

      state = IDLE;

      haveFirstSample = false;

      Serial.println(
        ">>> Motion ended - no overspeed - resetting"
      );

      delay(100);

      return;
    }

    // -------------------------------------------------
    // ULTRASONIC SAMPLE
    // -------------------------------------------------

    if (
      nowMs - lastSampleTime >= SAMPLE_INTERVAL_MS
    ) {

      float d = readDistanceCM();

      unsigned long sampleTime = nowMs;

      if (d > 0) {

        if (haveFirstSample) {

          float deltaDistanceCM =
            lastDistance - d;

          float deltaTimeMs =
            sampleTime - lastSampleTime;

          if (
            deltaDistanceCM > 0 &&
            deltaTimeMs > 0
          ) {

            float speedKMPH =
              (deltaDistanceCM / deltaTimeMs) * 36.0;

            Serial.print("Distance: ");
            Serial.print(d);

            Serial.print(" cm | Speed: ");
            Serial.print(speedKMPH);

            Serial.println(" km/h");

            // =========================================
            // OVER SPEED DETECTED
            // =========================================

            if (
              speedKMPH > OVERSPEED_THRESHOLD_KMPH
            ) {

              DateTime now = rtc.now();

              Serial.println();
              Serial.print(
                ">>> OVERSPEEDING DETECTED: "
              );

              Serial.print(speedKMPH);

              Serial.print(" km/h | ");

              printTime(now);

              // ---------------------------------------
              // BUZZER ON
              // ---------------------------------------

              digitalWrite(BUZZER_PIN, LOW);

              state = ALARM;

              alarmStart = nowMs;

              // ---------------------------------------
              // WAIT BEFORE CAMERA CAPTURE
              // ---------------------------------------

              Serial.print(
                ">>> Waiting "
              );

              Serial.print(
                CAMERA_DELAY_MS
              );

              Serial.println(
                " ms before camera trigger..."
              );

              delay(CAMERA_DELAY_MS);

              // ---------------------------------------
              // TRIGGER PYTHON
              // ---------------------------------------

              triggerCamera();
            }
          }
        }

        lastDistance = d;

        lastSampleTime = sampleTime;

        haveFirstSample = true;
      }
    }
  }
}